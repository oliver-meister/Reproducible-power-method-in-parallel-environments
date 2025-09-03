#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

// Keep your original test includes (harmless if unused)
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>

#include "matrix.h"           // assumes sparseMatrixCOO / sparseMatrixCSR are declared here
#include "../external/mmio.h"  // MatrixMarket I/O (mmio.h/.c from NIST)

#if defined(USE_CUDA) || defined(USE_EXBLAS)
  #include <cuda_runtime.h>
#endif

// -------------------------------
// helpers & safety
// -------------------------------
#define DIE(...) do { \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    exit(EXIT_FAILURE); \
} while (0)

static inline void* xmalloc(size_t nbytes) {
    void* p = malloc(nbytes);
    if (!p && nbytes) DIE("malloc failed (%zu bytes)", nbytes);
    return p;
}

static inline void* xcalloc(size_t n, size_t s) {
    void* p = calloc(n, s);
    if (!p && n && s) DIE("calloc failed (%zu x %zu)", n, s);
    return p;
}

#if defined(USE_CUDA) || defined(USE_EXBLAS)
static inline void checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        DIE("CUDA error in %s: %s", what, cudaGetErrorString(e));
    }
}
#endif

// Pair struct for sorting/dedup within CSR rows
struct ColVal { int c; double v; };

static int cmp_colval(const void* a, const void* b) {
    int ca = ((const struct ColVal*)a)->c;
    int cb = ((const struct ColVal*)b)->c;
    return (ca > cb) - (ca < cb);
}

// -------------------------------
// Public API
// -------------------------------

sparseMatrixCOO* createSparseMatrixCOO(const char* file) {
    FILE* f = fopen(file, "r");
    if (!f) {
        fprintf(stderr, "Cannot open matrix file '%s': %s\n", file, strerror(errno));
        exit(EXIT_FAILURE);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fclose(f);
        DIE("mm_read_banner failed");
    }
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fclose(f);
        DIE("Only MatrixMarket coordinate (sparse) format is supported");
    }
    if (mm_is_hermitian(matcode) || mm_is_complex(matcode)) {
        fclose(f);
        DIE("Hermitian/complex matrices are not supported by this reader");
    }

    int rows = 0, cols = 0, nnz = 0;
    if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
        fclose(f);
        DIE("mm_read_mtx_crd_size failed");
    }

    const bool is_sym  = mm_is_symmetric(matcode);
    const bool is_skew = mm_is_skew(matcode);
    const bool is_pat  = mm_is_pattern(matcode);
    const bool is_real_or_int = mm_is_real(matcode) || mm_is_integer(matcode);

    if (!is_pat && !is_real_or_int) {
        fclose(f);
        DIE("Unsupported numeric type in MatrixMarket file");
    }

    size_t max_nnz = (size_t)nnz * ((is_sym || is_skew) ? 2u : 1u);
    int*    row = (int*)   xmalloc(sizeof(int)    * max_nnz);
    int*    col = (int*)   xmalloc(sizeof(int)    * max_nnz);
    double* val = (double*)xmalloc(sizeof(double) * max_nnz);

    size_t cur = 0;
    for (int k = 0; k < nnz; ++k) {
        int r, c; double v = 1.0;
        if (is_pat) {
            if (fscanf(f, "%d %d", &r, &c) != 2) { fclose(f); DIE("read error (pattern)"); }
        } else {
            if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) { fclose(f); DIE("read error (real/int)"); }
        }
        // convert to 0-based
        r--; c--;
        if (r < 0 || r >= rows || c < 0 || c >= cols) { fclose(f); DIE("entry out of bounds"); }

        row[cur] = r; col[cur] = c; val[cur] = v; cur++;

        if ((is_sym || is_skew) && r != c) {
            row[cur] = c; col[cur] = r; val[cur] = is_skew ? -v : v; cur++;
        }
    }
    fclose(f);

    sparseMatrixCOO* A = (sparseMatrixCOO*)xmalloc(sizeof(*A));
    A->row  = row;
    A->col  = col;
    A->val  = val;
    A->rows = rows;
    A->cols = cols;
    A->nnz  = (int)cur; // note: trunc ok for typical SSMC sizes using int
    A->d_row = NULL; A->d_col = NULL; A->d_val = NULL;

#if defined(USE_CUDA) || defined(USE_EXBLAS)
    checkCuda(cudaMalloc((void**)&A->d_row, sizeof(int) * (size_t)A->nnz), "cudaMalloc d_row");
    checkCuda(cudaMalloc((void**)&A->d_col, sizeof(int) * (size_t)A->nnz), "cudaMalloc d_col");
    checkCuda(cudaMalloc((void**)&A->d_val, sizeof(double) * (size_t)A->nnz), "cudaMalloc d_val");

    checkCuda(cudaMemcpy(A->d_row, A->row, sizeof(int) * (size_t)A->nnz, cudaMemcpyHostToDevice), "cudaMemcpy d_row");
    checkCuda(cudaMemcpy(A->d_col, A->col, sizeof(int) * (size_t)A->nnz, cudaMemcpyHostToDevice), "cudaMemcpy d_col");
    checkCuda(cudaMemcpy(A->d_val, A->val, sizeof(double) * (size_t)A->nnz, cudaMemcpyHostToDevice), "cudaMemcpy d_val");
#endif

    return A;
}

// Convert COO -> CSR, then sort columns within each row and sum duplicates.
sparseMatrixCSR* coo_to_csr(sparseMatrixCOO* coo) {
    if (!coo) DIE("coo_to_csr: coo == NULL");

    sparseMatrixCSR* csr = (sparseMatrixCSR*)xmalloc(sizeof(*csr));
    csr->rows = coo->rows;
    csr->cols = coo->cols;
    csr->nnz  = coo->nnz; // provisional; may shrink after dedup

    csr->row_ptr = (int*)   xcalloc((size_t)coo->rows + 1u, sizeof(int));
    csr->col     = (int*)   xmalloc(sizeof(int)    * (size_t)coo->nnz);
    csr->val     = (double*)xmalloc(sizeof(double) * (size_t)coo->nnz);

    // 1) histogram rows
    for (int k = 0; k < coo->nnz; ++k) {
        int r = coo->row[k];
        if (r < 0 || r >= coo->rows) DIE("COO row out of range");
        csr->row_ptr[r + 1]++;
    }
    // 2) prefix sum
    for (int r = 0; r < coo->rows; ++r) {
        csr->row_ptr[r + 1] += csr->row_ptr[r];
    }

    // 3) scatter into CSR (preserves input order per row)
    int* next = (int*)xmalloc(sizeof(int) * (size_t)coo->rows);
    memcpy(next, csr->row_ptr, sizeof(int) * (size_t)coo->rows);

    for (int k = 0; k < coo->nnz; ++k) {
        int r = coo->row[k];
        int dst = next[r]++;
        csr->col[dst] = coo->col[k];
        csr->val[dst] = coo->val[k];
    }
    free(next);

    // 4) sort each row by column and sum duplicates
    // We'll compact in-place in a second pass.
    for (int r = 0; r < csr->rows; ++r) {
        int start = csr->row_ptr[r];
        int end   = csr->row_ptr[r + 1];
        int len   = end - start;
        if (len <= 1) continue;

        struct ColVal* tmp = (struct ColVal*)xmalloc(sizeof(struct ColVal) * (size_t)len);
        for (int i = 0; i < len; ++i) { tmp[i].c = csr->col[start + i]; tmp[i].v = csr->val[start + i]; }
        qsort(tmp, (size_t)len, sizeof(struct ColVal), cmp_colval);

        int w = 0;
        for (int i = 0; i < len; ) {
            int c = tmp[i].c; double s = tmp[i].v; i++;
            while (i < len && tmp[i].c == c) { s += tmp[i].v; i++; }
            csr->col[start + w] = c;
            csr->val[start + w] = s;
            w++;
        }
        // mark the remainder as empty using a sentinel col = -1
        for (int i = w; i < len; ++i) { csr->col[start + i] = -1; csr->val[start + i] = 0.0; }
        free(tmp);
    }

    // 5) global compaction to remove sentinels and rebuild row_ptr
    int write = 0;
    for (int r = 0; r < csr->rows; ++r) {
        int old_start = csr->row_ptr[r];
        int old_end   = csr->row_ptr[r + 1];
        csr->row_ptr[r] = write;
        for (int i = old_start; i < old_end; ++i) {
            if (csr->col[i] >= 0) {
                csr->col[write] = csr->col[i];
                csr->val[write] = csr->val[i];
                write++;
            }
        }
    }
    csr->row_ptr[csr->rows] = write;
    csr->nnz = write;

#if defined(USE_CUDA) || defined(USE_EXBLAS)
    csr->d_row_ptr = NULL; csr->d_col = NULL; csr->d_val = NULL;
    checkCuda(cudaMalloc((void**)&csr->d_row_ptr, sizeof(int) * ((size_t)csr->rows + 1u)), "cudaMalloc d_row_ptr");
    checkCuda(cudaMalloc((void**)&csr->d_col,     sizeof(int) * (size_t)csr->nnz),        "cudaMalloc d_col");
    checkCuda(cudaMalloc((void**)&csr->d_val,     sizeof(double) * (size_t)csr->nnz),     "cudaMalloc d_val");

    checkCuda(cudaMemcpy(csr->d_row_ptr, csr->row_ptr, sizeof(int) * ((size_t)csr->rows + 1u), cudaMemcpyHostToDevice), "memcpy d_row_ptr");
    checkCuda(cudaMemcpy(csr->d_col,     csr->col,     sizeof(int)    * (size_t)csr->nnz,    cudaMemcpyHostToDevice), "memcpy d_col");
    checkCuda(cudaMemcpy(csr->d_val,     csr->val,     sizeof(double) * (size_t)csr->nnz,    cudaMemcpyHostToDevice), "memcpy d_val");
#else
    csr->d_row_ptr = NULL; csr->d_col = NULL; csr->d_val = NULL;
#endif

    return csr;
}

void delete_COO(sparseMatrixCOO* coo) {
    if (!coo) return;
    free(coo->row);
    free(coo->col);
    free(coo->val);
#if defined(USE_CUDA) || defined(USE_EXBLAS)
    if (coo->d_row) cudaFree(coo->d_row);
    if (coo->d_col) cudaFree(coo->d_col);
    if (coo->d_val) cudaFree(coo->d_val);
#endif
    free(coo);
}

void delete_CSR(sparseMatrixCSR* csr) {
    if (!csr) return;
    free(csr->row_ptr);
    free(csr->col);
    free(csr->val);
#if defined(USE_CUDA) || defined(USE_EXBLAS)
    if (csr->d_row_ptr) cudaFree(csr->d_row_ptr);
    if (csr->d_col)     cudaFree(csr->d_col);
    if (csr->d_val)     cudaFree(csr->d_val);
#endif
    free(csr);
}
