#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include <stdbool.h>
#include "../external/mmio.h"


sparseMatrixCOO* createSparseMatrixCOO(char* file) {
    FILE *f;
    int *row, *col;
    double *val;
    int rows, cols, nnz;

    if ((f = fopen(file, "r")) == NULL) {
        perror("Cannot open matrix file");
        exit(1);
    }

    // Read matrix type and size
    MM_typecode matcode;
    mm_read_banner(f, &matcode);
    mm_read_mtx_crd_size(f, &rows, &cols, &nnz);

    // Determine if symmetric
    bool is_symmetric = mm_is_symmetric(matcode);

    // Allocate memory (symmetric → up to 2× nnz)
    int max_nnz = is_symmetric ? nnz * 2 : nnz;
    row = (int *) malloc(sizeof(int) * max_nnz);
    col = (int *) malloc(sizeof(int) * max_nnz);
    val = (double *) malloc(sizeof(double) * max_nnz);

    int current_nnz = 0;

    for (int i = 0; i < nnz; i++) {
        int r, c;
        double v;

        fscanf(f, "%d %d %lg", &r, &c, &v);
        r--; c--; // Convert to 0-based indexing

        // Always store the original (r, c)
        row[current_nnz] = r;
        col[current_nnz] = c;
        val[current_nnz] = v;
        current_nnz++;

        // If symmetric and off-diagonal, store (c, r)
        if (is_symmetric && r != c) {
            row[current_nnz] = c;
            col[current_nnz] = r;
            val[current_nnz] = v;
            current_nnz++;
        }
    }

    fclose(f);

    sparseMatrixCOO *A = malloc(sizeof(sparseMatrixCOO));
    A->row = row;
    A->col = col;
    A->val = val;
    A->rows = rows;
    A->cols = cols;
    A->nnz = current_nnz;  // Actual nnz used

    return A;
}


sparseMatrixCSR* coo_to_csr(sparseMatrixCOO *coo){

    sparseMatrixCSR *csr = malloc(sizeof(sparseMatrixCSR));
    csr->rows = coo->rows;
    csr->cols = coo->cols;
    csr->nnz = coo->nnz;

    csr->row_ptr = calloc(coo->rows + 1, sizeof(int));
    csr->col = malloc(sizeof(int) * coo->nnz);
    csr->val = malloc(sizeof(double) * coo->nnz); 


    for(int i = 0; i < coo->nnz; i++){
        csr->row_ptr[coo->row[i] + 1 ]++;
    }

    for (int i = 0; i < coo->rows; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    int* offset = calloc(coo->rows, sizeof(int));

    for (int i = 0; i < coo->nnz; i++) {
        int r = coo->row[i];
        int dest = csr->row_ptr[r] + offset[r]++;
        csr->col[dest] = coo->col[i];
        csr->val[dest] = coo->val[i];
    }

    free(offset);
    return csr;

}

