#ifndef MATRIX_H
#define MATRIX_H

// This header matches matrix_io.c (robust Matrix Market reader + COO->CSR)
// It keeps CUDA device pointers in the structs, but they are only used
// when compiled with USE_CUDA or USE_EXBLAS.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------------
// Dense Matrix (optional helper)
// -------------------------------
typedef struct {
    double* data;  // row-major, length = rows*cols
    int rows;
    int cols;
} denseMatrix;

// -------------------------------
// Sparse Matrix: COO
// -------------------------------
typedef struct {
    int*    row;  // row indices of non-zeros (length = nnz)
    int*    col;  // column indices of non-zeros (length = nnz)
    double* val;  // values of non-zeros (length = nnz)

    int rows;     // number of rows
    int cols;     // number of columns
    int nnz;      // number of stored non-zeros

    // Device copies (allocated only if USE_CUDA or USE_EXBLAS is defined)
    int*    d_row;
    int*    d_col;
    double* d_val;
} sparseMatrixCOO;

// -------------------------------
// Sparse Matrix: CSR
// -------------------------------
typedef struct {
    int*    row_ptr; // length = rows + 1
    int*    col;     // length = nnz
    double* val;     // length = nnz

    int rows;
    int cols;
    int nnz;

    // Device copies (allocated only if USE_CUDA or USE_EXBLAS is defined)
    int*    d_row_ptr;
    int*    d_col;
    double* d_val;
} sparseMatrixCSR;

// -------------------------------
// Generic sparse union/tag (optional helper)
// -------------------------------
typedef enum {
    COO = 0,
    CSR = 1
} MatrixType;

typedef union {
    sparseMatrixCOO* coo;
    sparseMatrixCSR* csr;
} MatrixUnion;

typedef struct {
    MatrixUnion mat;
    MatrixType  type;
} SparseMatrixAny;

// -------------------------------
// API
// -------------------------------
// Read a Matrix Market (.mtx) file into COO.
// - Supports coordinate sparse real/integer/pattern
// - Expands symmetric/skew-symmetric (Hermitian/complex are rejected)
// - 0-based indices in memory
sparseMatrixCOO* createSparseMatrixCOO(const char* filepath);

// Convert COO -> CSR, sort columns within each row, and sum duplicates.
sparseMatrixCSR* coo_to_csr(sparseMatrixCOO* coo);

// Destructors (free host memory; free device buffers if present)
void delete_COO(sparseMatrixCOO* coo);
void delete_CSR(sparseMatrixCSR* csr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MATRIX_H
