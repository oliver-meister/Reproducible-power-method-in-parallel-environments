#ifndef MATRIX_H
#define MATRIX_H

// Dense Matrix
typedef struct {
    double* data;
    int rows;
    int cols;
} denseMatrix;

// Sparse Matrix
typedef struct {
    int* row; // row indices of non-zero values
    int* col; // Column indices of non-zero values
    double* val; // The actual non-zero values
    int rows; // Total number of rows
    int cols; // Total number of columns
    int nnz; // Number of non-zero values
} sparseMatrixCOO;

typedef struct {
    int* row_ptr;  //
    int* col;
    double* val;
    int rows;
    int cols; 
    int nnz;
} sparseMatrixCSR;

typedef enum {
    COO,
    CSR
} MatrixType;

typedef union {
    sparseMatrixCOO *coo;
    sparseMatrixCSR *csr;
} MatrixUnion;

typedef struct {
    MatrixUnion mat;
    MatrixType type;
} SparseMatrixAny;

sparseMatrixCOO* createSparseMatrixCOO(char*);
sparseMatrixCSR* coo_to_csr(sparseMatrixCOO*);

#endif