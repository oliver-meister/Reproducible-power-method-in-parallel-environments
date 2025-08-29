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
    int* d_row;
    int* d_col;
    double* d_val;
} sparseMatrixCOO;

typedef struct {
    int* row_ptr;  
    int* col;
    double* val;
    int rows;
    int cols; 
    int nnz;
    int* d_row_ptr;
    int* d_col;
    double* d_val;
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

void delete_COO(sparseMatrixCOO* coo);
void delete_CSR(sparseMatrixCSR* csr);

#endif