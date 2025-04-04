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
} sparseMatrix;

#endif