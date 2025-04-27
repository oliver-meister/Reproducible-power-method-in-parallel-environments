#ifndef EXBLAS_CPU_H
#define EXBLAS_CPU_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include <stdbool.h>
// Function for dense matrices.
void exblasCpu_dense_matvec_mult(const denseMatrix*, Vector*);
// Function for sparse matrices.
void exblasCpu_sparse_matvec_mult(const SparseMatrixAny*, Vector*);
// Functions for COO format
void exblasCpu_sparse_matvec_mult_COO(const sparseMatrixCOO*, Vector*);
// Functions for CSR format
void exBlasCpu_sparse_matvec_mult_CSR(const sparseMatrixCSR*, Vector*);
// Common functions.
double exBlasCpu_dot_product(const Vector*, const Vector*);

#endif