#ifndef OFF_H
#define OFF_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense matrices.
void off_dense_matvec_mult(const denseMatrix*, Vector*);
// Function for sparse matrices.
void off_sparse_matvec_mult(const SparseMatrixAny*, Vector*);
// Functions for COO format
void off_sparse_matvec_mult_COO(const sparseMatrixCOO*, Vector*);
// Functions for CSR format
void off_sparse_matvec_mult_CSR(const sparseMatrixCSR*, Vector*);
// Common functions.
double off_dot_product(const Vector*, const Vector*);
#endif 
