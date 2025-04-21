#ifndef SERIAL_H
#define SERIAL_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense matrices.
void serial_dense_matvec_mult(const denseMatrix*, Vector*);
// Function for sparse matrices.
void serial_sparse_matvec_mult(const SparseMatrixAny*, Vector*);
// Functions for COO format
void serial_sparse_matvec_mult_COO(const sparseMatrixCOO*, Vector*);
// Functions for CSR format
void serial_sparse_matvec_mult_CSR(const sparseMatrixCSR*, Vector*);
// Common functions.
double serial_dot_product(const Vector*, const Vector*);
#endif 
