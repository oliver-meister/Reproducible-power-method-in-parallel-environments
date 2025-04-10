#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense matrices.
double serial_dense_power_method(denseMatrix*);
void serial_dense_matvec_mult(denseMatrix*, Vector*);
double serial_dense_approximate_eigenvalue(denseMatrix*, Vector*);
// Function for sparse matrices.
double serial_sparse_power_method(SparseMatrixAny*);
double serial_sparse_approximate_eigenvalue(SparseMatrixAny*, Vector*);
void serial_sparse_matvec_mult(SparseMatrixAny*, Vector*);
// Functions for COO format
void serial_sparse_matvec_mult_COO(sparseMatrixCOO*, Vector*);
// Functions for CSR format
void serial_sparse_matvec_mult_CSR(sparseMatrixCSR*, Vector*);
// Common functions.
void serial_normalize_vector(Vector*);
bool serial_convergence(double, double, double);
#endif 
