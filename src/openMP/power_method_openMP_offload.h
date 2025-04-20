#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense power method
double openMP_offload_dense_power_method(denseMatrix*);
void openMP_offload_dense_matvec_mult(denseMatrix*, Vector*);
double openMP_offload_dense_approximate_eigenvalue(denseMatrix*, Vector*);
//Function for sparse power method
// Function for sparse matrices.
double openMP_offload_sparse_power_method(SparseMatrixAny*);
double openMP_offload_sparse_approximate_eigenvalue(SparseMatrixAny*, Vector*);
void openMP_offload_sparse_matvec_mult(SparseMatrixAny*, Vector*);
// Functions for CSR format
void openMP_offload_sparse_matvec_mult_CSR(sparseMatrixCSR*, Vector*);
// Common functions
bool openMP_offload_convergence(double, double, double);
void openMP_offload_normalize_vector(Vector*);
double openMP_offload_dot_product(Vector*, Vector*);
double openMP_offload_dot_product2(Vector*, Vector*);


#endif 
