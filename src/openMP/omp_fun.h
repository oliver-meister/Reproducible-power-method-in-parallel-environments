#ifndef OPENMP_H
#define OPENMP_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense power method
void openMP_dense_matvec_mult(const denseMatrix*, Vector*, Vector*);
//Function for sparse power method
void openMP_sparse_matvec_mult(const SparseMatrixAny*, Vector*, Vector*);
// Functions for CSR format
void openMP_sparse_matvec_mult_CSR(const sparseMatrixCSR*, Vector*, Vector*);
// Common functions
double openMP_dot_product(const Vector*, const Vector*);
double openMP_dot_product2(const Vector*, const Vector*);

void openMP_vector_norm_div(const Vector*, Vector*, double);
#endif 
