#ifndef VECTOR_H
#define VECTOR_H
#include "matrix.h"

typedef struct {
    double* data;
    int size;
} Vector;

double generate_random_double(double, double);
Vector* generate_random_vector(int size);
Vector* generate_vector(int size);
void delete_vector(Vector* x);
Vector* generate_sum_vector_sparse(SparseMatrixAny *A);
Vector* generate_sum_vector_dense(denseMatrix *A);
Vector* generate_sum_vector_COO(sparseMatrixCOO *A);
Vector* generate_sum_vector_CSR(sparseMatrixCSR *A);
#endif 

