#ifndef SPARSE_POWER_METHOD_H
#define SPARSE_POWER_METHOD_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include "common.h"
#include <stdbool.h>
#include <stddef.h>

Res sparse_power_method(const SparseMatrixAny* A, double threshold);
void test_sparse_power_method(SparseMatrixAny *A, char* file_name, double threshold);
double sparse_approximate_eigenvalue(Vector*, Vector*);
double sparse_approximate_eigenvalue_CUDA(Vector* x, Vector *y, double* d_result, int numBlocks);
double sparse_approximate_eigenvalue_EXBLAS(Vector* x, Vector *y, long long int* d_PartialSuperaccs, double* d_result ,size_t size);
#endif 
