#ifndef SPARSE_POWER_METHOD_H
#define SPARSE_POWER_METHOD_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include "common.h"
#include <stdbool.h>

Res sparse_power_method(const SparseMatrixAny* );
double sparse_approximate_eigenvalue(Vector*, Vector*);
double sparse_approximate_eigenvalue_CUDA(Vector*, Vector*, double* d_result, int numBlocks);
void test_sparse_power_method(SparseMatrixAny *A, char* file_name);
#endif 
