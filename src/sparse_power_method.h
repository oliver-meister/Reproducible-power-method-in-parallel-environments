#ifndef SPARSE_POWER_METHOD_H
#define SPARSE_POWER_METHOD_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include <stdbool.h>

double sparse_power_method(const SparseMatrixAny* );
double sparse_approximate_eigenvalue(const SparseMatrixAny*, const Vector*, bool);

#endif 
