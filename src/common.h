#ifndef COMMON_H
#define COMMON_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include <stdbool.h>

typedef void (*dense_matvec_fn)(const denseMatrix*, Vector*);
typedef void (*sparse_matvec_fn)(const SparseMatrixAny*, Vector*);
typedef double (*dot_fn)(const Vector*, const Vector*);
typedef double (*dense_eigen_fn)(const denseMatrix*, const Vector*);
typedef double (*sparse_eigen_fn)(const SparseMatrixAny*, const Vector*);

bool convergence(double prev, double curr, double tolerance);
void normalize_vector(Vector* x);

#endif