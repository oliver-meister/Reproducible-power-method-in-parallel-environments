#ifndef COMMON_H
#define COMMON_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include <stdbool.h>

typedef void (*dense_matvec_fn)(const denseMatrix*, Vector*, Vector*);
typedef void (*sparse_matvec_fn)(const SparseMatrixAny*, Vector*, Vector*);
typedef double (*dot_fn)(const Vector*, const Vector*);
typedef void (*vector_norm_div_fun)(const Vector*, Vector*, double);

typedef double (*start_timer)();
typedef double (*stop_timer)(double start);

bool convergence(double prev, double curr, double tolerance);
void normalize_vector(Vector* x, Vector* y);
void init_backend();

typedef struct {
    double lambda;
    double time;
} Res;


#endif