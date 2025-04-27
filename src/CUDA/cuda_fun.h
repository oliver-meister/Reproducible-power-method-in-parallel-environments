#ifndef CUDA_H
#define CUDA_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense matrices.
void cuda_dense_matvec_mult(const denseMatrix*, Vector*);
// Function for sparse matrices.
void cuda_sparse_matvec_mult(const SparseMatrixAny*, Vector*);
// Functions for COO format
void cuda_sparse_matvec_mult_COO(const sparseMatrixCOO*, Vector*);
// Functions for CSR format
void cuda_sparse_matvec_mult_CSR(const sparseMatrixCSR*, Vector*);
// Common functions.
double cuda_dot_product(const Vector*, const Vector*);

struct Multiply {
    __device__ __forceinline__
    int operator()(int a, int b) const {
        return a * b;
    }
};

struct Add {
    __device__ __forceinline__
    int operator()(int a, int b) const {
        return a + b;
    }
};

#endif 
