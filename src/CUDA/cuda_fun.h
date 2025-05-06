#ifndef CUDA_H
#define CUDA_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif




// Function implemented in a C file
double cuda_dot_product(Vector* x, Vector* y);
void cuda_sparse_matvec_mult(const SparseMatrixAny *A, Vector *x);
void cuda_sparse_matvec_mult_CSR(const sparseMatrixCSR *A, Vector *x);
void cuda_dense_matvec_mult(const denseMatrix *A, Vector *x);
// Launch wrappers implemented in .cu (compiled with nvcc)
void launch_reduce1_kernel(double* x, double* y, double* result, int n, int numBlocks, int blockSize);
void launch_reduce2_kernel(double* input, double* output, int currentSize, int nextSize, int blockSize);
void launch_matvec_CSR_kernel(const int num_rows, 
                                const int *row_ptr, 
                                const int *col, 
                                const double *val, 
                                const double *input_vector, 
                                double *output_vector);
                                
void launch_matvec_dense_kernel(const int num_rows, const int num_cols, 
                                    const double *val, const double *input_vector, double *output_vector)
#ifdef __cplusplus
}
#endif

#endif // CUDA_H
