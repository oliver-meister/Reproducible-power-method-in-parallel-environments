#ifndef CUDA_H
#define CUDA_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


void cuda_dense

// Function implemented in a C file
double cuda_dot_product(Vector* x, Vector* y);

// Launch wrappers implemented in .cu (compiled with nvcc)
void launch_reduce1_kernel(double* x, double* y, double* result, int n, int numBlocks, int blockSize);
void launch_reduce2_kernel(double* input, double* output, int currentSize, int nextSize, int blockSize);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H
