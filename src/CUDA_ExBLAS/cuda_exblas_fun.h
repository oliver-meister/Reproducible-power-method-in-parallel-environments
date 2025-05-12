#ifndef CUDA_EXBLAS_FUN_H
#define CUDA_EXBLAS_FUN_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


// Function implemented in a C file
double cuda_ExBLAS_dot_product(const Vector *x, const Vector *y);
// Launch wrappers implemented in .cu (compiled with nvcc)
void launch_ExDOT(
    long long int *d_PartialSuperaccs,
    double *d_a,
    double *d_b,
    const unsigned int NbElements
);

void launch_ExDOTComplete(
    long long int *d_PartialSuperaccs
);

void launch_FinalReduceAndRound(double *d_Res, long long int *d_PartialSuperaccs);

                                
#ifdef __cplusplus
}
#endif

#endif 
