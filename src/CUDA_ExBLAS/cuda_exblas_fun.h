#ifndef CUDA_EXBLAS_FUN_H
#define CUDA_EXBLAS_FUN_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


// Function implemented in a C file
double cuda_ExBLAS_dot_product(const Vector *x, const Vector *y, const int fpe, const bool early_exit);
// Launch wrappers implemented in .cu (compiled with nvcc)
void launch_ExDOT(
    long *d_PartialSuperaccs,
    double *d_a,
    double *d_b,
    const unsigned int NbElements
);

void launch_ExDOTComplete(
    double *d_Res,
    long *d_PartialSuperaccs,
    unsigned int PartialSuperaccusCount
);

                                
#ifdef __cplusplus
}
#endif

#endif 
