#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_exblas_fun.h"
#include <stdio.h>

double runExDOT(const double *d_x, const double *d_y, long long int* d_PartialSuperaccs, double* d_result ,int N, size_t size){
    /*
    static int call_count = 0;
    call_count++;
    printf("[DEBUG] runExDOT called %d times\n", call_count);
    */
    // Running CUDA ExDOT
    cudaMemset(d_PartialSuperaccs, 0, size);
    launch_ExDOT(d_PartialSuperaccs, d_x, d_y, N);
    launch_ExDOTComplete(d_PartialSuperaccs);
    cudaMemset(d_result, 0, sizeof(double));
    launch_FinalReduceAndRound(d_result, d_PartialSuperaccs);
    double h_result;  
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return h_result;
}

double cuda_ExBLAS_dot_product(const Vector *x, const Vector *y, long long int* d_PartialSuperaccs, double* d_result ,size_t size){

    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }

    /*
    if(early_exit){
        if(fpe <= 4)
        ckKernel = 
        ckComplete =
        if(fpe <= 6)
        ckKernel = 
        ckComplete =
        if(fpe <= 8)
        ckKernel = 
        ckComplete =
    } else {
        ckKernel = 
        ckComplete =
    }
    */

    double dot = runExDOT(x->d_data, y->d_data, d_PartialSuperaccs, d_result ,x->size, size);
    return dot;
    
}