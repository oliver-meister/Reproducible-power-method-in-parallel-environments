#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>early_exit
#include <stdlib.h>
#include "cuda_exblas_fun.h"
#include <stdio.h>

#define PARTIAL_SUPERACCS_COUNT 512
#define BIN_COUNT      39

double runExDOT(const double *h_x, const double *h_y, int N){
    
    const double *d_x;
    const double *d_y;
    long *d_result;
    // Allocate CUDA memory
    
    // Initializing CUDA ExDOT
    
    unsigned long long int* d_PartialSuperaccs;
    size_t size = PARTIAL_SUPERACCS_COUNT * BIN_COUNT * sizeof(unsigned long long int);
    
    
    cudaMalloc((void**)&d_PartialSuperaccs, size);
    cudaMalloc((void**)&d_x, sizeof(double) * N);
    cudaMalloc((void**)&d_y, sizeof(double) * N);
    cudaMalloc((void**)&d_result, sizeof(long));
    
    cudaMemcpy(d_x, h_x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(double) * N, cudaMemcpyHostToDevice);
    
    // Running CUDA ExDOT
    
    launch_ExDOT(d_PartialSuperaccs, d_x, d_y, N);
    launch_ExDOTComplete(d_result, d_PartialSuperaccs, PARTIAL_SUPERACCS_COUNT);
    
    long h_result;  
    cudaMemcpy(&h_result, d_result, sizeof(long), cudaMemcpyDeviceToHost);

    cudaFree(d_PartialSuperaccs);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    
    
    return (double) h_result;
}

double cuda_ExBLAS_dot_product(const Vector *x, const Vector *y, const int fpe, const bool early_exit){

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

    double dot = runExDOT(x->data, y->data, x->size);
    return dot;
    
}