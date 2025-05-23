#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_exblas_fun.h"
#include <stdio.h>

#define PARTIAL_SUPERACCS_COUNT 512
#define BIN_COUNT      39

double runExDOT(const double *h_x, const double *h_y, int N){
    /*
    static int call_count = 0;
    call_count++;
    printf("[DEBUG] runExDOT called %d times\n", call_count);
    */

    double *d_x;
    double *d_y;
    double *d_result;
    // Allocate CUDA memory
    
    // Initializing CUDA ExDOT
    
    
    long long int* d_PartialSuperaccs;
    size_t size = PARTIAL_SUPERACCS_COUNT * BIN_COUNT * sizeof(long long int);  
    
    
    cudaMalloc((void**)&d_PartialSuperaccs, size);
    cudaMalloc((void**)&d_x, sizeof(double) * N);
    cudaMalloc((void**)&d_y, sizeof(double) * N);
    cudaMalloc((void**)&d_result, sizeof(double));
    
   
    cudaMemcpy(d_x, h_x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(double) * N, cudaMemcpyHostToDevice);
    
    // Running CUDA ExDOT
    cudaMemset(d_PartialSuperaccs, 0, size);
    launch_ExDOT(d_PartialSuperaccs, d_x, d_y, N);
    launch_ExDOTComplete(d_PartialSuperaccs);
    cudaMemset(d_result, 0, sizeof(double));
    launch_FinalReduceAndRound(d_result, d_PartialSuperaccs);
    double h_result;  
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("ExDOT dot result, not in kernel: %.20e\n", h_result);

    cudaFree(d_PartialSuperaccs);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return h_result;
}

double cuda_ExBLAS_dot_product(const Vector *x, const Vector *y){

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

    double dot = runExDOT(x->data, y->data, x->size);
    return dot;
    
}