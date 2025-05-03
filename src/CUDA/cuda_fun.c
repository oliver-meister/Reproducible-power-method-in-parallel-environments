#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_fun.h"
#include <stdio.h>

#define WARP_COUNT 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_COUNT * WARP_SIZE)



double cuda_dot_product(Vector* x, Vector* y) {
    const int vector_size = x->size;
    if (vector_size != y->size) {
        fprintf(stderr, "Vector size mismatch.\n");
        return 0.0;
    }

    double *d_x, *d_y, *d_result, *d_temp;
    cudaMalloc((void**)&d_x, sizeof(double) * vector_size);
    cudaMalloc((void**)&d_y, sizeof(double) * vector_size);

    
    int numBlocks = (vector_size + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc((void**)&d_result, sizeof(double) * numBlocks);

    cudaMemcpy(d_x, x->data, sizeof(double) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y->data, sizeof(double) * vector_size, cudaMemcpyHostToDevice);

    // First kernel: compute x[i]*y[i]
    launch_reduce1_kernel(d_x, d_y, d_result, vector_size, numBlocks, BLOCK_SIZE);

    // Keep reducing until one value remains
    double result = 0.0;
    int currentSize = numBlocks;
    while (currentSize > 1) {
        int nextSize = (currentSize + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        cudaMalloc((void**)&d_temp, sizeof(double) * nextSize);

        launch_reduce2_kernel(d_result, d_temp, currentSize, nextSize, BLOCK_SIZE);

        cudaFree(d_result);
        d_result = d_temp;
        currentSize = nextSize;
    }

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    return result;
}



/*
__global__ void DOT(const double* d_x, const double* d_y, const int incx, const int incy, const int offsetx, const int offsety, const int NbElements, double* result){
    
extern __shared__ double sdata[];

int tid = threadIdx.x;
int i = blockIdx.x * blockDim.x + local_id;
int total_threads = blockDim.x * gridDim.x;

double local_sum = 0.0;

for(int pos = i; pos < NbElements; pos += total_threads){
    local_sum += d_x[pos] * d_y[pos];
}

sdata[tid] = local_sum;
__syncthreads();


for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
    sdata[tid] += sdata[tid + s];
    __syncthreads();
}

if (tid == 0){
    result[blockIdx.x] = sdata[0];
}
}

__global__ void DOTComplete(double *d_res ){
    
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
    sd[tid] += sdata[tid + s];
    __syncthreads();
}
}


double cuda_dot(const Vector* x, const Vector* y, const int incx, const int incy, const int offsetx, const int offsety){
    int size_x = x->size;
    int size_y = y->size;
    const double* d_x, d_y;
    
    if(size_x != size_y){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", size_x, size_y);
        return 0.0;
    }
    //Allocate CUDA memory
    cudaMalloc(&d_x, sizeof(double) * size_x);
    cudaMalloc(&d_y, sizeof(double) * size_y);
    cudaMemcpy(d_x, x->data, sizeof(double) * size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y->data, sizeof(double) * size_y, cudaMemcpyHostToDevice);
    
    DOT<<<>>>
}
*/
