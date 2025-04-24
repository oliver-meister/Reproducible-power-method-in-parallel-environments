#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_fun.h"
#include <stdio.h>
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

/*
int *g_idata - input array in global memory.

int *g_outdata - output array in global memory.

unsigned int n - total number of elements in the input array.

*/
template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
// shared memory
extern __shared__ int sdata[];
// Thread ID within the block
unsigned int tid = threadIdx.x;
// Global index of current thread
unsigned int i = blockIdx.x*(blockSize*2) + tid;
// Jump distance for strided loop
unsigned int gridSize = blockSize*2*gridDim.x;
sdata[tid] = 0;
while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
__syncthreads();
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) warpReduce(sdata, tid); 
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Use c++ template to chose operation in while, such that we can control the operation and not having it as an function pointer which takes som time to read for each thread.

double cuda_dot_product(const Vector* x, const Vector* y){

    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }
    int n = x->size;
    double* input;
    int blocksize = 256;
    int numbBlocks  = 
    cudaMalloc(&input, sizeof(double) * n * 2);
    cudaMemcpy(input, x->data, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(input + n , y->data, sizeof(double) * n, cudaMemcpyHostToDevice);
 
}