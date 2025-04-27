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
template <unsigned int blockSize, typename Op>
__global__ void reduce(double *g_idata, double *g_odata, unsigned int n, Op, op) {
extern __shared__ double sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize*2) + tid;
// Jump distance for strided loop
unsigned int gridSize = blockSize*2*gridDim.x;
sdata[tid] = 0;
while (i < n) { sdata[tid] += op(g_idata[i], g_idata[i + blockSize]); i += gridSize; }
__syncthreads();
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) warpReduce(sdata, tid); 
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

double cuda_dot_product(const Vector* x, const Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }
    
    int vectorSize = x->size;
    int numElements = vectorSize * 2;
    double* d_input;
    double* d_output;

    int blocksize = 256;
    int numBlocks  = std::max(1, static_cast<int>(numElements / (blocksize * std::log(numElements))));

    cudaMalloc(&d_input, sizeof(double) * numElements);
    cudaMalloc(&d_output, sizeof(double) * blocksize);

    cudaMemcpy(d_input, x->data, sizeof(double) * vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input + vectorSize , y->data, sizeof(double) * vectorSize, cudaMemcpyHostToDevice);

    reduce<blocksize><<<numBlocks, blocksize, blocksize * sizeof(double)>>>(d_input, d_output, numElements, Multiply());

        while(true){
            if(numBlocks == 1){
                double result;
                cudaMemcpy(&result, d_output, sizeof(double), cudaMemcpyDeviceToHost);
                cudafree(d_input);
                cudafree(d_output);
                return result;
            }
            numElements = numBlocks;
            numBlocks = std::max(1, static_cast<int>(numElements / (blocksize * std::log(numElements))));
            double* temp = d_input;
            d_input = d_output;
            d_output = temp;
            reduce<blocksize><<<numBlocks, blocksize, blocksize * sizeof(double)>>>(d_input, d_output, numElements, Add());
        }

}

