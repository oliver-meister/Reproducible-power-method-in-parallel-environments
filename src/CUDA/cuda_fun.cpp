#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_fun.h"
#include <stdio.h>

#define WARP_COUNT 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_COUNT * WARP_SIZE)
#define NUM_BLOCKS

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void reduce(double *x_idata, double *y_idata, int *odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { 
        sdata[tid] += x_idata[i] * y_idata[i] + x_idata[i+blockSize] * y_idata[i+blockSize];
        i += gridSize; 
    }
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
    if (tid == 0)
    result[blockIdx.x] = sdata[0]
}

__global__ void DOTComplete(){


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
