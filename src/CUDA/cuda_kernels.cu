#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256


template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce2(double *g_idata, double *g_odata, unsigned int n) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1(double *x_idata, double *y_idata, double* result, unsigned int n) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += x_idata[i] * y_idata[i] + x_idata[i+blockSize] * y_idata[i+blockSize] ; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0) result[blockIdx.x] = sdata[0];
    }

    extern "C" void launch_reduce1_kernel(double* x, double* y, double* result, int n, int numBlocks, int blockSize) {
        reduce1<BLOCK_SIZE><<<numBlocks, blockSize, blockSize * sizeof(double)>>>(x, y, result, n);
    }
    extern "C" void launch_reduce2_kernel(double* input, double* output, int currentSize, int nextSize, int blockSize) {
        reduce2<BLOCK_SIZE><<<nextSize, blockSize, blockSize * sizeof(double)>>>(input, output, currentSize);
    }


// one thread per row
__global__ void matvec_CSR_kernel(const int num_rows, 
                                    const int *row_ptr, 
                                    const int *col, 
                                    const double *val, 
                                    const double *input_vector, 
                                    double* output_vector)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < num_rows){
        double dot = 0.0;
        for(int i = row_ptr[row]; i < row_ptr[row + 1]; i++){
            dot += val[i] * input_vector[col[i]];
        }
        output_vector[row] = dot;
    }
}

extern "C" void launch_matvec_CSR_kernel(const int num_rows, 
                                            const int *row_ptr, 
                                            const int *col, 
                                            const double *val, 
                                            const double *input_vector, 
                                            double* output_vector)
{
    int gridSize = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_CSR_kernel<<<gridSize, BLOCK_SIZE>>>(num_rows, 
                                                row_ptr, 
                                                col, 
                                                val,
                                                input_vector, 
                                                output_vector);


}