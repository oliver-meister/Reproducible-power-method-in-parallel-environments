#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


#define BLOCK_SIZE 32


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
__global__ void reduce(double *g_idata, double *g_odata, unsigned int n) {
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
__global__ void dotprod(double *x_idata, double *y_idata, double* result, unsigned int n) {
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

    extern "C" void launch_dotprod_kernel(double* x, double* y, double* result, int n, int numBlocks) {
        dotprod<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(x, y, result, n);
    }
    extern "C" void launch_reduce_kernel(double* input, double* output, int currentSize, int nextSize, int blockSize) {
        reduce<BLOCK_SIZE><<<nextSize, blockSize>>>(input, output, currentSize);
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

__global__ void matvec_dense_kernel(const int num_rows, const int num_cols, 
                                    const double *val, const double *input_vector, double *output_vector)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double dot = 0.0;
        for (int col = 0; col < num_cols; col++) {
            dot += val[row * num_cols + col] * input_vector[col];
        }
        output_vector[row] = dot;
    }
}

extern "C" void launch_matvec_dense_kernel(const int num_rows, const int num_cols, 
                                            const double *val, const double *input_vector, double *output_vector)
{

    int gridSize = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_dense_kernel<<<gridSize, BLOCK_SIZE>>>(num_rows, num_cols,
                                                    val, input_vector, output_vector);

}

__global__ void vector_norm_div_kernel(const double* input_vector, double* output_vector, double norm, int numElem){

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    for(unsigned int pos = gid; pos < numElem; pos += gridDim.x * blockDim.x){
        output_vector[pos] = input_vector[pos] / norm;
    }
}

extern "C" void launch_vector_norm_div(const double* input_vector, double* output_vector, double norm, int numElem){
    int gridSize = (numElem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_norm_div_kernel<<<gridSize, BLOCK_SIZE>>>(input_vector, output_vector, norm, numElem);
}