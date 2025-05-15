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


void cuda_sparse_matvec_mult_CSR(const sparseMatrixCSR *A, Vector *x, Vector *y){

    int *row_ptr, *col;
    double *val, *ivector, *ovector;
    cudaMalloc((void**)&row_ptr, sizeof(int) * (A->rows + 1));
    cudaMalloc((void**)&col, sizeof(int) * A->nnz);
    cudaMalloc((void**)&val, sizeof(double) * A->nnz);
    cudaMalloc((void**)&ivector, sizeof(double) * x->size);
    cudaMalloc((void**)&ovector, sizeof(double) * x->size);

    cudaMemcpy(row_ptr, A->row_ptr, sizeof(int) * (A->rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(col, A->col, sizeof(int) * A->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(val, A->val, sizeof(double) * A->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(ivector, x->data, sizeof(double) * x->size, cudaMemcpyHostToDevice);

    launch_matvec_CSR_kernel(A->rows, row_ptr, col, val, ivector, ovector);

    cudaMemcpy(y->data, ovector, sizeof(double) * x->size, cudaMemcpyDeviceToHost);
  
    cudaFree(row_ptr);
    cudaFree(col);
    cudaFree(val);
    cudaFree(ivector);
    cudaFree(ovector);

}

void cuda_sparse_matvec_mult(const SparseMatrixAny *A, Vector *x, Vector *y){
    if (A->type == CSR) {
        cuda_sparse_matvec_mult_CSR(A->mat.csr, x, y);
    } else {
        printf("Runtime error: OpenMP currently only works with CSR format\n");
        exit(EXIT_FAILURE);
    }
}


void cuda_dense_matvec_mult(const denseMatrix *A, Vector *x, Vector *y){

    double *val, *ivector, *ovector;

    cudaMalloc((void **)&val, sizeof(double) * A->cols * A->rows);
    cudaMalloc((void **)&ivector, sizeof(double) * x->size);
    cudaMalloc((void **)&ovector, sizeof(double) * x->size);

    cudaMemcpy(val, A->data, sizeof(double) * A->cols * A->rows, cudaMemcpyHostToDevice);
    cudaMemcpy(ivector, x->data, sizeof(double) * x->size, cudaMemcpyHostToDevice);

    launch_matvec_dense_kernel(A->rows, A->cols, val, ivector, ovector);

    cudaMemcpy(y->data, ovector, sizeof(double) * x->size, cudaMemcpyDeviceToHost);

    cudaFree(val);
    cudaFree(ivector);
    cudaFree(ovector);
}


void cuda_vector_norm_div(const Vector *x, Vector *y, double norm){

    double *ivector, *ovector;

    cudaMalloc((void **)&ivector, sizeof(double) * x->size);
    cudaMalloc((void **)&ovector, sizeof(double) * y->size);

    cudaMemcpy(ivector, x->data, sizeof(double) * x->size, cudaMemcpyHostToDevice);

    launch_vector_norm_div(ivector, ovector, norm, x->size);

    cudaMemcpy(y->data, ovector, sizeof(double) * x->size, cudaMemcpyDeviceToHost);

    cudaFree(ivector);
    cudaFree(ovector);
}
