#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_fun.h"
#include <stdio.h>

double cuda_dot_product(const Vector* x, const Vector* y, double *d_result, int numBlocks) {
    const int vector_size = x->size;
    if (vector_size != y->size) {
        fprintf(stderr, "Vector size mismatch.\n");
        return 0.0;
    }

    // what shall the block size be?
    //int numBlocks = (vector_size + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    // First reduction kernel: compute x[i] * y[i]
    launch_dotprod_kernel(x->d_data, y->d_data, d_result, vector_size, numBlocks);

    double result = 0.0;
    
    if (numBlocks > 1){
        launch_dot_complete_kernel(d_result, d_result, numBlocks);
    }
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}


void cuda_sparse_matvec_mult_CSR(const sparseMatrixCSR *A, Vector *x, Vector *y){
    launch_matvec_CSR_kernel(A->rows, A->d_row_ptr, A->d_col, A->d_val, x->d_data, y->d_data);
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
    launch_vector_norm_div(x->d_data, y->d_data, norm, x->size);
}

void copy_vector_from_device_to_host(Vector* v)
{
    cudaMemcpy(v->data, v->d_data, sizeof(double) * v->size, cudaMemcpyDeviceToHost);
}
