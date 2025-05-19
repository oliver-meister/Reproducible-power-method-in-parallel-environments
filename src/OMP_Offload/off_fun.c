#include "off_fun.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../common.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


#define BLOCK_SIZE 512


/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */

 // we shall have the same amount of threads as number of rows, sutch that eatch thread handles an entire row.
void off_dense_matvec_mult(const denseMatrix* A, Vector* x, Vector* y) {
    int rows = A->rows;
    int cols = A->cols;
    double* A_data = A->data;
    double* x_data = x->data;
    double* y_data = y->data;

    #pragma omp target teams distribute parallel for \
        num_teams((rows + BLOCK_SIZE -1 )/ BLOCK_SIZE) thread_limit(BLOCK_SIZE) \
        map(to: A_data[0:rows * cols], x_data[0:cols]) \
        map(from: y_data[0:rows]) \
        default(none) firstprivate(rows, cols) shared(A_data, x_data, y_data)
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A_data[i * cols + j] * x_data[j];
        }
        y_data[i] = sum;
    }
}



double off_dot_product(const Vector* x, const Vector* y) {
    if (x->size != y->size) {
        printf("Error: Vectors must have the same size \n");
        return 0.0;
    }

    int size = x->size;
    double* x_data = x->data;
    double* y_data = y->data;
    double dot = 0.0;

    #pragma omp target teams distribute parallel for \
        num_teams((size + BLOCK_SIZE - 1 )/ BLOCK_SIZE) thread_limit(BLOCK_SIZE) \
        map(to: x_data[0:size], y_data[0:size]) \
        map(tofrom: dot) default(none) firstprivate(size) shared(x_data, y_data) reduction(+:dot)
    for (int i = 0; i < size; i++) {
        dot += x_data[i] * y_data[i];
    }

    return dot;
}




///////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */


void off_sparse_matvec_mult_CSR(const sparseMatrixCSR* A, Vector* x, Vector *y) {
    int rows = A->rows;
    int nnz = A->nnz;
    int* row_ptr = A->row_ptr;
    int* col = A->col;
    double* val = A->val;

    int x_size = x->size;
    double* x_data = x->data;
    double* y_data = y->data;

    #pragma omp target teams distribute parallel for num_teams((rows + BLOCK_SIZE -1 )/ BLOCK_SIZE) thread_limit(BLOCK_SIZE) \
        map(to: row_ptr[0:rows + 1], val[0:nnz], col[0:nnz], x_data[0:x_size]) \
        map(from: y_data[0:rows]) \
        default(none) firstprivate(rows) shared(row_ptr, col, val, x_data, y_data)
    for (int i = 0; i < rows; i++) {
        double aux = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            aux += x_data[col[j]] * val[j];
        }
        y_data[i] = aux;
    }
}



void off_sparse_matvec_mult(const SparseMatrixAny* A, Vector* x, Vector *y){
    
    if (A->type == CSR) {
        off_sparse_matvec_mult_CSR(A->mat.csr, x, y);
    } else {
        printf("Runtime error: OpenMP currently only works with CSR format\n");
        exit(EXIT_FAILURE);
    }
}

void off_vector_norm_div(const Vector *x, Vector *y, double norm) {
    int size = x->size;
    double* x_data = x->data;
    double* y_data = y->data;

    #pragma omp target teams distribute parallel for num_teams((size + BLOCK_SIZE -1 )/ BLOCK_SIZE) thread_limit(BLOCK_SIZE) \
        map(to: x_data[0:size]) \
        map(from: y_data[0:size]) \
        default(none) firstprivate(size, norm) shared(x_data, y_data)
    for (int i = 0; i < size; i++) {
        y_data[i] = x_data[i] / norm;
    }
}
