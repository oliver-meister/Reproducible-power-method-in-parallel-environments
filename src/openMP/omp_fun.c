#include "omp_fun.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../common.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>




/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */

 // we shall have the same amount of threads as number of rows, sutch that eatch thread handles an entire row.
void openMP_dense_matvec_mult(const denseMatrix* A, Vector* x, Vector* y){

    double sum;
    #pragma omp parallel for default(none) private(sum) shared(y, A, x)
    for(int i = 0; i < A->rows; i++){
        sum = 0;
        
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum = fma(value, x->data[j], sum);
        }
        //int thread_id = omp_get_thread_num();
        //printf("Thread %d is processing row %d\n", thread_id, i);
        y->data[i] = sum;
    }
}


double openMP_dot_product(const Vector* x, const Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }

    double dot = 0.0;
    #pragma omp parallel for default(none) shared(x, y) reduction(+:dot) 
    for(int i = 0; i < x->size; i++){
        //dot = fma(x->data[i], y->data[i], dot);
        dot += x->data[i] * y->data[i];
    }
    return dot;
}


double openMP_dot_product2(const Vector* x, const Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }

    int n_threads = omp_get_max_threads();
    double *partial_sums = calloc(n_threads, sizeof(double));

    #pragma omp parallel default(none) shared(x, y, partial_sums)
    {
        int tid = omp_get_thread_num();
        double sum = 0.0; 
        //printf("Thread number %d\n", tid);
        #pragma omp for
        for(int i = 0; i < x->size; i++){

            sum += x->data[i] * y->data[i];
        }
        
        partial_sums[tid] = sum;
    }

    double dot = 0.0;

    for (int i = 0; i < n_threads; i++){
        dot += partial_sums[i];
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


 void openMP_sparse_matvec_mult_CSR(const sparseMatrixCSR* A, Vector* x, Vector *y){

    double aux;

    #pragma omp parallel for default(none) private(aux) shared(y, A, x)
    for(int i = 0; i < A->rows; i++){
        aux = 0.0;
        for(int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++){
            aux = fma(x->data[A->col[j]], A->val[j], aux);
        }
        y->data[i] = aux;
    }


 }


void openMP_sparse_matvec_mult(const SparseMatrixAny* A, Vector* x, Vector *y){
    
    if (A->type == CSR) {
        openMP_sparse_matvec_mult_CSR(A->mat.csr, x, y);
    } else {
        printf("Runtime error: OpenMP currently only works with CSR format\n");
        exit(EXIT_FAILURE);
    }
}


void openMP_vector_norm_div(const Vector *x, Vector *y, double norm){

    #pragma omp parallel for default(none) shared(x,y,norm) 
    for(int i = 0; i < x->size; i++){
        y->data[i] =  x->data[i] / norm;
    }
}