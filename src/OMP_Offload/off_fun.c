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




/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */

 // we shall have the same amount of threads as number of rows, sutch that eatch thread handles an entire row.
void off_dense_matvec_mult(const denseMatrix* A, Vector* x){
    double* temp = malloc(sizeof(double) * x->size);
    double sum;
    
    #pragma omp target map(to: A->data[0: A->rows * A->cols], x->data[0: x->size]) map(from: temp[0: x->size])
    #pragma omp parallel for default(none) private(sum) shared(temp, A, x)
    for(int i = 0; i < A->rows; i++){
        sum = 0;
        for (int j = 0; j < A->cols; j++){
            double value = A->data[i * A->cols + j];
            sum = fma(value, x->data[j], sum);
        }
        temp[i] = sum;
    }
    //TODO: parallize this part also
    for(int i = 0; i < x->size; i++){
        x->data[i] = temp[i];
    }
    free(temp);
}


double off_dot_product(const Vector* x, const Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }

    double dot = 0.0;
    #pragma omp target map(to: x->data[0: x->size], y->data[0: y->size]) map(from: dot)
    #pragma omp parallel for default(none) shared(x, y) reduction(+:dot) 
    for(int i = 0; i < x->size; i++){
        //dot = fma(x->data[i], y->data[i], dot);
        dot += x->data[i] * y->data[i];
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


 void off_sparse_matvec_mult_CSR(const sparseMatrixCSR* A, Vector* x){

    double* temp = calloc(x->size, sizeof(double));
    double aux;

    #pragma omp target map(to: A->row_ptr[0: A->rows + 1], A->val[0: A->nnz], A->col[0: A->nnz], x->data[0: x->size]) map(from: temp[0: x->size])
    #pragma omp parallel for default(none) private(aux) shared(temp, A, x)
    for(int i = 0; i < A->rows; i++){
        aux = 0.0;
        for(int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++){
            aux = fma(x->data[A->col[j]], A->val[j], aux);
        }
        temp[i] += aux;
    }
    for (int i = 0; i < x->size; i++){
        x->data[i] = temp[i];
    }
    free(temp);

 }


void off_sparse_matvec_mult(const SparseMatrixAny* A, Vector* x){
    
    if (A->type == CSR) {
        openMP_sparse_matvec_mult_CSR(A->mat.csr, x);
    } else {
        printf("Runtime error: OpenMP currently only works with CSR format\n");
        exit(EXIT_FAILURE);
    }
}
