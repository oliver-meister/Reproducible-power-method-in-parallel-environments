#include "serial_fun.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../common.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>



/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_dense_matvec_mult(const denseMatrix* A, Vector* x, Vector* y){

    
    double sum;
    for(int i = 0; i < A->rows; i++){
        sum = 0.0;
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum = fma(value, x->data[j], sum);
        }
        y->data[i] = sum;
    }
 
}


/**
 * @brief Calculates the dot product of two Vectors.
 * 
 * @param x First vector.
 * @param y Second vector.
 * 
 * @return The dot product of vector x and y.
 */
double serial_dot_product(const Vector* x, const Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }
    double dot = 0;
    for(int i = 0; i < x->size; i++){
        dot += x->data[i] * y->data[i];
    }
    return dot;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Sparse power method




/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_sparse_matvec_mult_CSR(const sparseMatrixCSR* A, Vector* x, Vector* y){
    
    double aux;
    for(int i = 0; i < A->rows; i++){
        aux = 0.0;
        for(int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++){
            aux = fma(x->data[A->col[j]], A->val[j], aux);
        }
        y->data[i] = aux;
    }
}

/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_sparse_matvec_mult_COO(const sparseMatrixCOO* A, Vector* x, Vector* y){
    
    for (int i = 0; i < y->size; i++) {
        y->data[i] = 0.0;
    }
    // iterate thrue all non zero elemets
    for (int i = 0; i < A->nnz ; i++){
        double value = A->val[i];
        int value_row = A->row[i];
        int value_column = A->col[i];
        // In contrast to the dense case, we must write results directly to temp,
        // as we don't iterate in row-major order.
        y->data[value_row] = fma(value, x->data[value_column], y->data[value_row]);
    }
}

void serial_sparse_matvec_mult(const SparseMatrixAny* A, Vector* x, Vector* y) {
    if (A->type == CSR) {
        serial_sparse_matvec_mult_CSR(A->mat.csr, x, y);
    } else {
        serial_sparse_matvec_mult_COO(A->mat.coo, x, y);
    }
}

void serial_vector_norm_div(const Vector *x, Vector *y, double norm){
      for(int i = 0; i < x->size; i++){
        y->data[i] =  x->data[i] / norm;
    }
}