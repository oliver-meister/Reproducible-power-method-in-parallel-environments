#include "power_method_serial.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>




// Dense power method

/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double serial_dense_power_method(denseMatrix* A){
   
    // initial vector
    double lambda_old = 0;
    double lambda_new = 0;
    Vector* x = generate_random_vector(A->rows);

    do{
        lambda_old = lambda_new;
        serial_dense_matvec_mult(A, x);
        serial_normalize_vector(x);
        lambda_new = serial_dense_approximate_eigenvalue(A, x);
        //printf(" dense lambda approximation: %f\n", lambda_new);
    } while(!serial_convergence(lambda_new, lambda_old, 0.00001));

    free(x->data);
    free(x);
    return lambda_new;
}

/**
 * @brief Compares two eigenvalues and determines whether they have converged
 *          (i.e., if the difference between them is less than a given threshold).
 * 
 * @param lambda_new he most recent eigenvalue estimate.
 * @param lambda_old The previous eigenvalue estimate.
 * @param threshold The threshold below which convergence is assumed.
 * 
 * @return True if the difference between the two eigenvalues is less than the threshold, false otherwise.
 */
bool serial_convergence(double lambda_new, double lambda_old, double threshold){
    return (fabs(lambda_new - lambda_old) < threshold);
}

/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_dense_matvec_mult(denseMatrix* A, Vector* x){

    double* temp = malloc(sizeof(double) * x->size);
    double sum;

    for(int i = 0; i < A->rows; i++){
        sum = 0.0;
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum = fma(value, x->data[j], sum);
        }
        temp[i] = sum;
    }
    for(int i = 0; i < x->size; i++){
        x->data[i] = temp[i];
    }
    free(temp);
}


/**
 * @brief Normalize the vector into a unit vector.
 * 
 * @param x The input/output vector. It is overwritten by the unit vector.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_normalize_vector(Vector* x){

    double norm = sqrt(dot_product(x, x));
    if (norm == 0) return;
    for(int i = 0; i < x->size; i++){
        x->data[i] =  x->data[i] / norm;
    }
}


/**
 * @brief  Approximates the dominant eigenvalue.
 * 
 * @param A The input matrix.
 * @param x The normalized input vector.
 * 
 * @return The approximated dominant eigenvalue.
 */
double serial_dense_approximate_eigenvalue(denseMatrix* A, Vector* x){
    Vector copy;
    copy.size = x->size;
    copy.data = malloc(sizeof(double) * copy.size);
    for(int i = 0; i < copy.size; i++){
        copy.data[i] = x->data[i];
    }
    // Ax_{i+1}
    serial_dense_matvec_mult(A, &copy);

    double lambda = dot_product(x, &copy);

    free(copy.data);
    return lambda;
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Sparse power method



/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double serial_sparse_power_method(sparseMatrix* A){
   
    // initial vector
    double lambda_old = 0;
    double lambda_new = 0;
    //Vector* x = generate_random_vector(A->rows);
    Vector* x = malloc(sizeof(Vector));
    x->size = A->rows;
    x->data = malloc(sizeof(double) * x->size);

    for(int i = 0; i < x->size; i++){
        x->data[i] = 1.0;
    }
    
    do{
        lambda_old = lambda_new;
        serial_sparse_matvec_mult(A, x);
        serial_normalize_vector(x);
        lambda_new = serial_sparse_approximate_eigenvalue(A, x);
        
        //printf("sparse lambda approximation: %f\n", lambda_new);
    } while(!serial_convergence(lambda_new, lambda_old, 0.000001));

    free(x->data);
    free(x);
    return lambda_new;
}


/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void serial_sparse_matvec_mult(sparseMatrix* A, Vector* x){

    double* temp = calloc(x->size, sizeof(double));

    // iterate thrue all non zero elemets
    for (int i = 0; i < A->nnz ; i++){
        double value = A->val[i];
        int value_row = A->row[i];
        int value_column = A->col[i];
        // In contrast to the dense case, we must write results directly to temp,
        // as we don't iterate in row-major order.
        temp[value_row] = fma(value, x->data[value_column], temp[value_row]);
    }
    for(int i = 0; i < x->size; i++){
        x->data[i] = temp[i];
    }
    free(temp);
}


double serial_sparse_approximate_eigenvalue(sparseMatrix* A, Vector* x){
    Vector copy;
    copy.size = x->size;
    copy.data = malloc(sizeof(double) * copy.size);
    for(int i = 0; i < copy.size; i++){
        copy.data[i] = x->data[i];
    }
    serial_sparse_matvec_mult(A, &copy);
    double lambda = dot_product(x, &copy);

    free(copy.data);
    return lambda;
}