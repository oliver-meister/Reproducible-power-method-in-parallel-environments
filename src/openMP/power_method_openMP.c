#include "power_method_openMP.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>



/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double openMP_power_method(Matrix* A){
   
    // initial vector
    double lambda_old = 0;
    double lambda_new = 0;
    Vector* x = generate_random_vector(A->rows);

    do{
        lambda_old = lambda_new;
        openMP_matvec_mult(A, x);
        openMP_normalize_vector(x);
        lambda_new = openMP_approximate_eigenvalue(A, x);
    } while(openMP_convergence(lambda_new, lambda_old, 0.00001));

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
bool openMP_convergence(double lambda_new, double lambda_old, double threshold){
    return (fabs(lambda_new - lambda_old) > threshold);
}

/**
 * @brief Computes the matrix-vector multiplication.
 * 
 * @param A The input matrix.
 * @param x The input/output vector. It is overwritten with the result A * x.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */

void openMP_matvec_mult(Matrix* A, Vector* x){

    double* temp = malloc(sizeof(double) * x->size);
    double sum;
    // parallize the outer for loop
    #pragma omp parallel for default(none) private(sum) shared(temp, A, x)
    for(int i = 0; i < A->rows; i++){
        sum = 0;
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum += value * x->data[j];
        }
        int thread_id = omp_get_thread_num();
        printf("Thread %d is processing row %d\n", thread_id, i);
        temp[i] = sum;
    }
    //TODO: parallize this part also
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
void openMP_normalize_vector(Vector* x){

    double norm = sqrt(openMP_dot_product(x, x));
    if (norm == 0) return;

    #pragma omp parallel for default(none) shared(x, norm)
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
double openMP_approximate_eigenvalue(Matrix* A, Vector* x){
    Vector copy;
    copy.size = x->size;
    copy.data = malloc(sizeof(double) * copy.size);
    //TODO: test parallize this for loop
    for(int i = 0; i < copy.size; i++){
        copy.data[i] = x->data[i];
    }
    openMP_matvec_mult(A, &copy);
    double lambda = openMP_dot_product(x, &copy);

    free(copy.data);
    return lambda;
}



double openMP_dot_product(Vector* x, Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }
    double dot = 0;
    #pragma omp parallel for reduction(+:dot) shared(x,y)
    for(int i = 0; i < x->size; i++){
        dot += x->data[i] * y->data[i];
    }
    return dot;
}