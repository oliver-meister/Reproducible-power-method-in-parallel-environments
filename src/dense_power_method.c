#include "dense_power_method.h"
#include <stdio.h>
#include <stdbool.h>
#include "../include/matrix.h"
#include "../include/vector.h"
#include "serial/serial_fun.h"
#include "openMP/omp_fun.h"
#include "OMP_Offload/off_fun.h"
#include "CUDA/cuda_fun.h"
#include "CUDA_ExBLAS/cuda_exblas_fun.h"
#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAX_ITERATIONS 1000

extern dot_fn dotprod;
extern dense_matvec_fn dense_matvec;

// Dense power method

/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double dense_power_method(const denseMatrix* A){

    // initial vector
    double lambda_old = 0;
    double lambda_new = 0;
    Vector* x = generate_random_vector(A->rows);
    Vector* y = generate_vector(A->rows);
    int iterations = 0;
    clock_t start = clock();
    do{
        lambda_old = lambda_new;
        dense_matvec(A,x,y);
        normalize_vector(y,x);
        lambda_new = dense_approximate_eigenvalue(x, y);

    } while(!convergence(lambda_new, lambda_old, 0.00001) && iterations < MAX_ITERATIONS);

    clock_t end = clock();
    double time = (double) (end - start) / CLOCKS_PER_SEC;

    if (iterations >= MAX_ITERATIONS) {
        printf("Warning: Power method did not converge within max iterations.\n");
    } else{
        printf("Number of iterations: %d\n", iterations);
        printf("Execution time: %f\n", time);
    }
   delete_vector(x);
   delete_vector(y);
    return lambda_new;
}


/**
 * @brief  Approximates the dominant eigenvalue.
 * 
 * @param A The input matrix.
 * @param x The normalized input vector.
 * 
 * @return The approximated dominant eigenvalue.
 */
double dense_approximate_eigenvalue(Vector* x, Vector *y){
    double lambda = dotprod(x,y);
    return lambda;
}


