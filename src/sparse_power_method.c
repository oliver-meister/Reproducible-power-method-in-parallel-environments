#include "sparse_power_method.h"
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

extern dot_fn dotprod;
extern sparse_matvec_fn sparse_matvec;

#define MAX_ITERATIONS 1000


/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double sparse_power_method(const SparseMatrixAny *A){
    
    double lambda_old = 0;
    double lambda_new = 0;
    int size;

    if(A->type == CSR){
        size = A->mat.csr->rows;
    }else{
        size = A->mat.coo->rows;
    }

    // initial vector
    Vector* x = generate_random_vector(size);
    Vector* y = generate_vector(size);
    int iterations = 0;

    clock_t start = clock();
    do{
        iterations += 1;
        lambda_old = lambda_new;
        sparse_matvec(A,x,y);
        normalize_vector(y,x);
        lambda_new = sparse_approximate_eigenvalue(A, x, y);
        
    } while(!convergence(lambda_new, lambda_old, 1.0E-6) && iterations < MAX_ITERATIONS);

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

 double sparse_approximate_eigenvalue(const SparseMatrixAny* A, Vector* x, Vector *y){
    
    double lambda = dotprod(x, y);
    
    return lambda;
}


