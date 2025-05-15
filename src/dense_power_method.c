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
    #ifdef USE_OMP
        printf("in OMP def\n");
        dotprod = openMP_dot_product;
        dense_matvec = openMP_dense_matvec_mult;
    #elif defined(USE_OFF)
        printf("in OFF 1 def\n");
        dotprod = off_dot_product;
        dense_matvec = off_dense_matvec_mult;
    #elif defined(USE_CUDA)
        printf("in CUDA def\n");
        dotprod = cuda_dot_product;
        dense_matvec = cuda_dense_matvec_mult;
    #elif defined(USE_EXBLAS)
        printf("in EXBLAS def\n");
        dotprod = cuda_ExBLAS_dot_product;
        dense_matvec = cuda_dense_matvec_mult;
    #else
        printf("in SERIAL def\n");
        dotprod = serial_dot_product;
        dense_matvec = serial_dense_matvec_mult;
    #endif
    
    // initial vector
    double lambda_old = 0;
    double lambda_new = 0;
    Vector* x = generate_random_vector(A->rows);
    Vector* y = generate_vector(A->rows);
  

    do{
        lambda_old = lambda_new;
        dense_matvec(A,x,y);
        normalize_vector(y,x);
        lambda_new = dense_approximate_eigenvalue(A, x, y, false);

    } while(!convergence(lambda_new, lambda_old, 0.00001));

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
double dense_approximate_eigenvalue(const denseMatrix* A, Vector* x, Vector *y, bool test){

    if (test){
        #ifdef USE_OMP
        printf("in OMP def\n");
        dotprod = openMP_dot_product;
        dense_matvec = openMP_dense_matvec_mult;
        #elif defined(USE_OFF)
            printf("in OFF 2 def\n");
            dotprod = off_dot_product;
            dense_matvec = off_dense_matvec_mult;
        #elif defined(USE_CUDA)
            printf("in CUDA def\n");
            dotprod = cuda_dot_product;
            dense_matvec = cuda_dense_matvec_mult;
        #elif defined(USE_EXBLAS)
            printf("in EXBLAS def\n");
            dotprod = cuda_ExBLAS_dot_product;
            dense_matvec = cuda_dense_matvec_mult;
        #else
            printf("in SERIAL def\n");
            dotprod = serial_dot_product;
            dense_matvec = serial_dense_matvec_mult;
        #endif
    }

    dense_matvec(A, x, y);
    double lambda = dotprod(x,y);
    return lambda;
}


