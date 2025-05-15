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



/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
double sparse_power_method(const SparseMatrixAny *A){
    
    #ifdef USE_OMP
        printf("in OMP def\n");
        dotprod = openMP_dot_product;
        sparse_matvec = openMP_sparse_matvec_mult;
    #elif defined(USE_OFF)
        printf("in OFF 3 def\n");
        dotprod = off_dot_product;
        sparse_matvec = off_sparse_matvec_mult;
    #elif defined(USE_CUDA)
        printf("in CUDA def\n");
        dotprod = cuda_dot_product;
        sparse_matvec = cuda_sparse_matvec_mult;
    #elif defined(USE_EXBLAS)
        printf("in EXBLAS def\n");
        dotprod = cuda_ExBLAS_dot_product;
        sparse_matvec = cuda_sparse_matvec_mult;
    #else
        printf("in SERIAL def\n");
        dotprod = serial_dot_product;
        sparse_matvec = serial_sparse_matvec_mult;

    #endif

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

    do{

        lambda_old = lambda_new;
        sparse_matvec(A,x,y);

        normalize_vector(y,x);
        lambda_new = sparse_approximate_eigenvalue(A, x, y, false);
        
        printf("lambda approximation: %f\n", lambda_new);
    } while(!convergence(lambda_new, lambda_old, 1.0E-6));

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

 double sparse_approximate_eigenvalue(const SparseMatrixAny* A, Vector* x, Vector *y, bool test){
    if (test){
        #ifdef USE_OMP
            printf("in OMP def\n");
            dotprod = openMP_dot_product;
            sparse_matvec = openMP_sparse_matvec_mult;
        #elif defined(USE_OFF)
            printf("in OFF 4 def\n");
            dotprod = off_dot_product;
            sparse_matvec = off_sparse_matvec_mult;
        #elif defined(USE_CUDA)
            printf("in CUDA def\n");
            dotprod = cuda_dot_product;
            sparse_matvec = cuda_sparse_matvec_mult;
        #elif defined(USE_EXBLAS)
            printf("in EXBLAS def\n");
            dotprod = cuda_ExBLAS_dot_product;
            sparse_matvec = cuda_sparse_matvec_mult;
        #else
            printf("in SERIAL def\n");
            dotprod = serial_dot_product;
            sparse_matvec = serial_sparse_matvec_mult;
        #endif
    }

   
    
    sparse_matvec(A, x, y);
    double lambda = dotprod(x, y);
    
    return lambda;
}


