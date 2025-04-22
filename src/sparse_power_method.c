#include "sparse_power_method.h"
#include <stdio.h>
#include <stdbool.h>
#include "../include/matrix.h"
#include "../include/vector.h"
#include "serial/serial_fun.h"
#include "openMP/omp_fun.h"
#include "OMP_Offload/off_fun.h"
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
        dotprod = openMP_dot_product;
        sparse_matvec = openMP_sparse_matvec_mult;
    #elif defined(USE_OFF)
        dotprod = off_dot_product;
        sparse_matvec = off_sparse_matvec_mult;
    #else
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

    do{
        lambda_old = lambda_new;
        sparse_matvec(A,x);
        normalize_vector(x);
        lambda_new = sparse_approximate_eigenvalue(A, x, false);
        
        //printf("sparse lambda approximation: %f\n", lambda_new);
    } while(!convergence(lambda_new, lambda_old, 0.000001));

    free(x->data);
    free(x);
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

 double sparse_approximate_eigenvalue(const SparseMatrixAny* A, const Vector* x, bool test){
    if (test){
        #ifdef USE_OMP
            dotprod = openMP_dot_product;
            sparse_matvec = openMP_sparse_matvec_mult;
        #elif defined(USE_OFF)
            dotprod = off_dot_product;
            sparse_matvec = off_sparse_matvec_mult;
        #else
            dotprod = serial_dot_product;
            sparse_matvec = serial_sparse_matvec_mult;
        #endif
    }
    Vector copy;
    copy.size = x->size;
    copy.data = malloc(sizeof(double) * copy.size);
    for(int i = 0; i < copy.size; i++){
        copy.data[i] = x->data[i];
    }
    sparse_matvec(A, &copy);
    double lambda = dotprod(x, &copy);
    
    free(copy.data);
    return lambda;
}
