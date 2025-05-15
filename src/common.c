
#include <stdbool.h>
#include <math.h>
#include "common.h"
#include <stdio.h>
#include "serial/serial_fun.h"
#include "openMP/omp_fun.h"
#include "OMP_Offload/off_fun.h"
#include "CUDA/cuda_fun.h"
#include "CUDA_ExBLAS/cuda_exblas_fun.h"
#include "../include/matrix.h"
#include "../include/vector.h"

dot_fn dotprod;

dense_matvec_fn dense_matvec;
sparse_matvec_fn sparse_matvec;

vector_norm_div_fun vector_norm_div;


void init_backend() {
    #ifdef USE_OMP
        printf("Backend: OpenMP\n");
        dotprod = openMP_dot_product;
        vector_norm_div = openMP_vector_norm_div;
        dense_matvec = openMP_dense_matvec_mult;
        sparse_matvec = openMP_sparse_matvec_mult;
    #elif defined(USE_OFF)
        printf("Backend: OpenMP Offload\n");
        dotprod = off_dot_product;
        vector_norm_div = off_vector_norm_div;
        dense_matvec = off_dense_matvec_mult;
        sparse_matvec = off_sparse_matvec_mult;
    #elif defined(USE_CUDA)
        printf("Backend: CUDA\n");
        dotprod = cuda_dot_product;
        vector_norm_div = cuda_vector_norm_div;
        dense_matvec = cuda_dense_matvec_mult;
        sparse_matvec = cuda_sparse_matvec_mult;
    #elif defined(USE_EXBLAS)
        printf("Backend: CUDA + ExBLAS\n");
        dotprod = cuda_ExBLAS_dot_product;
        vector_norm_div = cuda_vector_norm_div;
        dense_matvec = cuda_dense_matvec_mult;
        sparse_matvec = cuda_sparse_matvec_mult;
    #else
        printf("Backend: Serial\n");
        dotprod = serial_dot_product;
        vector_norm_div = serial_vector_norm_div;
        dense_matvec = serial_dense_matvec_mult;
        sparse_matvec = serial_sparse_matvec_mult;
    #endif
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
bool convergence(double lambda_new, double lambda_old, double threshold){
    return (fabs(lambda_new - lambda_old) < threshold);
}


/**
 * @brief Normalize the vector into a unit vector.
 * 
 * @param x The input/output vector. It is overwritten by the unit vector.
 * 
 * @return Nothing. The result is stored directly in the vector x.
 */
void normalize_vector(Vector* x, Vector *y){

    double norm = sqrt(dotprod(x,x));
    if (norm < 1.0E-10){
        return;
    }
    vector_norm_div(x,y,norm);

}