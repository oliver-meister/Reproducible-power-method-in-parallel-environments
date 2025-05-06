
#include <stdbool.h>
#include <math.h>
#include "common.h"
#include "serial/serial_fun.h"
#include "openMP/omp_fun.h"
#include "OMP_Offload/off_fun.h"
#include "CUDA/cuda_fun.h"
#include "../include/matrix.h"
#include "../include/vector.h"

dot_fn dotprod;
sparse_matvec_fn sparse_matvec;
sparse_eigen_fn sparse_eigen;
dense_matvec_fn dense_matvec;
dense_eigen_fn dense_eigen;


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
void normalize_vector(Vector* x){

    #ifdef USE_OMP
        dotprod = openMP_dot_product;
    #elif defined(USE_OFF)
        dotprod = off_dot_product;
    #elif defined(USE_CUDA)
        dotprod = cuda_dot_product;
    #else
        dotprod = serial_dot_product;
    #endif

    double norm = sqrt(dotprod(x,x));
    if (norm == 0) return;
    for(int i = 0; i < x->size; i++){
        x->data[i] =  x->data[i] / norm;
    }
}