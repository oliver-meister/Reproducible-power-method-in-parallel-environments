
#include <stdbool.h>
#include <math.h>
#include "common.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "serial/serial_fun.h"
#include "openMP/omp_fun.h"
#include "OMP_Offload/off_fun.h"
#include "CUDA/cuda_fun.h"
#include "CUDA_ExBLAS/cuda_exblas_fun.h"
#include "../include/matrix.h"
#include "../include/vector.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_SYNC()
#endif


dot_fn dotprod;

dense_matvec_fn dense_matvec;
sparse_matvec_fn sparse_matvec;

vector_norm_div_fun vector_norm_div;

start_timer timer_start;
stop_timer timer_stop;


double timer_cpu_start() {
    return (double)clock();
}

double timer_cpu_stop(double start) {
    return ((double)clock() - start) / CLOCKS_PER_SEC;
}

double timer_cuda_start() {
    CUDA_SYNC();
    return (double)clock();
}

double timer_cuda_stop(double start) {
    CUDA_SYNC();
    return ((double)clock() - start) / CLOCKS_PER_SEC;
}


double timer_omp_start() {
    return omp_get_wtime();
}

double timer_omp_stop(double start) {
    return omp_get_wtime() - start;
}


void init_backend() {
    #ifdef USE_OMP
        printf("Backend: OpenMP\n");
        dotprod = openMP_dot_product;
        vector_norm_div = openMP_vector_norm_div;
        dense_matvec = openMP_dense_matvec_mult;
        sparse_matvec = openMP_sparse_matvec_mult;
        timer_start = timer_omp_start;
        timer_stop = timer_omp_stop;
    #elif defined(USE_OFF)
        printf("Backend: OpenMP Offload\n");
        dotprod = off_dot_product;
        vector_norm_div = off_vector_norm_div;
        dense_matvec = off_dense_matvec_mult;
        sparse_matvec = off_sparse_matvec_mult;
        timer_start = timer_omp_start;
        timer_stop = timer_omp_stop;    
    #elif defined(USE_CUDA)
        printf("Backend: CUDA\n");
        dotprod = cuda_dot_product;
        vector_norm_div = cuda_vector_norm_div;
        dense_matvec = cuda_dense_matvec_mult;
        sparse_matvec = cuda_sparse_matvec_mult;
        timer_start = timer_cuda_start;
        timer_stop = timer_cuda_stop;
    #elif defined(USE_EXBLAS)
        printf("Backend: CUDA + ExBLAS\n");
        dotprod = cuda_ExBLAS_dot_product;
        vector_norm_div = cuda_vector_norm_div;
        dense_matvec = cuda_dense_matvec_mult;
        sparse_matvec = cuda_sparse_matvec_mult;
        timer_start = timer_cuda_start;
        timer_stop = timer_cuda_stop;
    #else
        printf("Backend: Serial\n");
        dotprod = serial_dot_product;
        vector_norm_div = serial_vector_norm_div;
        dense_matvec = serial_dense_matvec_mult;
        sparse_matvec = serial_sparse_matvec_mult;
        timer_start = timer_cpu_start;
        timer_stop = timer_cpu_stop;
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


    //printf("call from norm\n");
   double dot = dotprod(x, x);

    if (dot <= 1.0e-20 || isnan(dot)) {
        fprintf(stderr, "Warning: norm is too small or invalid, skipping normalization.\n");
        return;
    }

    double norm = sqrt(dot);
    vector_norm_div(x, y, norm);

}

