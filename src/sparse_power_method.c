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
extern start_timer timer_start;
extern stop_timer timer_stop;

#define MAX_ITERATIONS 10000
#define NUM_RUNS 20

/**
 * @brief Calculates the dominant eigenvalue and its coresponding eigenvector of a matrix.
 * 
 * @param A The matrix.
 * 
 * @return The dominant eigenvalue of matrix A.
 */
Res sparse_power_method(const SparseMatrixAny *A){
    
    double lambda_old = 0;
    double lambda_new = 0;
    int size;

    if(A->type == CSR){
        size = A->mat.csr->rows;
    }else{
        size = A->mat.coo->rows;
    }

    // initial vector
    Vector* x = generate_1_vector(size);
    Vector* y = generate_vector(size);
    int iterations = 0;

    double start = timer_start();
    //y_1
    sparse_matvec(A,x,y);
    do{
        lambda_old = lambda_new;
        normalize_vector(y,x);
        lambda_new = sparse_approximate_eigenvalue(x, y);
        sparse_matvec(A,x,y);
        iterations += 1;
        
    } while(!convergence(lambda_new, lambda_old, 1.0E-6) && iterations < MAX_ITERATIONS);
    double time = timer_stop(start);
    
    Res result;

    if (iterations >= MAX_ITERATIONS) {
        // error result
        result.lambda = -1.0;
    } else{
        result.lambda = lambda_new;
        result.time = time;
        printf("Number of iterations: %d\n", iterations);
        printf("Lambda: %f\n", lambda_new);
    }
    delete_vector(x);
    delete_vector(y);
    return result;
}


/**
 * @brief  Approximates the dominant eigenvalue.
 * 
 * @param A The input matrix.
 * @param x The normalized input vector.
 * 
 * @return The approximated dominant eigenvalue.
 */

 double sparse_approximate_eigenvalue(Vector* x, Vector *y){
    
    //sparse_matvec(A, x, y);
    double lambda = dotprod(x, y);
    
    return lambda;
}


void test_sparse_power_method(SparseMatrixAny *A, char* file_name){
    double times[NUM_RUNS];
    double total_time = 0.0;

    Res warmup = sparse_power_method(A);
    if (warmup.lambda == -1.0) {
        printf("%s: did not converge\n", file_name);
        return;
    }

    for (int i = 0; i < NUM_RUNS; i++){
        Res result = sparse_power_method(A);
        if(result.lambda == -1.0){
            printf("%s: did not converge\n", file_name);
            return;
        }
        times[i] = result.time;
        total_time += result.time;

    }
    double avg = total_time / NUM_RUNS;
    double variance = 0.0;

    for (int i = 0; i < NUM_RUNS; i++) {
        variance += (times[i] - avg) * (times[i] - avg);
    }

    double stddev = sqrt(variance / (NUM_RUNS -1));
    printf("%s: avg time = %.6f s, stddev = %.6f s\n",file_name, avg, stddev);
}