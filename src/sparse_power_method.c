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
#include <cuda_runtime.h>

extern dot_fn dotprod;
extern sparse_matvec_fn sparse_matvec;
extern start_timer timer_start;
extern stop_timer timer_stop;

#define MAX_ITERATIONS 100000
#define NUM_RUNS 20

#ifdef USE_EXBLAS
    #define PARTIAL_SUPERACCS_COUNT 512
    #define BIN_COUNT 39
#endif

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

    // for CUDA allocate directly on the GPU // done
    // initial vector
    Vector* x = generate_1_vector(size);
    Vector* y = generate_vector(size);

    #ifdef USE_CUDA
        // Number of CUDA Blocks for dotproduct
        int numBlocks = 1;
        double *d_result;
        cudaMalloc((void**)&d_result, sizeof(double) * numBlocks);

    #elif defined(USE_EXBLAS)
        long long int* d_PartialSuperaccs;
        size_t size = PARTIAL_SUPERACCS_COUNT * BIN_COUNT * sizeof(long long int);
        cudaMalloc((void**)&d_PartialSuperaccs, size);
    #endif

    // allocate GPU memory for d_result
    int iterations = 0;

    double start = timer_start();
    //y_1
    sparse_matvec(A,x,y);

    do{
        lambda_old = lambda_new;
        #ifdef USE_CUDA
            normalize_vector_CUDA(y,x,d_result,numBlocks);
            lambda_new = sparse_approximate_eigenvalue_CUDA(x, y, d_result, numBlocks);
        #elif defined(USE_EXBLAS)
            normalize_vector_EXBLAS(y,x,d_PartialSuperaccs, size);
            lambda_new = sparse_approximate_eigenvalue_EXBLAS(x, y, d_PartialSuperaccs, size);
        #else 
            normalize_vector(y,x);
            lambda_new = sparse_approximate_eigenvalue(x, y);
        #endif
        sparse_matvec(A,x,y);
        iterations += 1;
        
    } while(!convergence(lambda_new, lambda_old, 1.0E-15) && iterations < MAX_ITERATIONS);
    double time = timer_stop(start);
    
    Res result;

    if (iterations >= MAX_ITERATIONS) {
        // error result
        result.lambda = -1.0;
    } else{
        result.lambda = lambda_new;
        result.time = time;
        result.iter = iterations;
        //printf("Number of iterations: %d\n", iterations);
        //printf("Lambda: %.16f\n", lambda_new);
    }
    delete_vector(x);
    delete_vector(y);
    #ifdef USE_CUDA
        cudaFree(d_result);
    #elif defined(USE_EXBLAS)
        cudaFree(d_PartialSuperaccs);
    #endif
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
    //printf("call from approx \n");
    double lambda = dotprod(x, y);
    //printf("ExDOT dot result, approx: %.20e\n", lambda);
    return lambda;
}


/**
 * @brief  Approximates the dominant eigenvalue.
 * 
 * @param A The input matrix.
 * @param x The normalized input vector.
 * 
 * @return The approximated dominant eigenvalue.
 */

 double sparse_approximate_eigenvalue_CUDA(Vector* x, Vector *y, double* d_result, int numBlocks){
    
    //sparse_matvec(A, x, y);
    //printf("call from approx \n");
    double lambda = cuda_dot_product(x, y, d_result, numBlocks);
    //printf("ExDOT dot result, approx: %.20e\n", lambda);
    return lambda;
}

/**
 * @brief  Approximates the dominant eigenvalue.
 * 
 * @param A The input matrix.
 * @param x The normalized input vector.
 * 
 * @return The approximated dominant eigenvalue.
 */

 double sparse_approximate_eigenvalue_EXBLAS(Vector* x, Vector *y, long long int* d_PartialSuperaccs, size_t size){
    double lambda = cuda_ExBLAS_dot_product(x, y, d_PartialSuperaccs, size);
    return lambda;
}

void test_sparse_power_method(SparseMatrixAny *A, char* file_name){
    double times[NUM_RUNS];
    double total_time = 0.0;
    int min_iter = 0;
    int max_iter = 0;
    double min_lambda = 0.0;
    double max_lambda = 0.0;

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

        if(i == 0)
        {
            min_iter = result.iter;
            max_iter = result.iter;
            
            min_lambda = result.lambda;
            max_lambda = result.lambda;
        }
        else{
            if(min_iter > result.iter){
                min_iter = result.iter;
            }
            if(max_iter < result.iter)
            {
                max_iter = result.iter;
            }
            if(min_lambda > result.lambda){
                min_lambda = result.lambda;
            }
            if(max_lambda < result.lambda)
            {
                max_lambda = result.lambda;
            }

        }

    }
    double avg = total_time / NUM_RUNS;
    double variance = 0.0;

    for (int i = 0; i < NUM_RUNS; i++) {
        variance += (times[i] - avg) * (times[i] - avg);
    }

    double stddev = sqrt(variance / (NUM_RUNS -1));
    printf("%s: avg time = %.6f s, stddev = %.6f s, max iter = %d, min iter = %d, max lambda = %.16f, min lambda = %.16f \n",file_name, avg, stddev, max_iter, min_iter, max_lambda, min_lambda);
}