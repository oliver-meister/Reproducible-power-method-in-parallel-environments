#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include "../../src/openMP/power_method_openMP.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <omp.h>

void test_dense_openMP_matvec_mult(){

    denseMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.data = malloc(sizeof(double) * 4);
    A.data[0] = 2;
    A.data[1] = 4;
    A.data[2] = 2;
    A.data[3] = 3;

    Vector x;
    x.size = 2;
    x.data = malloc(sizeof(double) * 2);
    x.data[0] = 2;
    x.data[1] = 1;

    double* test_array = malloc(sizeof(double) * 2);

    test_array[0] = 8;
    test_array[1] = 7;
    openMP_dense_matvec_mult(&A, &x);
    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
    free(A.data);
    free(x.data);
    free(test_array);
}

test_openMP_norm(){

    Vector x;
    x.size = 2;
    x.data = malloc(sizeof(double) * 2);
    x.data[0] = 3.0;
    x.data[1] = 4.0;

    openMP_normalize_vector(&x);
    CU_ASSERT_DOUBLE_EQUAL(x.data[0], 0.6, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[1], 0.8, 0.0001);
    free(x.data);

}

test_dense_openMP_approximate_eigenvalue(){

    denseMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.data = malloc(sizeof(double) * 4);
    A.data[0] = 2;
    A.data[1] = 0; 
    A.data[2] = 0; 
    A.data[3] = 3;

    Vector x;
    x.size = 2;
    x.data = malloc(sizeof(double) * 2);
    x.data[0] = 1;
    x.data[1] = 0;

    double lambda = openMP_dense_approximate_eigenvalue(&A, &x);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 2.0, 0.0001);
    free(A.data);
    free(x.data);

}

test_openMP_generate_random_vector(){
    Vector* x = generate_random_vector(2);
    CU_ASSERT_EQUAL(2, x->size);
    for(int i = 0; i < x->size; i++){
        //printf("index %d: %f\n", i, x->data[i]);
        CU_ASSERT(x->data[i] >= -1.0 && x->data[i] <= 1.0);
    }
    free(x->data);
    free(x);
}

test_dense_openMP_power_method(){
    denseMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.data = malloc(sizeof(double) * 4);
    A.data[0] = 2.0; 
    A.data[1] = 1.0; 
    A.data[2] = 1.0; 
    A.data[3] = 3.0; 

    double lambda = openMP_dense_power_method(&A);
    printf("Eigenvalue %lg\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 3.6180, 0.001);
    free(A.data);
}

int main(int argc, char* argv[]){

    if(argc < 2){
        printf("expected number of threads as argumet\n");
        return 1;
    }

    int num_threads = atoi(argv[1]);

    if (num_threads <= 0) {
        printf("Error: num_threads must be greater than 0\n");
        return 1;
    }
    
    omp_set_num_threads(num_threads);
    
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method openMP Tests", NULL, NULL);
    // Tests for dense matrices.
    CU_add_test(suite, "Matrix vector multiplication test", test_dense_openMP_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue test", test_dense_openMP_approximate_eigenvalue);
    CU_add_test(suite, "Power method test", test_dense_openMP_power_method);
    // Tests for sparse matrices.

    // Tests for common functions.
    CU_add_test(suite, "Vector normalization test", test_openMP_norm);
    CU_add_test(suite, "Generate random vector test", test_openMP_generate_random_vector);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}