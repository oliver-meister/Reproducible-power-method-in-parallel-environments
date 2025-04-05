#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include "../../src/serial/power_method_serial.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"


//Tests for dense matrices.

void test_serial_dense_matvec_mult(){

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
    serial_dense_matvec_mult(&A, &x);
    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
    free(A.data);
    free(x.data);
    free(test_array);
}


test_serial_dense_approximate_eigenvalue(){

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

    double lambda = serial_dense_approximate_eigenvalue(&A, &x);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 2.0, 0.0001);
    free(A.data);
    free(x.data);

}


test_serial_dense_power_method(){
    denseMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.data = malloc(sizeof(double) * 4);
    A.data[0] = 2.0; 
    A.data[1] = 1.0; 
    A.data[2] = 1.0; 
    A.data[3] = 3.0; 

    double lambda = serial_dense_power_method(&A);
    printf("Eigenvalue %f\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 3.6180, 0.001);
    free(A.data);
}


///////////////////////////////////////////////////////////////

//Tests for sparse matrices.

void test_serial_sparse_matvec_mult(){
    int row[] = {0, 0, 1, 2, 2};
    int col[] = {0, 2, 1, 0, 2};
    double val[] = {2, 1, 3, 1, 2};

    sparseMatrix A = {row, col, val, 3, 3, 5};
    
    Vector x;
    x.size = 3;
    x.data = malloc(sizeof(double) * 3);
    x.data[0] = 1;
    x.data[1] = 0;
    x.data[2] = 2;

    double* test_array = malloc(sizeof(double) * 3);
    test_array[0] = 4;
    test_array[1] = 0;
    test_array[2] = 5;

    serial_sparse_matvec_mult(&A, &x);

    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
    free(x.data);
    free(test_array);

}

/*

2, 0, 1     1                    1
0, 3, 0     0    =   4 0 5   *   0   =   14
1, 0, 2     2                    2



*/

test_serial_sparse_approximate_eigenvalue(){

    int row[] = {0, 0, 1, 2, 2};
    int col[] = {0, 2, 1, 0, 2};
    double val[] = {2, 1, 3, 1, 2};

    sparseMatrix A = {row, col, val, 3, 3, 5};
    
    Vector x;
    x.size = 3;
    x.data = malloc(sizeof(double) * 3);
    x.data[0] = 1;
    x.data[1] = 0;
    x.data[2] = 2;


    double lambda = serial_sparse_approximate_eigenvalue(&A, &x);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 14.0, 0.0001);
    free(x.data);


}


test_serial_sparse_power_method(){

}

///////////////////////////////////////////////////////////////
test_serial_norm(){

    Vector x;
    x.size = 2;
    x.data = malloc(sizeof(double) * 2);
    x.data[0] = 3.0;
    x.data[1] = 4.0;

    serial_normalize_vector(&x);
    CU_ASSERT_DOUBLE_EQUAL(x.data[0], 0.6, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[1], 0.8, 0.0001);
    free(x.data);

}

test_serial_generate_random_vector(){
    Vector* x = generate_random_vector(2);
    CU_ASSERT_EQUAL(2, x->size);
    for(int i = 0; i < x->size; i++){
        CU_ASSERT(x->data[i] >= -1.0 && x->data[i] <= 1.0);
    }
    free(x->data);
    free(x);
}

int main(){
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial Tests", NULL, NULL);

    // Tests for dense matrices.
    CU_add_test(suite, "Matrix vector multiplication dense test", test_serial_dense_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue dense test", test_serial_dense_approximate_eigenvalue);
    CU_add_test(suite, "Power method dense test", test_serial_dense_power_method);

    // Tests for sparse matrices.
    CU_add_test(suite, "Matrix vector multiplication sparse test", test_serial_sparse_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue sparse test", test_serial_sparse_approximate_eigenvalue);
    //CU_add_test(suite, "Power method sparse test", test_serial_sparse_power_method);

    // Tests for common functions.
    CU_add_test(suite, "Generate random vector test", test_serial_generate_random_vector);
    CU_add_test(suite, "Vector normalization test", test_serial_norm);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}