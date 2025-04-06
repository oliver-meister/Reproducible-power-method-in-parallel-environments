#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include "../../src/serial/power_method_serial.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"


//Tests for dense matrices.

// works
void test_serial_dense_matvec_mult(){

    denseMatrix A;
    A.rows = 5;
    A.cols = 5;
    A.data = malloc(sizeof(double) * 25);
    A.data[0] = 4.0; 
    A.data[1] = 1.0; 
    A.data[2] = 0.0; 
    A.data[3] = 0.0;
    A.data[4] = 0.0; 

    A.data[5] = 1.0; 
    A.data[6] = 3.0; 
    A.data[7] = 1.0; 
    A.data[8] = 0.0; 
    A.data[9] = 0.0; 

    A.data[10] = 0.0; 
    A.data[11] = 1.0; 
    A.data[12] = 2.0; 
    A.data[13] = 1.0; 
    A.data[14] = 0.0; 

    A.data[15] = 0.0; 
    A.data[16] = 0.0; 
    A.data[17] = 1.0; 
    A.data[18] = 3.0; 
    A.data[19] = 1.0; 

    A.data[20] = 0.0; 
    A.data[21] = 0.0; 
    A.data[22] = 0.0; 
    A.data[23] = 1.0; 
    A.data[24] = 4.0; 

    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -1;
    x.data[1] = 1;
    x.data[2] = -1;
    x.data[3] = 1;
    x.data[4] = 1;

    double* test_array = malloc(sizeof(double) * 5);

    test_array[0] = 5;
    test_array[1] = 5;
    test_array[2] = 4;
    test_array[3] = 5;
    test_array[4] = 5;


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
    A.rows = 5;
    A.cols = 5;
    A.data = malloc(sizeof(double) * 25);
    A.data[0] = 4.0; 
    A.data[1] = 1.0; 
    A.data[2] = 0.0; 
    A.data[3] = 0.0;
    A.data[4] = 0.0; 
    A.data[5] = 1.0; 
    A.data[6] = 3.0; 
    A.data[7] = 1.0; 
    A.data[8] = 0.0; 
    A.data[9] = 0.0; 
    A.data[10] = 0.0; 
    A.data[11] = 1.0; 
    A.data[12] = 2.0; 
    A.data[13] = 1.0; 
    A.data[14] = 0.0; 
    A.data[15] = 0.0; 
    A.data[16] = 0.0; 
    A.data[17] = 1.0; 
    A.data[18] = 3.0; 
    A.data[19] = 1.0; 
    A.data[20] = 0.0; 
    A.data[21] = 0.0; 
    A.data[22] = 0.0; 
    A.data[23] = 1.0; 
    A.data[24] = 4.0; 

    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = 1;
    x.data[1] = 1;
    x.data[2] = 1;
    x.data[3] = 1;
    x.data[4] = 1;

    serial_normalize_vector(&x);

    double lambda = serial_dense_approximate_eigenvalue(&A, &x);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 4.8, 0.0001);
    free(A.data);
    free(x.data);

}


test_serial_dense_power_method(){
    
    denseMatrix A;
    A.rows = 5;
    A.cols = 5;
    A.data = malloc(sizeof(double) * 25);
    A.data[0] = 4.0; 
    A.data[1] = 1.0; 
    A.data[2] = 0.0; 
    A.data[3] = 0.0;
    A.data[4] = 0.0; 
    A.data[5] = 1.0; 
    A.data[6] = 3.0; 
    A.data[7] = 1.0; 
    A.data[8] = 0.0; 
    A.data[9] = 0.0; 
    A.data[10] = 0.0; 
    A.data[11] = 1.0; 
    A.data[12] = 2.0; 
    A.data[13] = 1.0; 
    A.data[14] = 0.0; 
    A.data[15] = 0.0; 
    A.data[16] = 0.0; 
    A.data[17] = 1.0; 
    A.data[18] = 3.0; 
    A.data[19] = 1.0; 
    A.data[20] = 0.0; 
    A.data[21] = 0.0; 
    A.data[22] = 0.0; 
    A.data[23] = 1.0; 
    A.data[24] = 4.0; 

    double lambda = serial_dense_power_method(&A);
    printf("Eigenvalue dense: %f\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 5.236, 0.001);
    free(A.data);
}


///////////////////////////////////////////////////////////////

//Tests for sparse matrices.

void test_serial_sparse_matvec_mult(){
    int row[] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
    int col[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double val[] = {4, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1, 1, 4};
    int nnz = 13;

    sparseMatrix A = {row, col, val, 3, 3, nnz};
    
    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = 1;
    x.data[1] = 1;
    x.data[2] = 1;
    x.data[3] = 1;
    x.data[4] = 1;

    double* test_array = malloc(sizeof(double) * 5);

    test_array[0] = 5;
    test_array[1] = 5;
    test_array[2] = 4;
    test_array[3] = 5;
    test_array[4] = 5;

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

    int row[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4};
    int col[] = {0, 1, 0, 1, 1, 2, 3, 2, 3, 3, 4};
    double val[] = {4, 1, 1, 3, 1, 2, 1, 1, 3, 1, 4};
    sparseMatrix A = {row, col, val, 5, 5, 11};

    double lambda = serial_sparse_power_method(&A);
    printf("Eigenvalue sparse: %f\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 11.372, 0.001);


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
    //CU_add_test(suite, "Matrix vector multiplication dense test", test_serial_dense_matvec_mult);
    //CU_add_test(suite, "Approximate eigenvalue dense test", test_serial_dense_approximate_eigenvalue);
    CU_add_test(suite, "Power method dense test", test_serial_dense_power_method);

    // Tests for sparse matrices.
    //CU_add_test(suite, "Matrix vector multiplication sparse test", test_serial_sparse_matvec_mult);
    //CU_add_test(suite, "Approximate eigenvalue sparse test", test_serial_sparse_approximate_eigenvalue);
    CU_add_test(suite, "Power method sparse test", test_serial_sparse_power_method);

    // Tests for common functions.
    CU_add_test(suite, "Generate random vector test", test_serial_generate_random_vector);
    CU_add_test(suite, "Vector normalization test", test_serial_norm);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}