
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../../src/sparse_power_method.h"
#include "../../src/dense_power_method.h"
#include "../../src/common.h"
#include "../../src/CUDA_EXBLAS/cuda_exblas_fun.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../../external/mmio.h"


void test_dot(){
    Vector x;
    x.size = 1000;
    x.data = malloc(sizeof(double) * 1000);

    Vector y;
    y.size = 1000;
    y.data = malloc(sizeof(double) * 1000);

    for(int i = 0; i < 1000; i++){
        x.data[i] = (double)i;
        y.data[i] = (double)(2*i);
    }

    double dot = cuda_ExBLAS_dot_product(&x,&y);
    CU_ASSERT_DOUBLE_EQUAL(dot, 665667000, 1e-6)
    free(x.data);
    free(y.data);
}

void test_EXBLAS_norm(){

    Vector x;
    x.size = 2;
    x.data = malloc(sizeof(double) * 2);
    x.data[0] = 3.0;
    x.data[1] = 4.0;

    normalize_vector(&x);
    CU_ASSERT_DOUBLE_EQUAL(x.data[0], 0.6, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[1], 0.8, 0.0001);
    free(x.data);

}


void test_dense_EXBLAS_approximate_eigenvalue(){
    
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
    
    double lambda = dense_approximate_eigenvalue(&A, &x, true);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 2.0, 0.0001);
    free(A.data);
    free(x.data);
    
}

void test_EXBLAS_sparse_CSR_large_power_method(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double lambda = sparse_power_method(A);
    printf("cuda dot: %f", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.14176, 1.0E-4);

    free(my_coo->row);
    free(my_coo->col);
    free(my_coo->val);
    free(my_coo);
    
    free(my_csr->row_ptr);
    free(my_csr->col);
    free(my_csr->val);
    free(my_csr);
    
    free(A);

}


void test_dense_EXBLAS_power_method(){
    denseMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.data = malloc(sizeof(double) * 4);
    A.data[0] = 2.0; 
    A.data[1] = 1.0; 
    A.data[2] = 1.0; 
    A.data[3] = 3.0; 

    double lambda = dense_power_method(&A);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 3.6180, 0.001);
    free(A.data);
}



int main(){
    
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial CUDA", NULL, NULL);

    //CU_add_test(suite, "Vector normalization test", test_EXBLAS_norm);
    CU_add_test(suite, "Test dot product CUDA", test_dot);
    //CU_add_test(suite, "Test CSR powermethod", test_EXBLAS_sparse_CSR_large_power_method);
    //CU_add_test(suite, "Approximate eigenvalue test", test_dense_EXBLAS_approximate_eigenvalue);    
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}