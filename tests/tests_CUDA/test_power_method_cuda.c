
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../../src/sparse_power_method.h"
#include "../../src/dense_power_method.h"
#include "../../src/common.h"
#include "../../src/CUDA/cuda_fun.h"
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

    double dot = cuda_dot_product(&x,&y);
    CU_ASSERT_DOUBLE_EQUAL(dot, 665667000, 1e-6)
    free(x.data);
    free(y.data);
}


void test_CUDA_sparse_CSR_large_power_method(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double lambda = sparse_power_method(A);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.14176, 0.0001);

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


int main(){

    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial CUDA", NULL, NULL);

    CU_add_test(suite, "Test dot product CUDA", test_dot);
    CU_add_test(suite, "Test CSR powermethod", test_CUDA_sparse_CSR_large_power_method);


    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}