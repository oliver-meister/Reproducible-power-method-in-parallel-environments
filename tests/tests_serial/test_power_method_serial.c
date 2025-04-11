#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../src/serial/power_method_serial.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../../external/mmio.h"


//Tests for dense matrices.

// Works
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

    test_array[0] = -3;
    test_array[1] = 1;
    test_array[2] = 0;
    test_array[3] = 3;
    test_array[4] = 5;


    serial_dense_matvec_mult(&A, &x);
    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
    free(A.data);
    free(x.data);
    free(test_array);
}

// Works
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
    x.data[0] = -1;
    x.data[1] = 1;
    x.data[2] = -1;
    x.data[3] = 1;
    x.data[4] = 1;

    double lambda = serial_dense_approximate_eigenvalue(&A, &x);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 12, 0.0001);
    free(A.data);
    free(x.data);

}

// Works
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
    CU_ASSERT_DOUBLE_EQUAL(lambda, 4.8608, 0.001);
    free(A.data);
}


///////////////////////////////////////////////////////////////

//Tests for sparse matrices.


// Works
void test_serial_sparse_COO_matvec_mult(){
    
    sparseMatrixCOO *coo = malloc(sizeof(sparseMatrixCOO));
    coo->rows = 5;
    coo->cols = 5;
    coo->nnz = 13;
    coo->row = malloc(sizeof(int) * coo->nnz);
    coo->col = malloc(sizeof(int) * coo->nnz);
    coo->val = malloc(sizeof(double) * coo->nnz);

    coo->row[0] = 0;
    coo->row[1] = 0;
    coo->row[2] = 1;
    coo->row[3] = 1;
    coo->row[4] = 1;
    coo->row[5] = 2;
    coo->row[6] = 2;
    coo->row[7] = 2;
    coo->row[8] = 3;
    coo->row[9] = 3;
    coo->row[10] = 3;
    coo->row[11] = 4;
    coo->row[12] = 4;

    coo->col[0] = 0;
    coo->col[1] = 1;
    coo->col[2] = 0;
    coo->col[3] = 1;
    coo->col[4] = 2;
    coo->col[5] = 1;
    coo->col[6] = 2;
    coo->col[7] = 3;
    coo->col[8] = 2;
    coo->col[9] = 3;
    coo->col[10] = 4;
    coo->col[11] = 3;
    coo->col[12] = 4;

    coo->val[0] = 4;
    coo->val[1] = 1;
    coo->val[2] = 1;
    coo->val[3] = 3;
    coo->val[4] = 1;
    coo->val[5] = 1;
    coo->val[6] = 2;
    coo->val[7] = 1;
    coo->val[8] = 1;
    coo->val[9] = 3;
    coo->val[10] = 1;
    coo->val[11] = 1;
    coo->val[12] = 4;

  
    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -1;
    x.data[1] = 1;
    x.data[2] = -1;
    x.data[3] = 1;
    x.data[4] = 1;

    double* test_array = malloc(sizeof(double) * 5);

    test_array[0] = -3;
    test_array[1] = 1;
    test_array[2] = 0;
    test_array[3] = 3;
    test_array[4] = 5;

    serial_sparse_matvec_mult_COO(coo, &x);

    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
   
    free(coo->col);
    free(coo->row);
    free(coo->val);
    free(coo);
    free(x.data);
    free(test_array);
}



void test_serial_sparse_CSR_matvec_mult(){
    
    sparseMatrixCOO *coo = malloc(sizeof(sparseMatrixCOO));
    coo->rows = 5;
    coo->cols = 5;
    coo->nnz = 13;
    coo->row = malloc(sizeof(int) * coo->nnz);
    coo->col = malloc(sizeof(int) * coo->nnz);
    coo->val = malloc(sizeof(double) * coo->nnz);

    coo->row[0] = 0;
    coo->row[1] = 0;
    coo->row[2] = 1;
    coo->row[3] = 1;
    coo->row[4] = 1;
    coo->row[5] = 2;
    coo->row[6] = 2;
    coo->row[7] = 2;
    coo->row[8] = 3;
    coo->row[9] = 3;
    coo->row[10] = 3;
    coo->row[11] = 4;
    coo->row[12] = 4;

    coo->col[0] = 0;
    coo->col[1] = 1;
    coo->col[2] = 0;
    coo->col[3] = 1;
    coo->col[4] = 2;
    coo->col[5] = 1;
    coo->col[6] = 2;
    coo->col[7] = 3;
    coo->col[8] = 2;
    coo->col[9] = 3;
    coo->col[10] = 4;
    coo->col[11] = 3;
    coo->col[12] = 4;

    coo->val[0] = 4;
    coo->val[1] = 1;
    coo->val[2] = 1;
    coo->val[3] = 3;
    coo->val[4] = 1;
    coo->val[5] = 1;
    coo->val[6] = 2;
    coo->val[7] = 1;
    coo->val[8] = 1;
    coo->val[9] = 3;
    coo->val[10] = 1;
    coo->val[11] = 1;
    coo->val[12] = 4;

    sparseMatrixCSR* csr = coo_to_csr(coo);

    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -1;
    x.data[1] = 1;
    x.data[2] = -1;
    x.data[3] = 1;
    x.data[4] = 1;

    double* test_array = malloc(sizeof(double) * 5);

    test_array[0] = -3;
    test_array[1] = 1;
    test_array[2] = 0;
    test_array[3] = 3;
    test_array[4] = 5;

    serial_sparse_matvec_mult_CSR(csr, &x);

    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
   
    free(coo->col);
    free(coo->row);
    free(coo->val);
    free(coo);
    free(csr->col);
    free(csr->row_ptr);
    free(csr->val);
    free(csr);
    free(x.data);
    free(test_array);
}





//Works
test_serial_sparse_approximate_eigenvalue(){

    sparseMatrixCOO *coo = malloc(sizeof(sparseMatrixCOO));
    coo->rows = 5;
    coo->cols = 5;
    coo->nnz = 13;
    coo->row = malloc(sizeof(int) * coo->nnz);
    coo->col = malloc(sizeof(int) * coo->nnz);
    coo->val = malloc(sizeof(double) * coo->nnz);

    coo->row[0] = 0;
    coo->row[1] = 0;
    coo->row[2] = 1;
    coo->row[3] = 1;
    coo->row[4] = 1;
    coo->row[5] = 2;
    coo->row[6] = 2;
    coo->row[7] = 2;
    coo->row[8] = 3;
    coo->row[9] = 3;
    coo->row[10] = 3;
    coo->row[11] = 4;
    coo->row[12] = 4;

    coo->col[0] = 0;
    coo->col[1] = 1;
    coo->col[2] = 0;
    coo->col[3] = 1;
    coo->col[4] = 2;
    coo->col[5] = 1;
    coo->col[6] = 2;
    coo->col[7] = 3;
    coo->col[8] = 2;
    coo->col[9] = 3;
    coo->col[10] = 4;
    coo->col[11] = 3;
    coo->col[12] = 4;

    coo->val[0] = 4;
    coo->val[1] = 1;
    coo->val[2] = 1;
    coo->val[3] = 3;
    coo->val[4] = 1;
    coo->val[5] = 1;
    coo->val[6] = 2;
    coo->val[7] = 1;
    coo->val[8] = 1;
    coo->val[9] = 3;
    coo->val[10] = 1;
    coo->val[11] = 1;
    coo->val[12] = 4;

    SparseMatrixAny *A = malloc(sizeof(SparseMatrixAny));
    A->type = COO;
    A->mat.coo = coo;
    
    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -1;
    x.data[1] = 1;
    x.data[2] = -1;
    x.data[3] = 1;
    x.data[4] = 1;


    double lambda = serial_sparse_approximate_eigenvalue(A, &x);
    printf("lambda: %lg\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 12.0, 0.0001);
    
    free(coo->col);
    free(coo->row);
    free(coo->val);
    free(coo);
    free(A);
    free(x.data);


}


test_serial_sparse_power_method(){

    sparseMatrixCOO *coo = malloc(sizeof(sparseMatrixCOO));
    coo->rows = 5;
    coo->cols = 5;
    coo->nnz = 13;
    coo->row = malloc(sizeof(int) * coo->nnz);
    coo->col = malloc(sizeof(int) * coo->nnz);
    coo->val = malloc(sizeof(double) * coo->nnz);

    coo->row[0] = 0;
    coo->row[1] = 0;
    coo->row[2] = 1;
    coo->row[3] = 1;
    coo->row[4] = 1;
    coo->row[5] = 2;
    coo->row[6] = 2;
    coo->row[7] = 2;
    coo->row[8] = 3;
    coo->row[9] = 3;
    coo->row[10] = 3;
    coo->row[11] = 4;
    coo->row[12] = 4;

    coo->col[0] = 0;
    coo->col[1] = 1;
    coo->col[2] = 0;
    coo->col[3] = 1;
    coo->col[4] = 2;
    coo->col[5] = 1;
    coo->col[6] = 2;
    coo->col[7] = 3;
    coo->col[8] = 2;
    coo->col[9] = 3;
    coo->col[10] = 4;
    coo->col[11] = 3;
    coo->col[12] = 4;

    coo->val[0] = 4;
    coo->val[1] = 1;
    coo->val[2] = 1;
    coo->val[3] = 3;
    coo->val[4] = 1;
    coo->val[5] = 1;
    coo->val[6] = 2;
    coo->val[7] = 1;
    coo->val[8] = 1;
    coo->val[9] = 3;
    coo->val[10] = 1;
    coo->val[11] = 1;
    coo->val[12] = 4;

    SparseMatrixAny *A = malloc(sizeof(SparseMatrixAny));
    A->type = COO;
    A->mat.coo = coo;

    double lambda = serial_sparse_power_method(A);
    printf("Eigenvalue sparse: %f\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 4.8608, 0.001);

    free(coo->col);
    free(coo->row);
    free(coo->val);
    free(coo);
    free(A);

}

test_serial_sparse_COO_large_power_method(){
 
    sparseMatrixCOO * my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
   
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = COO;
    A->mat.coo = my_coo;

    double lambda = serial_sparse_power_method(A);
    printf("Eigenvalue sparse: %f\n", lambda);
    CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.14176, 0.0001);

    free(my_coo->row);
    free(my_coo->col);
    free(my_coo->val);
    free(my_coo);
    free(A);


}

test_serial_sparse_CSR_large_power_method(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double lambda = serial_sparse_power_method(A);
    printf("Large test: Eigenvalue sparse: %f\n", lambda);
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

//////////////////////////////////////////////////////////////

// Works
test_serial_norm(){

    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -3;
    x.data[1] = 1;
    x.data[2] = 0;
    x.data[3] = 3;
    x.data[4] = 5;

    serial_normalize_vector(&x);

    CU_ASSERT_DOUBLE_EQUAL(x.data[0], -0.452267, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[1], 0.150756, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[2], 0.000000, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[3], 0.452267, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(x.data[4], 0.753778, 0.0001);
    free(x.data);

}

//Works
test_serial_generate_random_vector(){
    Vector* x = generate_random_vector(5);
    CU_ASSERT_EQUAL(5, x->size);
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
    CU_add_test(suite, "Matrix vector multiplication sparse COO test", test_serial_sparse_COO_matvec_mult);
    CU_add_test(suite, "Matrix vector multiplication sparse CSR test", test_serial_sparse_CSR_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue sparse test", test_serial_sparse_approximate_eigenvalue);
    CU_add_test(suite, "Power method sparse test", test_serial_sparse_power_method);
    CU_add_test(suite, "Power method sparse COO large test", test_serial_sparse_COO_large_power_method);
    CU_add_test(suite, "Power method sparse CSR large test", test_serial_sparse_CSR_large_power_method);

    //Tests for common functions.
    CU_add_test(suite, "Generate random vector test", test_serial_generate_random_vector);
    CU_add_test(suite, "Vector normalization test", test_serial_norm);

    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}