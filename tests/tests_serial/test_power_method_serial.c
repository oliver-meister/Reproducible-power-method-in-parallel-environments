
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../../src/sparse_power_method.h"
#include "../../src/dense_power_method.h"
#include "../../src/common.h"
#include "../../src/serial/serial_fun.h"
#include "../../src/openMP/omp_fun.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include "../../external/mmio.h"


//Tests for dense matrices.

/*

void test_generate_sum_vector_dense(){
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
    
    double *x = malloc(sizeof(double) * 5);
    x[0] = 5;
    x[1] = 5;
    x[2] = 4;
    x[3] = 5;
    x[4] = 5;
    
    Vector *y = generate_sum_vector_dense(&A);
    
    for(int i = 0; i < y->size; i++){
        CU_ASSERT_DOUBLE_EQUAL(y->data[i], x[i], 1e-6);
    }
    
    delete_vector(y);
    free(x);
    free(A.data);
}

void test_generate_sum_vector_COO(){
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
    
    double *x = malloc(sizeof(double) * 5);
    x[0] = 5; 
    x[1] = 5;
    x[2] = 4;
    x[3] = 5;
    x[4] = 5;
    
    Vector *y = generate_sum_vector_COO(coo);
    
    for(int i = 0; i < y->size; i++){
        CU_ASSERT_DOUBLE_EQUAL(y->data[i], x[i], 1e-6);
    }
    
    delete_vector(y);
    free(x);
    free(coo->row);
    free(coo->col);
    free(coo->val);
    free(coo);
}

void test_generate_sum_vector_CSR(){
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
    
    double *x = malloc(sizeof(double) * 5);
    x[0] = 5; 
    x[1] = 5;
    x[2] = 4;
    x[3] = 5;
    x[4] = 5;
    
    Vector *y = generate_sum_vector_CSR(csr);
    
    for(int i = 0; i < y->size; i++){
        CU_ASSERT_DOUBLE_EQUAL(y->data[i], x[i], 1e-6);
    }
    
    delete_vector(y);
    free(x);
    free(coo->row);
    free(coo->col);
    free(coo->val);
    free(coo);
    free(csr->col);
    free(csr->row_ptr);
    free(csr->val);
    free(csr);
}

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

Vector *y = generate_vector(5);
serial_dense_matvec_mult(&A, &x, y);
for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_array[i], 1e-6);
}
free(A.data);
free(x.data);
free(test_array);
delete_vector(y);

}

// Works
void test_serial_dense_approximate_eigenvalue(){
    
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

Vector *y = generate_vector(5);
double lambda = dense_approximate_eigenvalue(&A, &x, y);

CU_ASSERT_DOUBLE_EQUAL(lambda, 12, 0.0001);
free(A.data);
free(x.data);
delete_vector(y);

}

// Works
void test_serial_dense_power_method(){
    
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

double lambda = dense_power_method(&A);
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

Vector *y = generate_vector(5);
serial_sparse_matvec_mult_COO(coo, &x, y);

for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_array[i], 1e-6);
}

free(coo->col);
free(coo->row);
free(coo->val);
free(coo);
free(x.data);
free(test_array);
delete_vector(y);
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

Vector *y = generate_vector(5);
serial_sparse_matvec_mult_CSR(csr, &x, y);

for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_array[i], 1e-6);
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
delete_vector(y);
}





//Works
void test_serial_sparse_approximate_eigenvalue(){
    
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

Vector *y = generate_vector(5);
double lambda = sparse_approximate_eigenvalue(A, &x, y);
CU_ASSERT_DOUBLE_EQUAL(lambda, 12.0, 0.0001);

free(coo->col);
free(coo->row);
free(coo->val);
free(coo);
free(A);
free(x.data);
delete_vector(y);


}

/*
void test_serial_sparse_power_method(){
    
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

double lambda = sparse_power_method(A);
CU_ASSERT_DOUBLE_EQUAL(lambda, 4.8608, 0.001);

free(coo->col);
free(coo->row);
free(coo->val);
free(coo);
free(A);

}
*/


/*
void test_serial_sparse_COO_large_power_method(){
    
sparseMatrixCOO * my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");

SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
A->type = COO;
A->mat.coo = my_coo;

double lambda = sparse_power_method(A);
CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.14176, 0.0001);

free(my_coo->row);
free(my_coo->col);
free(my_coo->val);
free(my_coo);
free(A);
}
*/
void test_serial_CSR_cage10(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cage10/cage10.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    test_sparse_power_method(A, "cage10");

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


void test_serial_CSR_494_bus(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

   test_sparse_power_method(A, "494_bus");

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


void test_serial_CSR_venkat01(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/venkat01/venkat01.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    test_sparse_power_method(A, "venkat01");

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

void test_serial_CSR_siH4(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/siH4/siH4.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    test_sparse_power_method(A, "siH4");

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

void test_serial_CSR_benzene(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/benzene/benzene.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    test_sparse_power_method(A, "benzene");

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

void test_serial_CSR_SiO(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/SiO/SiO.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

   test_sparse_power_method(A, "SiO");

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

void test_serial_CSR_bcsstk01(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/bcsstk01/bcsstk01.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

   test_sparse_power_method(A, "bcsstk01");

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

void test_serial_CSR_pkustk13(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/pkustk13/pkustk13.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    test_sparse_power_method(A, "pkustk13");

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
void test_serial_norm(){

    Vector x;
    x.size = 5;
    x.data = malloc(sizeof(double) * 5);
    x.data[0] = -3;
    x.data[1] = 1;
    x.data[2] = 0;
    x.data[3] = 3;
    x.data[4] = 5;

    Vector *y = generate_vector(5);
    normalize_vector(&x, y);

    CU_ASSERT_DOUBLE_EQUAL(y->data[0], -0.452267, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(y->data[1], 0.150756, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(y->data[2], 0.000000, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(y->data[3], 0.452267, 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(y->data[4], 0.753778, 0.0001);
    free(x.data);
    delete_vector(y);

}

//Works
void test_serial_generate_random_vector(){
    Vector* x = generate_random_vector(5);
    CU_ASSERT_EQUAL(5, x->size);
    for(int i = 0; i < x->size; i++){
        CU_ASSERT(x->data[i] >= -1.0 && x->data[i] <= 1.0);
    }
    free(x->data);
    free(x);
}

int main(){

    init_backend();

    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial Tests", NULL, NULL);
    // Tests for dense matrices.
    /*
    CU_add_test(suite, "generate dense sum vector test", test_generate_sum_vector_dense);
    CU_add_test(suite, "generate COO sum vector test", test_generate_sum_vector_COO);
    CU_add_test(suite, "generate CSR sum vector test", test_generate_sum_vector_CSR);
    CU_add_test(suite, "Matrix vector multiplication dense test", test_serial_dense_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue dense test", test_serial_dense_approximate_eigenvalue);
    CU_add_test(suite, "Power method dense test", test_serial_dense_power_method);

    // Tests for sparse matrices.
    CU_add_test(suite, "Matrix vector multiplication sparse COO test", test_serial_sparse_COO_matvec_mult);
    CU_add_test(suite, "Matrix vector multiplication sparse CSR test", test_serial_sparse_CSR_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue sparse test", test_serial_sparse_approximate_eigenvalue);
    CU_add_test(suite, "Power method sparse test", test_serial_sparse_power_method);
    CU_add_test(suite, "Power method sparse COO large test", test_serial_sparse_COO_large_power_method);
    
    */
    //CU_add_test(suite, "Power method cage10", test_serial_CSR_cage10);
    CU_add_test(suite, "Power method 494_bus", test_serial_CSR_494_bus);
    CU_add_test(suite, "Power method cage10", test_serial_CSR_venkat01);
    CU_add_test(suite, "Power method siH4", test_serial_CSR_siH4);
    CU_add_test(suite, "Power method benzene", test_serial_CSR_benzene);
    CU_add_test(suite, "Power method SiO", test_serial_CSR_SiO);
    CU_add_test(suite, "Power method bcsstk01", test_serial_CSR_bcsstk01);
    //CU_add_test(suite, "Power method pkustk13", test_serial_CSR_pkustk13);

    //Tests for common functions.
    //CU_add_test(suite, "Generate random vector test", test_serial_generate_random_vector);
    //CU_add_test(suite, "Vector normalization test", test_serial_norm);

    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}