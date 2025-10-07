
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


/*
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

void test_CUDA_norm(){
    
Vector x;
x.size = 2;
x.data = malloc(sizeof(double) * 2);
x.data[0] = 3.0;
x.data[1] = 4.0;
Vector *y = generate_vector(2);
normalize_vector(&x,y);
CU_ASSERT_DOUBLE_EQUAL(y->data[0], 0.6, 0.0001);
CU_ASSERT_DOUBLE_EQUAL(y->data[1], 0.8, 0.0001);
free(x.data);
delete_vector(y);

}

void test_sparse_CUDA_matvec_mult(){
    
sparseMatrixCSR* my_csr = malloc(sizeof(sparseMatrixCSR));
my_csr->rows = 4;
my_csr->cols = 4;
my_csr->nnz = 5;
my_csr->row_ptr = malloc(sizeof(double) * my_csr->rows + 1);
my_csr->val = malloc(sizeof(double) * my_csr->nnz);
my_csr->col = malloc(sizeof(double) * my_csr->nnz);

my_csr->row_ptr[0] = 0;
my_csr->row_ptr[1] = 1;
my_csr->row_ptr[2] = 2;
my_csr->row_ptr[3] = 4;
my_csr->row_ptr[4] = 5;

my_csr->val[0] = 10;
my_csr->val[1] = 20;
my_csr->val[2] = 30;
my_csr->val[3] = 40;
my_csr->val[4] = 50;

my_csr->col[0] = 0;
my_csr->col[1] = 1;
my_csr->col[2] = 0;
my_csr->col[3] = 2;
my_csr->col[4] = 3;

Vector* x = malloc(sizeof(Vector));
x->size = my_csr->rows;
x->data = malloc(sizeof(double) * x->size);
x->data[0] = 1;
x->data[1] = 2;
x->data[2] = 3;
x->data[3] = 4;

double* test_vector = malloc(sizeof(double) * x->size);
test_vector[0] = 10;
test_vector[1] = 40;
test_vector[2] = 150;
test_vector[3] = 200;

Vector *y  = generate_vector(4);
cuda_sparse_matvec_mult_CSR(my_csr,x, y);

for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_vector[i], 1e-6);
}

free(test_vector);
free(x->data);
free(x);
free(my_csr->col);
free(my_csr->val);
free(my_csr->row_ptr);
free(my_csr);
delete_vector(y);

}


void test_dense_CUDA_matvec_mult(){
    
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

Vector *y = generate_vector(2);

cuda_dense_matvec_mult(&A, &x, y);
for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_array[i], 1e-6);
}
free(A.data);
free(x.data);
free(test_array);
delete_vector(y);
}

void test_dense_CUDA_approximate_eigenvalue(){
    
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

Vector *y = generate_vector(2);
double lambda = dense_approximate_eigenvalue(&A, &x, y, true);
CU_ASSERT_DOUBLE_EQUAL(lambda, 2.0, 0.0001);
free(A.data);
free(x.data);
delete_vector(y);

}

void test_CUDA_sparse_CSR_large_power_method(){
    
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



void test_dense_CUDA_power_method(){
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
*/


void test_serial_CSR_cage10(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cage10/cage10.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "cage10", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_venkat01(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/venkat01/venkat01.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "venkat01", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}


void test_serial_CSR_siH4(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/SiH4/SiH4.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "siH4", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_benzene(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/benzene/benzene.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "benzene", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_arc130(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/arc130/arc130.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "arc130", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}
void test_serial_CSR_ex19(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/ex19/ex19.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "ex19", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}
void test_serial_CSR_S80PI_n(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/S80PI_n/S80PI_n.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "S80PI_n", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_rdb3200l(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/rdb3200l/rdb3200l.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "rdb32001", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_jpwh_991(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/jpwh_991/jpwh_991.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "jpwh_991", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_bfwa782(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/bfwa782/bfwa782.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "bfwa782", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_cage8(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cage8/cage8.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "cage8", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_tub100(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/tub100/tub100.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "tub100", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_cdde2(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cdde2/cdde2.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "cdde2", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_str_0(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/str_0/str_0.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "str_0", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}



void test_serial_CSR_494_bus(){
    
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);
    
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "494_bus", thresholds[i]);
    }
    
    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_cit_HepTh(){
    
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cit-HepTh/cit-HepTh.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);
    
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "cit-HepTh", thresholds[i]);
    }    
    
    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_ca_GrQc(){
    
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/ca-GrQc/ca-GrQc.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);
    
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "ca-GrQc", thresholds[i]);
    }    
    
    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_web_Stanford(){
    
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/web-Stanford/web-Stanford.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);
    
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "web-Stanford", thresholds[i]);
    }    
    
    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}


void test_serial_CSR_SiO(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/SiO/SiO.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "SiO", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}


void test_serial_CSR_pkustk13(){
    
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/pkustk13/pkustk13.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);
    
    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "pkustk13", thresholds[i]);
    }
    
    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}

void test_serial_CSR_1138_bus(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/1138_bus/1138_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "1138_bus", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}
void test_serial_CSR_bcsstk01(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/bcsstk01/bcsstk01.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;
    double thresholds[6] = {1.0E-6, 1.0E-9, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15};
    for(int i = 0; i < 6; i++)
    {
        test_sparse_power_method(A, "bcsstk01", thresholds[i]);
    }

    delete_COO(my_coo);
    delete_CSR(my_csr);
    free(A);
}


int main(){
    init_backend();
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method CUDA ExBLAS", NULL, NULL);
/*
CU_add_test(suite, "Test dot product CUDA", test_dot);
CU_add_test(suite, "Vector normalization test", test_CUDA_norm);
CU_add_test(suite, "Test CSR powermethod", test_CUDA_sparse_CSR_large_power_method);
CU_add_test(suite, "sparse matrix-vector multi test", test_sparse_CUDA_matvec_mult);
CU_add_test(suite, "Matrix vector multiplication test", test_dense_CUDA_matvec_mult);
CU_add_test(suite, "Approximate eigenvalue test", test_dense_CUDA_approximate_eigenvalue);
*/

    CU_add_test(suite, "Power method venkat01", test_serial_CSR_venkat01);
    CU_add_test(suite, "Power method benzene", test_serial_CSR_benzene);
    CU_add_test(suite, "Power method arc130", test_serial_CSR_arc130);
    CU_add_test(suite, "Power method ex19", test_serial_CSR_ex19);
    CU_add_test(suite, "Power method S80PI_n", test_serial_CSR_S80PI_n);
    CU_add_test(suite, "Power method cage10", test_serial_CSR_cage10);
    CU_add_test(suite, "Power method 1138_bus", test_serial_CSR_1138_bus);
    CU_add_test(suite, "Power method SiO", test_serial_CSR_SiO);
    CU_add_test(suite, "Power method tub100", test_serial_CSR_tub100);
    CU_add_test(suite, "Power method siH4", test_serial_CSR_siH4);
    CU_add_test(suite, "Power method rdb3200l", test_serial_CSR_rdb3200l);
    CU_add_test(suite, "Power method jpwh_991", test_serial_CSR_jpwh_991);
    CU_add_test(suite, "Power method bfwa782", test_serial_CSR_bfwa782);
    CU_add_test(suite, "Power method cage8", test_serial_CSR_cage8);
    CU_add_test(suite, "Power method cdde2", test_serial_CSR_cdde2);
    CU_add_test(suite, "Power method str_0", test_serial_CSR_str_0);
    CU_add_test(suite, "Power method ca-GrQc ", test_serial_CSR_ca_GrQc);
    CU_add_test(suite, "Power method web-Stanford ", test_serial_CSR_web_Stanford);
    CU_add_test(suite, "Power method pkustk13", test_serial_CSR_pkustk13);
    CU_add_test(suite, "Power method bcsstk01", test_serial_CSR_bcsstk01);
    CU_add_test(suite, "Power method 494_bus", test_serial_CSR_494_bus);
    CU_add_test(suite, "Power method cit-HepTh ", test_serial_CSR_cit_HepTh);


    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}