#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include <time.h>
#include "../../src/sparse_power_method.h"
#include "../../src/dense_power_method.h"
#include "../../src/common.h"
#include "../../src/serial/serial_fun.h"
#include "../../src/openMP/omp_fun.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <omp.h>


/*

void test_openMP_norm(){
    
Vector x;
x.size = 2;
x.data = malloc(sizeof(double) * 2);
x.data[0] = 3.0;
x.data[1] = 4.0;

Vector *y = generate_vector(2);
normalize_vector(&x, y);
CU_ASSERT_DOUBLE_EQUAL(y->data[0], 0.6, 0.0001);
CU_ASSERT_DOUBLE_EQUAL(y->data[1], 0.8, 0.0001);
free(x.data);
delete_vector(y);

}

void test_openMP_generate_random_vector(){
    Vector* x = generate_random_vector(2);
    CU_ASSERT_EQUAL(2, x->size);
    for(int i = 0; i < x->size; i++){
        CU_ASSERT(x->data[i] >= -1.0 && x->data[i] <= 1.0);
    }
    free(x->data);
    free(x);
}

void test_openMP_dotproduct(){
    Vector* x = malloc(sizeof(Vector));
    x->size = 10;
    x->data = malloc(sizeof(double) * x->size);
    
    x->data[0] = 1;
    x->data[1] = 2;
    x->data[2] = 3;
    x->data[3] = 4;
    x->data[4] = 5;
    x->data[5] = 6;
    x->data[6] = 7;
    x->data[7] = 8;
    x->data[8] = 9;
    x->data[9] = 10;
    
    double dot = openMP_dot_product(x, x);
    CU_ASSERT_DOUBLE_EQUAL(dot, 385, 0);
    
    free(x->data);
    free(x);
}

void test_openMP_sparse_CSR_large_power_method(){
    
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

void test_sparse_openMP_matvec_mult(){
    
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

Vector *y = generate_vector(4);
openMP_sparse_matvec_mult_CSR(my_csr, x, y);

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
*/

/*
void test_sparse_openMP_approximate_eigenvalue(){
    
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
    x->data[3] = 3;
}
*/

/*
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

Vector *y = generate_vector(2);
openMP_dense_matvec_mult(&A, &x, y);
for(int i = 0; i < y->size; i++){
    CU_ASSERT_DOUBLE_EQUAL(y->data[i], test_array[i], 1e-6);
}
free(A.data);
free(x.data);
free(test_array);
}

void test_dense_openMP_approximate_eigenvalue(){
    
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


void test_dense_openMP_power_method(){
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

void test_serial_CSR_494_bus(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/494_bus/494_bus.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141764, 1.0e-9);
    }
    printf("494_bus: average time: %.6f\n", total_time/10);

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

void test_serial_CSR_cage10(){
 
    sparseMatrixCOO *my_coo = createSparseMatrixCOO("ssget/cage10/cage10.mtx");
    sparseMatrixCSR *my_csr = coo_to_csr(my_coo);

    SparseMatrixAny * A = malloc(sizeof(SparseMatrixAny));
    A->type = CSR;
    A->mat.csr = my_csr;

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("venkat01: average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("siH4: average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("benzene: average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("SiO: average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("bcsstk01: average time: %.6f\n", total_time/10);

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

    double total_time = 0.0;
    for (int i = 0; i < 10; i++){
        Res result = sparse_power_method(A);
        double lambda = result.lambda;
        total_time += result.time;
        //CU_ASSERT_DOUBLE_EQUAL(lambda, 30005.141763, 1.0e-6);
    }
    printf("venkat01: average time: %.6f\n", total_time/10);

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
    init_backend();
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method openMP Tests", NULL, NULL);

    // Tests for common functions.
    /*
    CU_add_test(suite, "Vector normalization test", test_openMP_norm);
    CU_add_test(suite, "Generate random vector test", test_openMP_generate_random_vector);
    CU_add_test(suite, "dotproduct test", test_openMP_dotproduct);
    
    CU_add_test(suite, "Power method test large CSR", test_openMP_sparse_CSR_large_power_method);
    CU_add_test(suite, "sparse matrix-vector multi test", test_sparse_openMP_matvec_mult);
    
    CU_add_test(suite, "Matrix vector multiplication test", test_dense_openMP_matvec_mult);
    CU_add_test(suite, "Approximate eigenvalue test", test_dense_openMP_approximate_eigenvalue);
    CU_add_test(suite, "Power method test", test_dense_openMP_power_method);
    */
   
   //CU_add_test(suite, "Power method cage10", test_serial_CSR_cage10);
    //CU_add_test(suite, "Power method 494_bus", test_serial_CSR_494_bus);
    //CU_add_test(suite, "Power method cage10", test_serial_CSR_venkat01);
    //CU_add_test(suite, "Power method siH4", test_serial_CSR_siH4);
    //CU_add_test(suite, "Power method benzene", test_serial_CSR_benzene);
    //CU_add_test(suite, "Power method SiO", test_serial_CSR_SiO);
    //CU_add_test(suite, "Power method SiO", test_serial_CSR_bcsstk01);
    CU_add_test(suite, "Power method SiO", test_serial_CSR_pkustk13);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}