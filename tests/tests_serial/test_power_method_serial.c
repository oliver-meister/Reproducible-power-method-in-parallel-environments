#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdlib.h>
#include "../../src/serial/power_method_serial.h"
#include "../../include/matrix.h"
#include "../../include/vector.h"

void test_matvec_mult(){

    Matrix A;
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
    serial_matvec_mult(&A, &x);
    for(int i = 0; i < x.size; i++){
        CU_ASSERT_DOUBLE_EQUAL(x.data[i], test_array[i], 1e-6);
    }
    free(A.data);
    free(x.data);
    free(test_array);
}

int main(){
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial Tests", NULL, NULL);
    CU_add_test(suite, "Matrix vector multiplication test", test_matvec_mult);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}