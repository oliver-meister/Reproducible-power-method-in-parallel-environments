
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



int main(){

    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method Serial CUDA", NULL, NULL);

    CU_add_test(suite, "Test dot product CUDA", test_dot);


    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}