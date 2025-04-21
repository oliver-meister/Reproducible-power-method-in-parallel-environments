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


void test_openMP_norm(){

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
    
    srand(time(0));
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Power Method openMP Tests", NULL, NULL);

    // Tests for common functions.
    CU_add_test(suite, "Vector normalization test", test_openMP_norm);
    CU_add_test(suite, "Generate random vector test", test_openMP_generate_random_vector);
    CU_add_test(suite, "dotproduct test", test_openMP_dotproduct);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}