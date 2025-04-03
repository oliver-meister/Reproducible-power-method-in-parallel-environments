#include "power_method_serial.h"
#include <stdio.h>
#include <stdbool.h>
#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <math.h>

// Matrix is of size n*n
#define SIZE 20


void serial_power_method(Matrix* A){
    
}


void serial_convergence(){
    
}

void serial_matvec_mult(Matrix* A, Vector* x){

    double* temp = malloc(sizeof(double) * x->size);
    double sum;
    for(int i = 0; i < A->rows; i++){
        sum = 0;
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum += value * x->data[j];
        }
        temp[i] = sum;
    }
    for(int i = 0; i < x->size; i++){
        x->data[i] = temp[i];
    }
    free(temp);
}

// Normalize the vector
void serial_normalize_vector(Vector* x){

    double norm = sqrt(dot_product(x, x));
    if (norm == 0) return;
    for(int i = 0; i < x->size; i++){
        x->data[i] =  x->data[i] / norm;
    }
}

// Aproximate eigenvalue
double serial_approximate_eigenvalue(Matrix* A, Vector* x){
    Vector copy;
    copy.size = x->size;
    copy.data = malloc(sizeof(double) * copy.size);
    for(int i = 0; i < copy.size; i++){
        copy.data[i] = x->data[i];
    }
    serial_matvec_mult(A, &copy);
    double lambda = dot_product(x, &copy);

    free(copy.data);
    return lambda;
}
