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

// we write over to early
void serial_matvec_mult(Matrix* A, Vector* V){

    double* temp = malloc(sizeof(double) * V->size);
    double sum;
    for(int i = 0; i < A->rows; i++){
        sum = 0;
        for (int j = 0; j < A->cols; j++){
            // each row has cols elements
            double value = A->data[i * A->cols + j];
            sum += value * V->data[j];
        }
        temp[i] = sum;
    }
    for(int i = 0; i < V->size; i++){
        V->data[i] = temp[i];
    }
    free(temp);
}

// Normalize the vector
void serial_normalize_vector(Vector* V){

    double norm = sqrt(dot_product(V));
    if (norm == 0) return;
    for(int i = 0; i < V->size; i++){
        V->data[i] =  V->data[i] / norm;
    }
}

void serial_approximate_eigenvalue(){
    
}