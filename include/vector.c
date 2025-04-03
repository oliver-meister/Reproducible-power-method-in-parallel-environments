#include "vector.h"
#include <stdio.h>


// if error, return 0.0
double dot_product(Vector* x, Vector* y){
    if(x->size != y->size){
        printf("Error: Vectors must have the same size (x: %d, y: %d)\n", x->size, y->size);
        return 0.0;
    }
    double dot = 0;
    for(int i = 0; i < x->size; i++){
        dot += x->data[i] * y->data[i];
    }
    return dot;
}