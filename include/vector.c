#include "vector.h"

double dot_product(Vector* V){
    double dot = 0;
    for(int i = 0; i < V->size; i++){
        dot += V->data[i] * V->data[i];
    }
    return dot;
}