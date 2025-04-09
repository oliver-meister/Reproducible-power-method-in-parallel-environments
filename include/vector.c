#include "vector.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Calculates the dot product of two Vectors.
 * 
 * @param x First vector.
 * @param y Second vector.
 * 
 * @return The dot product of vector x and y.
 */
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

/**
 * @brief Generates a random double between [min, max]
 * 
 * @param min The lower bound of the interval.
 * @param max The upper bound of the interval.
 * 
 * @return A random double between min and max.
 */
double generate_random_double(double min, double max){
    return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

/**
 * @brief Generates a vector of random doubles.
 * 
 * @param size Number of elements in the vector.
 * 
 * @return Pointer to a heap allocated Vector.
 */
Vector* generate_random_vector(int size){
    Vector* x = malloc(sizeof(Vector));
    x->size = size;
    x->data = malloc(sizeof(double) * size);

    for(int i = 0; i < size; i++){
        x->data[i] = generate_random_double(-1, 1);
    }
    return x;
}