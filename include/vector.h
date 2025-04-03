#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
    double* data;
    int size;
} Vector;

double dot_product(Vector*, Vector*);
double generate_random_double(double, double);
Vector* generate_random_vector(double);

#endif 

