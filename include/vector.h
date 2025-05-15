#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
    double* data;
    int size;
} Vector;

double generate_random_double(double, double);
Vector* generate_random_vector(int size);
Vector* generate_vector(int size);
void delete_vector(Vector* x);

#endif 

