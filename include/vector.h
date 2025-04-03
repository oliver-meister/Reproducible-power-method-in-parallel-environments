#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
    double* data;
    int size;
} Vector;

double dot_product(Vector*);

#endif 

