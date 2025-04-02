#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "matrix.h"
#include "vector.h"

// Function to run the power method
void power_method(const Matrix*);
void convergence();
void matvec_mult();
void normalize_vector();
void approximate_eigenvalue();
#endif 
