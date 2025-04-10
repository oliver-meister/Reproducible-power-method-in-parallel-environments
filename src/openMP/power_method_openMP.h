#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function for dense power method
double openMP_dense_power_method(denseMatrix*);
void openMP_dense_matvec_mult(denseMatrix*, Vector*);
double openMP_dense_approximate_eigenvalue(denseMatrix*, Vector*);
//Function for sparse power method

// Common functions
bool openMP_convergence(double, double, double);
void openMP_normalize_vector(Vector*);
double openMP_dot_product(Vector*, Vector*);
double openMP_dot_product2(Vector*, Vector*);


#endif 
