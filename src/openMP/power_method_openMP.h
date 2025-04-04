#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function to run the power method
double openMP_power_method(Matrix*);
bool openMP_convergence(double, double, double);
void openMP_matvec_mult(Matrix*, Vector*);
void openMP_normalize_vector(Vector*);
double openMP_approximate_eigenvalue(Matrix*, Vector*);
double openMP_dot_product(Vector*, Vector*);

#endif 
