#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function to run the power method
double serial_power_method(Matrix*);
bool serial_convergence(double, double, double);
void serial_matvec_mult(Matrix*, Vector*);
void serial_normalize_vector(Vector*);
double serial_approximate_eigenvalue(Matrix*, Vector*);
Vector* generate_random_vector(double);
double generate_random_double(double, double);

#endif 
