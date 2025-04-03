#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"

// Function to run the power method
void serial_power_method(Matrix*);
void serial_convergence();
void serial_matvec_mult(Matrix*, Vector*);
void serial_normalize_vector();
void serial_approximate_eigenvalue();

#endif 
