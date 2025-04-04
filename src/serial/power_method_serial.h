#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include "../../include/matrix.h"
#include "../../include/vector.h"
#include <stdbool.h>

// Function to run the power method
double serial_dense_power_method(denseMatrix*);
void serial_dense_matvec_mult(denseMatrix*, Vector*);
double serial_dense_approximate_eigenvalue(denseMatrix*, Vector*);

double serial_sparse_power_method(sparseMatrix*);
void serial_sparse_matvec_mult(sparseMatrix*, Vector*);
double serial_sparse_approximate_eigenvalue(sparseMatrix*, Vector*);

void serial_normalize_vector(Vector*);
bool serial_convergence(double, double, double);
#endif 
