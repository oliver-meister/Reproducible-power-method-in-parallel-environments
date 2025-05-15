#ifndef DENSE_POWER_METHOD_H
#define DENSE_POWER_METHOD_H
#include "../include/matrix.h"
#include "../include/vector.h"
#include <stdbool.h>

double dense_power_method(const denseMatrix* );
double dense_approximate_eigenvalue(const denseMatrix*, Vector*, Vector*);
#endif 
