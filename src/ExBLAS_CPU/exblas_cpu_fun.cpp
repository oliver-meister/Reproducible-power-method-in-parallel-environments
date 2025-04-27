#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mkl_blas.h>
#include <mpi.h>
#include <hb_io.h>
#include <vector>

#include "../../external/ReproCG-master/src/reloj.h"
#include "../../external/ReproCG-master/src/ScalarVectors.h"
#include "../../external/ReproCG-master/src/SparseProduct.h"
#include "../../external/ReproCG-master/src/ToolsMPI.h"
#include "../../external/ReproCG-master/src/matrix.h"
#include "../../external/ReproCG-master/src/common.h"
#include "../../external/ReproCG-master/src/exblas/exdot.h"
//ExBLAS

double exBlasCpu_dot_product(const Vector* x, const Vector* y){

    std::vector<int64_t> h_superacc(2 * exblas::BIN_COUNT);
    exblas::cpu::exdot(x->size, x, y, )
}



