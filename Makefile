# Compiler and flags
CC = gcc
NVCC = nvcc

CFLAGS = -Wall -Wextra -lm -fopenmp
TEST_FLAGS = $(CFLAGS) -lcunit
CUDA_FLAGS =  -L/usr/local/cuda/lib64 -lcudart

# Use OpenMP variant: USE_OFF for offload, USE_OMP for regular OpenMP
OMPFLAG = USE_OFF

# Default target
all: test_all

# Clean the build directory and generated files
clean:
	rm -rf test_serial test_omp_sparse test_omp_dense test_omp_common test_cuda

mmio.o: external/mmio.c external/mmio.h
	$(CC) -c external/mmio.c -o mmio.o $(CFLAGS)

vector.o: include/vector.c include/vector.h
	$(CC) -c include/vector.c -o vector.o

matrix.o: include/matrix.c include/matrix.h mmio.o
	$(CC) -c include/matrix.c -o matrix.o $(CFLAGS)

serial_fun.o: src/serial/serial_fun.c src/serial/serial_fun.h vector.o matrix.o
	$(CC) -c src/serial/serial_fun.c -o serial_fun.o $(CFLAGS)

omp_fun.o: src/OMP/omp_fun.c src/OMP/omp_fun.h vector.o matrix.o
	$(CC) -c src/OMP/omp_fun.c -o omp_fun.o $(CFLAGS)

off_fun.o: src/OMP_Offload/off_fun.c src/OMP_Offload/off_fun.h vector.o matrix.o
	$(CC) -c src/OMP_Offload/off_fun.c -o off_fun.o $(CFLAGS)

cuda_fun.o: src/CUDA/cuda_fun.c src/CUDA/cuda_fun.h vector.o matrix.o
	$(NVCC) -c src/CUDA/cuda_fun.c -o cuda_fun.o $(CUDA_FLAGS)

power_method_sparse.o: src/power_method_sparse.c src/power_method_sparse.h 

# Serial test
test_serial: mmio.o
	

# OpenMP sparse test
test_omp_sparse: mmio.o


# OpenMP dense test
test_omp_dense: mmio.o
	

# OpenMP common test
test_omp_common: mmio.o


# CUDA test
test_cuda: mmio.o



# Run all tests

.PHONY: test_all

test_all: test_serial test_omp_sparse test_omp_dense test_omp_common
	./test_serial
	./test_omp_sparse 2
	./test_omp_dense 2
	./test_omp_common 2
	./test_cuda

# Intel MKL environment
mkl:
	source /opt/intel/oneapi/setvars.sh
