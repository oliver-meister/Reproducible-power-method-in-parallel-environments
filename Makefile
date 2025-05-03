# Define directories
EXBLAS_DIR = external/exblas-master
BUILD_DIR = $(EXBLAS_DIR)/build
EXECUTABLE_SERIAL = power_method_serial

# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -lm -fopenmp -lcunit
CUDA_FLAGS = -Iinclude -Iexternal -L/usr/local/cuda/lib64 -lcudart

# Use OpenMP variant: USE_OFF for offload, USE_OMP for regular OpenMP
OMPFLAG = USE_OFF

# Default target
all: test_all

# Clean the build directory and generated files
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE_SERIAL) test_serial test_omp_sparse test_omp_dense test_omp_common test_cuda

# Serial test
test_serial:
	$(CC) -o test_serial tests/tests_serial/test_power_method_serial.c \
	src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c \
	include/vector.c include/matrix.c external/mmio.c src/common.c \
	src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -DUSE_SERIAL

# OpenMP sparse test
test_omp_sparse:
	$(CC) -o test_omp_sparse tests/tests_openMP/sparse_test_power_method_openMP.c \
	src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c \
	include/vector.c include/matrix.c external/mmio.c src/common.c \
	src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

# OpenMP dense test
test_omp_dense:
	$(CC) -o test_omp_dense tests/tests_openMP/dense_test_power_method_openMP.c \
	src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c \
	include/vector.c include/matrix.c external/mmio.c src/common.c \
	src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

# OpenMP common test
test_omp_common:
	$(CC) -o test_omp_common tests/tests_openMP/common_test_power_method_openMP.c \
	src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c \
	include/vector.c include/matrix.c external/mmio.c src/common.c \
	src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

# CUDA test
test_cuda:
	$(NVCC) -o test_cuda tests/tests_CUDA/test_power_method_cuda.c \
	src/CUDA/cuda_fun.c src/CUDA/cuda_kernels.cu \
	include/vector.c include/matrix.c \
	external/mmio.c \
	$(CUDA_FLAGS) -lcunit


	


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
