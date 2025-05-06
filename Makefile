CC = gcc
NVCC = nvcc

INCLUDES = -Iinclude -Isrc -Iexternal
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
CFLAGS = -Wall -Wextra -O2 -fopenmp -lm
CUDA_FLAGS = $(INCLUDES) $(CUDA_LIBS) -Xcompiler="-Wall -Wextra -fopenmp"
CUNIT = -lcunit

COMMON_OBJS = include/vector.o include/matrix.o external/mmio.o src/common.o
SERIAL_OBJS = src/serial/serial_fun.o
OMP_OBJS = src/openMP/omp_fun.o
OFFLOAD_OBJS = src/OMP_Offload/off_fun.o
CUDA_OBJS = src/CUDA/cuda_fun.o src/CUDA/cuda_kernels.o
SPARSE_OBJS = src/sparse_power_method.o
DENSE_OBJS = src/dense_power_method.o

TEST_SERIAL = tests/tests_serial/test_power_method_serial.c
TEST_OMP = tests/tests_openMP/test_power_method_openMP.c
TEST_CUDA = tests/tests_CUDA/test_power_method_cuda.c

all: test_serial test_openmp test_offload test_cuda

include/vector.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS)

include/matrix.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS)

external/mmio.o: external/mmio.c external/mmio.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/common.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/serial/serial_fun.o: src/serial/serial_fun.c src/serial/serial_fun.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/openMP/omp_fun.o: src/openMP/omp_fun.c src/openMP/omp_fun.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/OMP_Offload/off_fun.o: src/OMP_Offload/off_fun.c src/OMP_Offload/off_fun.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/CUDA/cuda_fun.o: src/CUDA/cuda_fun.c src/CUDA/cuda_fun.h
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/CUDA/cuda_kernels.o: src/CUDA/cuda_kernels.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/sparse_power_method.o: src/sparse_power_method.c src/sparse_power_method.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/dense_power_method.o: src/dense_power_method.c src/dense_power_method.h
	$(CC) -c $< -o $@ $(CFLAGS)


test_serial: $(COMMON_OBJS) $(SERIAL_OBJS) $(SPARSE_OBJS) $(DENSE_OBJS)
	$(CC) -o test_serial $(TEST_SERIAL) $^ $(CFLAGS) $(CUNIT) -DUSE_SERIAL

test_openmp: $(COMMON_OBJS) $(OMP_OBJS) $(SPARSE_OBJS) $(DENSE_OBJS) $(SERIAL_OBJS)
	$(CC) -o test_openmp $(TEST_OMP) $^ $(CFLAGS) $(CUNIT) -DUSE_OMP

test_offload: $(COMMON_OBJS) $(OFFLOAD_OBJS) $(OMP_OBJS) $(SPARSE_OBJS) $(DENSE_OBJS) $(SERIAL_OBJS)
	$(CC) -o test_offload $(TEST_OMP) $^ $(CFLAGS) $(CUNIT) -DUSE_OFF

test_cuda: $(COMMON_OBJS) $(CUDA_OBJS) $(SPARSE_OBJS) $(DENSE_OBJS) $(SERIAL_OBJS)
	$(NVCC) -o test_cuda $(TEST_CUDA) $^ $(CUDA_FLAGS) $(CUNIT) -DUSE_CUDA

clean:
	rm -f *.o */*.o */*/*.o test_serial test_openmp test_offload test_cuda