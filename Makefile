# === Compilers ===
CC = gcc
NVCC = nvcc
OFFLOAD_CC = nvc

# === Flags ===
CFLAGS = -Wall -Wextra -O2 -fopenmp -I/usr/include -L/usr/lib/x86_64-linux-gnu -lm
OFFLOAD_FLAGS = -mp=gpu -O2 -Minfo=accel -lcudart
CUDA_LIBS = -L/usr/lib/x86_64-linux-gnu -lcudart

# CUDA architecture flag optimized for NVIDIA RTX 4060 (Ada / sm_89)
CUDA_ARCH_FLAGS = -gencode=arch=compute_89,code=sm_89
CUDA_WARN_FLAGS = -Wno-deprecated-gpu-targets
CUDA_FLAGS = $(CUDA_ARCH_FLAGS) $(CUDA_WARN_FLAGS) $(CUDA_LIBS) -Xcompiler="-Wall -Wextra -fopenmp"

CUNIT = -lcunit

# === Object Files ===
GENERAL_OBJ = include/vector.o include/matrix.o external/mmio.o

COMMON_OBJ_SERIAL = src/common_serial.o
COMMON_OBJS_OPENMP = src/common_openmp.o
COMMON_OBJS_OFFLOAD = src/common_offload.o
COMMON_OBJS_CUDA = src/common_cuda.o
COMMON_OBJS_EXBLAS = src/common_exblas.o

SERIAL_OBJS = src/serial/serial_fun.o
OMP_OBJS = src/openMP/omp_fun.o
OFFLOAD_OBJS = src/OMP_Offload/off_fun.o
CUDA_OBJS = src/CUDA/cuda_fun.o src/CUDA/cuda_kernels.o
CUDA_EXBLAS = src/CUDA_ExBLAS/cuda_exblas_fun.o src/CUDA_ExBLAS/cuda_exblas_kernels.o

SPARSE_OBJS = src/sparse_power_method.o
DENSE_OBJS = src/dense_power_method.o

# === Test Files ===
TEST_SERIAL = tests/tests_serial/test_power_method_serial.c
TEST_OMP = tests/tests_openMP/test_power_method_openMP.c
TEST_OFF = tests/tests_offload/test_power_method_offload.c
TEST_CUDA = tests/tests_CUDA/test_power_method_cuda.c
TEST_EXBLAS = tests/tests_CUDA_EXBLAS/test_power_method_cuda_exblas.c

# === Build Rules ===

include/vector.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS)

include/matrix.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS)

external/mmio.o: external/mmio.c external/mmio.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/common_serial.o: src/common.c src/common.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL

src/common_openmp.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP

src/common_offload.o: src/common.c src/common.h
	$(OFFLOAD_CC) -c $< -o $@ $(OFFLOAD_FLAGS) -DUSE_OFF

src/common_cuda.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA

src/common_exblas.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS

src/serial/serial_fun.o: src/serial/serial_fun.c src/serial/serial_fun.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/openMP/omp_fun.o: src/openMP/omp_fun.c src/openMP/omp_fun.h
	$(CC) -c $< -o $@ $(CFLAGS)

src/OMP_Offload/off_fun.o: src/OMP_Offload/off_fun.c src/OMP_Offload/off_fun.h
	$(OFFLOAD_CC) -c $< -o $@ $(OFFLOAD_FLAGS)

src/CUDA/cuda_fun.o: src/CUDA/cuda_fun.c src/CUDA/cuda_fun.h
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/CUDA/cuda_kernels.o: src/CUDA/cuda_kernels.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/CUDA_ExBLAS/cuda_exblas_fun.o: src/CUDA_ExBLAS/cuda_exblas_fun.c src/CUDA_ExBLAS/cuda_exblas_fun.h
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/CUDA_ExBLAS/cuda_exblas_kernels.o: src/CUDA_ExBLAS/cuda_exblas_kernels.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/sparse_power_method.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS)

src/dense_power_method.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS)

# === Final Executables ===

test_serial: $(DENSE_OBJS) $(SPARSE_OBJS) $(COMMON_OBJ_SERIAL) $(GENERAL_OBJ) $(SERIAL_OBJS)
	$(CC) -o test_serial $(TEST_SERIAL) $^ $(CFLAGS) $(CUNIT)

test_openmp: $(DENSE_OBJS) $(SPARSE_OBJS) $(COMMON_OBJS_OPENMP) $(GENERAL_OBJ) $(OMP_OBJS)
	$(CC) -o test_openmp $(TEST_OMP) $^ $(CFLAGS) $(CUNIT)

test_offload: $(DENSE_OBJS) $(SPARSE_OBJS) $(COMMON_OBJS_OFFLOAD) $(GENERAL_OBJ) $(OFFLOAD_OBJS)
	$(OFFLOAD_CC) -o test_offload $(TEST_OFF) $^ $(OFFLOAD_FLAGS) $(CUNIT)

test_cuda: $(DENSE_OBJS) $(SPARSE_OBJS) $(COMMON_OBJS_CUDA) $(GENERAL_OBJ) $(CUDA_OBJS)
	$(NVCC) -o test_cuda $(TEST_CUDA) $^ $(CUDA_FLAGS) $(CUNIT)

test_cuda_exblas: $(DENSE_OBJS) $(SPARSE_OBJS) $(COMMON_OBJS_EXBLAS) $(GENERAL_OBJ) $(CUDA_EXBLAS) $(CUDA_OBJS)
	$(NVCC) -o test_cuda_exblas $(TEST_EXBLAS) $^ $(CUDA_FLAGS) $(CUNIT)

# === Cleanup ===

clean:
	find . -name '*.o' -delete
	rm -f test_serial test_openmp test_offload test_cuda test_cuda_exblas

# === Test Offload Run Helper ===

run_test_offload: test_offload
	export OMP_TARGET_OFFLOAD=MANDATORY && \
	export LIBOMPTARGET_INFO=30 && \
	./test_offload
