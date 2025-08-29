# === Compilers ===
CC = gcc
NVCC = nvcc
OFFLOAD_CC = nvc

DEBUG ?= 1

ifeq ($(DEBUG),1)
  CDEBUG = -g -O0 -fno-omit-frame-pointer
  CUDADEBUG = -G -g -O0 -lineinfo -Xptxas -O0 -v
  OFFLOADDEBUG = -g -O0 -gpu=lineinfo
else
  CDEBUG = -O2
  CUDADEBUG =
  OFFLOADDEBUG = -O2 -Minfo=accel
endif


# === Flags ===
CFLAGS = -Wall -Wextra -fopenmp -I/usr/include -L/usr/lib/x86_64-linux-gnu -lm -I/home/o/olla5642/CUnit-2.1-3/install/include -L/home/o/olla5642/CUnit-2.1-3/install/lib $(CDEBUG)
OFFLOAD_FLAGS = -mp=gpu -lcudart $(OFFLOADDEBUG)
CUDA_LIBS = -L/usr/lib/x86_64-linux-gnu -lcudart

# CUDA architecture flag optimized for NVIDIA RTX 4060 (Ada / sm_89)
CUDA_ARCH_FLAGS = -gencode=arch=compute_70,code=sm_70
CUDA_WARN_FLAGS = -Wno-deprecated-gpu-targets
CUDA_FLAGS = $(CUDA_ARCH_FLAGS) $(CUDA_WARN_FLAGS) $(CUDA_LIBS) -Xcompiler="-Wall -Wextra -fopenmp" $(CUDADEBUG)

CUNIT = -I/home/o/olla5642/CUnit-2.1-3/install/include -L/home/o/olla5642/CUnit-2.1-3/install/lib -lcunit

# === Object Files ===
IMPORT_OBJ_SERIAL = include/vector_serial.o include/matrix_serial.o external/mmio.o
IMPORT_OBJ_OMP = include/vector_omp.o include/matrix_omp.o external/mmio.o
IMPORT_OBJ_OFF = include/vector_off.o include/matrix_off.o external/mmio.o
IMPORT_OBJ_CUDA = include/vector_cuda.o include/matrix_cuda.o external/mmio.o
IMPORT_OBJ_EXBLAS = include/vector_exblas.o include/matrix_exblas.o external/mmio.o

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

SPARSE_OBJS_SERIAL = src/sparse_power_method_serial.o
SPARSE_OBJS_OMP = src/sparse_power_method_omp.o
SPARSE_OBJS_OFF = src/sparse_power_method_off.o
SPARSE_OBJS_CUDA = src/sparse_power_method_cuda.o
SPARSE_OBJS_EXBLAS = src/sparse_power_method_exblas.o

DENSE_OBJS = src/dense_power_method.o

# === Test Files ===
TEST_SERIAL = tests/tests_serial/test_power_method_serial.c
TEST_OMP = tests/tests_openMP/test_power_method_openMP.c
TEST_OFF = tests/tests_offload/test_power_method_offload.c
TEST_CUDA = tests/tests_CUDA/test_power_method_cuda.c
TEST_EXBLAS = tests/tests_CUDA_EXBLAS/test_power_method_cuda_exblas.c

# === Build Rules ===

include/vector_serial.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL
include/vector_omp.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP
include/vector_off.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
include/vector_cuda.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
include/vector_exblas.o: include/vector.c include/vector.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS

include/matrix_serial.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL
include/matrix_omp.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP
include/matrix_off.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
include/matrix_cuda.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
include/matrix_exblas.o: include/matrix.c include/matrix.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS


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

src/sparse_power_method_serial.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL
src/sparse_power_method_omp.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP
src/sparse_power_method_off.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
src/sparse_power_method_cuda.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
src/sparse_power_method_exblas.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS

src/dense_power_method.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS)

# === Final Executables ===
test_serial: $(DENSE_OBJS) $(SPARSE_OBJS_SERIAL) $(COMMON_OBJ_SERIAL) $(IMPORT_OBJ_SERIAL) $(SERIAL_OBJS)
	$(CC) -o test_serial $(TEST_SERIAL) $^ $(CFLAGS) $(CUNIT)

test_openmp: $(DENSE_OBJS) $(SPARSE_OBJS_OMP) $(COMMON_OBJS_OPENMP) $(IMPORT_OBJ_OMP) $(OMP_OBJS)
	$(CC) -o test_openmp $(TEST_OMP) $^ $(CFLAGS) $(CUNIT)

test_offload: $(DENSE_OBJS) $(SPARSE_OBJS_OFF) $(COMMON_OBJS_OFFLOAD) $(IMPORT_OBJ_OFF) $(OFFLOAD_OBJS)
	$(OFFLOAD_CC) -o test_offload $(TEST_OFF) $^ $(OFFLOAD_FLAGS) $(CUNIT)

test_cuda: $(DENSE_OBJS) $(SPARSE_OBJS_CUDA) $(COMMON_OBJS_CUDA) $(IMPORT_OBJ_CUDA) $(CUDA_OBJS)
	$(NVCC) -o test_cuda $(TEST_CUDA) $^ $(CUDA_FLAGS) $(CUNIT)

test_cuda_exblas: $(DENSE_OBJS) $(SPARSE_OBJS_EXBLAS) $(COMMON_OBJS_EXBLAS) $(IMPORT_OBJ_EXBLAS) $(CUDA_EXBLAS) $(CUDA_OBJS)
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
