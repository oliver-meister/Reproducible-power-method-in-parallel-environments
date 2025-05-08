CC = gcc
NVCC = nvcc

CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
CFLAGS = -Wall -Wextra -O2 -fopenmp -lm
CUDA_FLAGS = $(CUDA_LIBS) -Xcompiler="-Wall -Wextra -fopenmp"
CUNIT = -lcunit

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
CUDA_EXBLAS = src/CUDA_ExBLAS/cuda_exblas_fun.o src/CUDA_ExBLAS/ExDOT.FPE.EX.4.o


SPARSE_OBJS_SERIAL = src/sparse_power_method_serial.o
SPARSE_OBJS_OPENMP = src/sparse_power_method_openmp.o
SPARSE_OBJS_OFFLOAD = src/sparse_power_method_offload.o
SPARSE_OBJS_CUDA = src/sparse_power_method_cuda.o
SPARSE_OBJS_EXBLAS = src/sparse_power_method_exdot.o

DENSE_OBJS_SERIAL = src/dense_power_method_serial.o
DENSE_OBJS_OPENMP = src/dense_power_method_openmp.o
DENSE_OBJS_OFFLOAD = src/dense_power_method_offload.o
DENSE_OBJS_CUDA = src/dense_power_method_cuda.o
DENSE_OBJS_EXBLAS = src/dense_power_method_exblas.o

TEST_SERIAL = tests/tests_serial/test_power_method_serial.c
TEST_OMP = tests/tests_openMP/test_power_method_openMP.c
TEST_OFF = tests/tests_offload/test_power_method_offload.c
TEST_CUDA = tests/tests_CUDA/test_power_method_cuda.c




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
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
src/common_cuda.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
src/common_exblas.o: src/common.c src/common.h
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS



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


src/CUDA_ExBLAS/cuda_exblas_fun.o: src/CUDA_ExBLAS/cuda_exblas_fun.c src/CUDA_ExBLAS/cuda_exblas_fun.h
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

src/CUDA_ExBLAS/ExDOT.FPE.EX.4.o: src/CUDA_ExBLAS/ExDOT.FPE.EX.4.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)



src/sparse_power_method_serial.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL
src/sparse_power_method_openmp.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP
src/sparse_power_method_offload.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
src/sparse_power_method_cuda.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
src/sparse_power_method_exdot.o: src/sparse_power_method.c src/sparse_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_EXBLAS


src/dense_power_method_serial.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_SERIAL
src/dense_power_method_openmp.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OMP
src/dense_power_method_offload.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_OFF
src/dense_power_method_cuda.o: src/dense_power_method.c src/dense_power_method.h 
	$(CC) -c $< -o $@ $(CFLAGS) -DUSE_CUDA
src/dense_power_method_exblas.o: src/dense_power_method.c src/dense_power_method.h 




test_serial:  $(DENSE_OBJS_SERIAL) $(SPARSE_OBJS_SERIAL) $(COMMON_OBJ_SERIAL) $(GENERAL_OBJ) $(SERIAL_OBJS)
	$(CC) -o test_serial $(TEST_SERIAL) $^ $(CFLAGS) $(CUNIT) 

test_openmp: $(DENSE_OBJS_OPENMP) $(SPARSE_OBJS_OPENMP) $(COMMON_OBJS_OPENMP) $(GENERAL_OBJ) $(OMP_OBJS)
	$(CC) -o test_openmp $(TEST_OMP) $^ $(CFLAGS) $(CUNIT) 

test_offload: $(DENSE_OBJS_OFFLOAD) $(SPARSE_OBJS_OFFLOAD) $(COMMON_OBJS_OFFLOAD) $(GENERAL_OBJ) $(OFFLOAD_OBJS)
	$(CC) -o test_offload $(TEST_OFF) $^ $(CFLAGS) $(CUNIT) 

test_cuda: $(DENSE_OBJS_CUDA) $(SPARSE_OBJS_CUDA) $(COMMON_OBJS_CUDA) $(GENERAL_OBJ) $(CUDA_OBJS)
	$(NVCC) -o test_cuda $(TEST_CUDA) $^ $(CUDA_FLAGS) $(CUNIT) 

test_cuda_exblas: $(DENSE_OBJS_EXBLAS) $(SPARSE_OBJS_EXBLAS) $(COMMON_OBJS_EXBLAS) $(GENERAL_OBJ) $(CUDA_EXBLAS)
	$(NVCC) -o test_cuda_exblas $(TEST_CUDA) $^ $(CUDA_FLAGS) $(CUNIT) 

clean:
	find . -name '*.o' -delete
	rm -f test_serial test_openmp test_offload test_cuda test_cuda_exblas
