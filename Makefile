# Define directories
EXBLAS_DIR = external/exblas-master
BUILD_DIR = $(EXBLAS_DIR)/build
EXECUTABLE_SERIAL = power_method_serial

CFLAGS = -Wall -Wextra -lm -fopenmp -lcunit


# USE_OFF for omp offloading
# USE_omp for regular omp
OMPFLAG = USE_OFF

# Default target
all: build

# Create build directory and run cmake
build:
	@echo "Creating build directory and running cmake..."
	mkdir -p $(BUILD_DIR)  # Create build directory
	cd $(BUILD_DIR) && cmake ..  # Run cmake in the build directory
	make make
# Run make inside the build directory
make:
	cd $(BUILD_DIR) && make  # Build the project using make

# Clean the build directory
clean:
	rm -rf $(BUILD_DIR)  # Clean the build directory
	rm -f $(EXECUTABLE_SERIAL)
	rm -f test*
# Creates and run executable
run_serial: build_serial 
	./$(EXECUTABLE_SERIAL)


build_serial:
	gcc src/Serial/$(EXECUTABLE_SERIAL).c -o $(EXECUTABLE_SERIAL)

test_serial:
	gcc -o test_serial tests/tests_serial/test_power_method_serial.c src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c include/vector.c include/matrix.c external/mmio.c src/common.c src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -DUSE_SERIAL

test_omp_sparse:
	gcc -o test_omp_sparse tests/tests_openMP/sparse_test_power_method_openMP.c src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c include/vector.c include/matrix.c external/mmio.c src/common.c src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

test_omp_dense:
	gcc -o test_omp_dense tests/tests_openMP/dense_test_power_method_openMP.c src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c include/vector.c include/matrix.c external/mmio.c src/common.c src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

test_omp_common:
	gcc -o test_omp_common tests/tests_openMP/common_test_power_method_openMP.c src/serial/serial_fun.c src/openMP/omp_fun.c src/OMP_Offload/off_fun.c include/vector.c include/matrix.c external/mmio.c src/common.c src/dense_power_method.c src/sparse_power_method.c $(CFLAGS) -D$(OMPFLAG)

test_all: test_serial test_omp_sparse test_omp_dense test_omp_common
	./test_serial
	./test_omp_sparse 2
	./test_omp_dense 2
	./test_omp_common 2

mkl:
	source /opt/intel/oneapi/setvars.sh
