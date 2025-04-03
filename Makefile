# Define directories
EXBLAS_DIR = external/exblas-master
BUILD_DIR = $(EXBLAS_DIR)/build
EXECUTABLE_SERIAL = power_method_serial

CFLAGS = -Wall -Wextra -lm

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
	gcc -o test_serial tests/tests_serial/test_power_method_serial.c src/serial/power_method_serial.c include/vector.c $(CFLAGS) -lcunit

