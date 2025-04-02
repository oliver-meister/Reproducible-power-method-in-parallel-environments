# Define directories
EXBLAS_DIR = external/exblas-master
BUILD_DIR = $(EXBLAS_DIR)/build
EXECUTABLE = power_method_serial

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
	rm -f $(EXECUTABLE)

run: 
	gcc Serial/$(EXECUTABLE).c -o $(EXECUTABLE)
	./$(EXECUTABLE)

