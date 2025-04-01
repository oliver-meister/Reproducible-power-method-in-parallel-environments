# CMake generated Testfile for 
# Source directory: /mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1
# Build directory: /mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/src/cpu/blas1
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TestSumNaiveNumbers "test.exsum" "24")
set_tests_properties(TestSumNaiveNumbers PROPERTIES  PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!" _BACKTRACE_TRIPLES "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;21;add_test;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;0;")
add_test(TestSumStdDynRange "test.exsum" "24" "2" "0" "n")
set_tests_properties(TestSumStdDynRange PROPERTIES  PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!" _BACKTRACE_TRIPLES "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;23;add_test;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;0;")
add_test(TestSumLargeDynRange "test.exsum" "24" "50" "0" "n")
set_tests_properties(TestSumLargeDynRange PROPERTIES  PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!" _BACKTRACE_TRIPLES "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;25;add_test;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;0;")
add_test(TestSumIllConditioned "test.exsum" "24" "1e+50" "0" "i")
set_tests_properties(TestSumIllConditioned PROPERTIES  PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!" _BACKTRACE_TRIPLES "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;27;add_test;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu/blas1/CMakeLists.txt;0;")
