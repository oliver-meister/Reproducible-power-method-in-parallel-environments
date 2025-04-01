# Install script for directory: /mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/src/cpu

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/lib/libexblas.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/lib" TYPE STATIC_LIBRARY FILES "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/src/cpu/libexblas.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/include/blas1.hpp;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/include/blas2.hpp;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/include/blas3.hpp;/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/include/common.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/include" TYPE FILE FILES
    "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/include/blas1.hpp"
    "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/include/blas2.hpp"
    "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/include/blas3.hpp"
    "/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/include/common.hpp"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/mnt/c/Users/olive/Bachelor-thesis/Reproducible-power-method-in-parallel-environments/external/exblas-master/build/src/cpu/blas1/cmake_install.cmake")

endif()

