cmake_minimum_required(VERSION 3.10)

project(prefix-scan)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA 10.0 REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52")

add_executable (scan
    src/prefixScanGpu.cu
    src/prefixScanCpu.cpp
    src/utils.cpp 
    src/test.cpp
    src/main.cpp)

