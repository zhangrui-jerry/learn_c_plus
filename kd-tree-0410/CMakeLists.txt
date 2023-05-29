cmake_minimum_required(VERSION 3.20)

project(InterfaceTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SAMPLE_COMMON_DIR})

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

add_executable(kd_test kd_tree_test.cpp kd_tree.h kernel.h kernel.cu)
# use the following line to enable host device mix compilation
set_target_properties(kd_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kd_test PUBLIC OpenMP::OpenMP_CXX)
endif()