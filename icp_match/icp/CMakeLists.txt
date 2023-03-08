cmake_minimum_required(VERSION 3.18.0)
project(icp VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(target_name icp)

file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h *.hpp *.cu)
add_library(${target_name} ${sources})
target_include_directories(${target_name} PRIVATE ../3rdParty/ .)