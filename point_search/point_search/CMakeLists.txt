cmake_minimum_required(VERSION 3.18.0)
set(CMAKE_CXX_STANDARD 17)
set(target_name point_search)
project(${target_name} VERSION 0.1.0 LANGUAGES CXX CUDA)

file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h *.hpp *.cu)
add_library(${target_name} STATIC ${sources})
target_include_directories(${target_name} PRIVATE ../3rdParty/ .)
target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)