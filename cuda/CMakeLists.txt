cmake_minimum_required(VERSION 3.18.0)
set(target_name cuda_learn)
set(CMAKE_CXX_STANDARD 17)

project(${target_name} VERSION 0.1.0 LANGUAGES CXX CUDA)


add_executable(cuda_learn main.cu)
target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
