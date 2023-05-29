cmake_minimum_required(VERSION 3.18.0)
set(CMAKE_CXX_STANDARD 17)
set(target_name icp_demo)
project(${target_name} VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()
add_subdirectory(./point_search)
add_subdirectory(./test)

file(GLOB sources CONFIGURE_DEPENDS ./demo/*.cpp ./demo/*.h ./demo/*.hpp)
add_executable(${target_name} ${sources})
target_include_directories(${target_name} PRIVATE ./icp ./ ./3rdParty/)
target_link_libraries(${target_name} point_search)
target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

