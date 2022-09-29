cmake_minimum_required(VERSION 3.0.0)
project(tcmalloc VERSION 0.1.0)
set(target_name tcmalloc)

include(CTest)
enable_testing()

add_subdirectory(tcmalloc_add)

add_executable(${target_name} main.cpp)
target_include_directories(${target_name} PRIVATE ./tcmalloc_add)
target_link_libraries(${target_name} tcmalloc_add)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
