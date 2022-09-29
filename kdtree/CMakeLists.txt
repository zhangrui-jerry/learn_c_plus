cmake_minimum_required(VERSION 3.18.0)
project(kdtree VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

include(CTest)
enable_testing()

add_executable(kdtree main.cpp ./TimeTest/TimeTest.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
