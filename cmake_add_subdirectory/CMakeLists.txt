cmake_minimum_required(VERSION 3.18.0)
project(cmake_test VERSION 0.1.0)
set(target_name cmake_test)

set(OpenCV_DIR ./extra/OpenCV)
find_package(OpenCV REQUIRED)

include(CTest)
enable_testing()

add_executable(${target_name} main.cpp)
target_include_directories(${target_name} PRIVATE ./sub_dir)
target_link_libraries (${target_name} ${OpenCV_LIBS} testlib)

add_subdirectory(./sub_dir)
add_subdirectory(./test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
