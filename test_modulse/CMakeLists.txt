cmake_minimum_required(VERSION 3.18.0)
project(MODULSE VERSION 0.1.0)

include(CTest)
enable_testing()

set(Open3D_DIR ./extra/Open3D/CMake)
find_package(Open3D REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(target_name demo)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
add_definitions(-DCV_IGNORE_DEBUG_BUILD_GUARD)
add_subdirectory(./src)
add_subdirectory(./contour)

add_executable(${target_name} main.cpp)
target_link_libraries(${target_name} ellipeselib contourlib)
target_link_libraries(${target_name} Open3D::Open3D)
target_include_directories(${target_name} PRIVATE ./src ./contour)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
