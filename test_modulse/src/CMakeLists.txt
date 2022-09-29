cmake_minimum_required(VERSION 3.18.0)

set(target_name ellipeselib)

set(OpenCV_DIR C:/OpenCV)
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.cpp *.h *.hpp)
add_library(${target_name} ${sources})
target_link_libraries(${target_name} ${OpenCV_LIBS})
target_compile_options(${target_name} PUBLIC /bigobj)
target_include_directories(${target_name} PRIVATE "C:\\OpenCV\\include" ../extra ../ .)