cmake_minimum_required(VERSION 3.18.0)

set(target_name testlib)

if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
    add_compile_options(/W4)
    add_compile_options(/wd4819)
endif()

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.cpp *.h *.hpp)
add_library(${target_name} STATIC ${sources})
target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_include_directories(${target_name} PRIVATE ../../extra .)
# target_compile_options(${target_name} PUBLIC /bigobj)
