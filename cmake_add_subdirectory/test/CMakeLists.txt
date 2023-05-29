cmake_minimum_required(VERSION 3.18.0)

set(target_name t)

if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
    add_compile_options(/W4)
    add_compile_options(/wd4819)
endif()

add_executable(${target_name} t.cpp)
target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS} ../sub_dir)
target_link_libraries (${target_name} ${OpenCV_LIBS} testlib)