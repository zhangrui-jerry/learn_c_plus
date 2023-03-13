cmake_minimum_required(VERSION 3.18.0)

set(target_name test_search)

add_executable(${target_name} test_point_search.cpp)
target_include_directories(${target_name} PRIVATE ../point_search/ ../3rdParty/)
target_link_libraries (${target_name} point_search)