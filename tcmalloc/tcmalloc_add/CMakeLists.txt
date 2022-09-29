cmake_minimum_required(VERSION 3.18.0)
project(tcmalloc_add VERSION 0.1.0)
set(target_name tcmalloc_add)

include(CTest)
enable_testing()

file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.cpp *.h *.hpp)
add_library(${target_name} SHARED ${sources})

set_target_properties(${target_name} PROPERTIES LINK_FLAGS /INCLUDE:__tcmalloc)
target_link_directories(${target_name} PUBLIC ../extra/libtcmalloc/Release)
target_link_libraries(${target_name} libtcmalloc_minimal)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
