set(target_name icp)

file(GLOB sources CONFIGURE_DEPENDS ./*.cpp ./*.h ./*.hpp)
add_library(${target_name} sources)
target_include_directories(${target_name} PRIVATE ../3rdParty/ .)