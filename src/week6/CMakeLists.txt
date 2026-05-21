cmake_minimum_required(VERSION 3.10)
project(Week6_JIT)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)

# Compilation flags for ARMv9 SME/SVE support
if(NOT MSVC)
    add_compile_options(-march=armv9-a+sme+sve)
endif()

# Library for Week 6
add_library(week6_lib Unary.cpp Gemm.cpp)
# Link with Week 5 to access JitEngine
target_link_libraries(week6_lib PRIVATE week5_lib)

# Benchmark App
add_executable(bench_week6 main_week6.cpp)
target_link_libraries(bench_week6 PRIVATE week6_lib)

# Tests
add_executable(test_week6 test_week6.cpp)
target_link_libraries(test_week6 PRIVATE week6_lib Catch2::Catch2WithMain)