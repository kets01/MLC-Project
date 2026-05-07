# Week 3 Library: Unary Ops + GEMM
add_library(week3_lib
    # Exercise 3.1 — Unary Ops
    unary.cpp
    identity_16_16_asm.S
    zero_16_16_asm.S
    relu_16_16_asm.S

    # Exercise 3.2 — GEMM
    gemm_sme.cpp
    gemm_sme.S
)

# Enable SME/SVE features
if(NOT MSVC)
    target_compile_options(week3_lib PRIVATE -march=armv9-a+sme)
endif()

# Test executable for Week 3
add_executable(test_week3 ../../tests/test_week3.cpp)
target_link_libraries(test_week3 PRIVATE week3_lib Catch2::Catch2WithMain)

# Benchmark executable for Week 3
add_executable(bench_week3 ../../apps/main_week3.cpp)
target_link_libraries(bench_week3 PRIVATE week3_lib)

# Register the test
add_test(NAME Week3Tests COMMAND test_week3)
