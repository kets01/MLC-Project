# Week 6 Library
add_library(week6_lib
    gemm.cpp
)

# Enable SME/SVE features
if(NOT MSVC)
    target_compile_options(week6_lib PRIVATE -march=armv9-a+sme-f64f64)
endif()

# Test executable for Week 6
add_executable(test_week6 ../../tests/test_week6.cpp)
target_link_libraries(test_week6 PRIVATE week6_lib week3_lib Catch2::Catch2WithMain)

# Benchmark executable for Week 6
add_executable(bench_week6 ../../apps/main_week6.cpp)
target_link_libraries(bench_week6 PRIVATE week6_lib week3_lib)

# Register the test
add_test(NAME Week6Tests COMMAND test_week6)

