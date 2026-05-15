# Week 5 Library
add_library(week5_lib
    jit_kernel.cpp
)

# Enable SME/SVE features
if(NOT MSVC)
    target_compile_options(week5_lib PRIVATE -march=armv9-a+sme)
endif()

# Test executable for Week 5
add_executable(test_week5 ../../tests/test_week5.cpp)
target_link_libraries(test_week5 PRIVATE week5_lib week3_lib Catch2::Catch2WithMain)

# Benchmark executable for Week 5
add_executable(bench_week5 ../../apps/main_week5.cpp)
target_link_libraries(bench_week5 PRIVATE week5_lib week3_lib)

# Register the test
add_test(NAME Week5Tests COMMAND test_week5)
