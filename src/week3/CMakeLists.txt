# Create Library for Week 2
add_library(week3_lib 
    gemm_sme.cpp 
    gemm_sme.S)

# Target-specific SME flags
target_compile_options(week3_lib PRIVATE -march=armv9-a+sme)

# Benchmark App for Week 3
add_executable(bench_week3 ../../apps/main_week3.cpp)
target_link_libraries(bench_week3 PRIVATE week3_lib)

# Create the test executable
add_executable(test_week3 ../../tests/test_week3.cpp)
target_link_libraries(test_week3 PRIVATE week3_lib Catch2::Catch2WithMain)

# Register test
add_test(NAME Week3Tests COMMAND test_week3)


