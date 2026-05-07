# Create Library for Week 3
# Includes the C++ reference (unary.cpp) and the three assembly kernels
add_library(week3_lib 
    unary.cpp
    identity_16_16_asm.S 
    zero_16_16_asm.S
    relu_16_16_asm.S
)

# Enable SVE/SME hardware features for the compiler/assembler
if(NOT MSVC)
    target_compile_options(week3_lib PRIVATE -march=armv9-a+sme)
endif()

# Test Executable for Week 3
add_executable(test_week3 ../../tests/test_week3.cpp)
add_executable(bench_week3 ../../apps/main_week3.cpp)
target_link_libraries(test_week3 PRIVATE week3_lib Catch2::Catch2WithMain)
target_link_libraries(bench_week3 PRIVATE week3_lib)

# Register the test for 'ctest'
add_test(NAME Week3Tests COMMAND test_week3)