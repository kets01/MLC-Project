# Create Library for Week 2
add_library(week2_lib 
    fmadd_asm.S 
)

# Benchmark App for Week 2
add_executable(bench_week2 ../../apps/main_week2.cpp)
target_link_libraries(bench_week2 PRIVATE week2_lib)

# Test Executable for Week 2
add_executable(test_week2 ../../tests/test_week2.cpp)
target_link_libraries(test_week2 PRIVATE week2_lib Catch2::Catch2WithMain)

# Register test
add_test(NAME Week2Tests COMMAND test_week2)