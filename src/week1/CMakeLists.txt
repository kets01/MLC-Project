# Create Library for Week 1
add_library(week1_lib 
    base_math.cpp 
    base_math.S
)

# Test Executable for Week 1
add_executable(test_week1 ../../tests/test_week1.cpp)
target_link_libraries(test_week1 PRIVATE week1_lib Catch2::Catch2WithMain)

# Register test
add_test(NAME Week1Tests COMMAND test_week1)