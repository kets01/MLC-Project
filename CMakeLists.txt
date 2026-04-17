cmake_minimum_required(VERSION 3.10)

# 1. Initialize Project (Enables C++ and AArch64 Assembly)
project(MLC-project CXX ASM)

# Requirement: C++17 and strict warnings (the "No Hacks" rule)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    # -Wall -Wextra: Show all warnings
    # -Werror: Treat warnings as errors (Strict requirement!)
    # -g: Include debugging info for GDB task
    add_compile_options(-Wall -Wextra -Werror -pedantic -g)
endif()

# 2. Setup Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # Or the latest stable version
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Catch2)

# 3. Create the Math Logic Library
add_library(math_logic 
    src/week1/base_math.cpp 
    src/week1/base_math.S
)
target_include_directories(math_logic PUBLIC include)

# 4. Create the Unit Tests
enable_testing()
add_executable(unit_tests tests/test_weeek1.cpp)

# Link to our logic AND Catch2
# 'Catch2::Catch2WithMain' provides a main() function automatically
target_link_libraries(unit_tests PRIVATE math_logic Catch2::Catch2WithMain)

# Add the test command for CI
include(Catch)
catch_discover_tests(unit_tests)

