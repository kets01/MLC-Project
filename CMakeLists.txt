cmake_minimum_required(VERSION 3.10)
project(MLC-Project CXX ASM)

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

# Tell CMake where to find the headers for all weeks
include_directories(include)

# Now "Enter" the weekly folders
add_subdirectory(src/week1)
add_subdirectory(src/week2)

enable_testing()