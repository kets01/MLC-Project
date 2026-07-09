cmake_minimum_required(VERSION 3.10)
project(MLC-Project CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Strict warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror -pedantic -g)
endif()

# Fetch Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.0
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Catch2)

include_directories(include)

# Must precede the add_subdirectory() calls: add_test() only registers with
# CTest in directories processed while testing is already enabled.  (It used
# to sit at the bottom of this file, which is why `ctest` found no tests.)
enable_testing()

# The weekly logic is now ONLY in these folders
add_subdirectory(src/week1)
add_subdirectory(src/week2)
add_subdirectory(src/week3)
add_subdirectory(src/week5)
add_subdirectory(src/week6)
add_subdirectory(src/week7)
add_subdirectory(src/norm)