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

# OpenMP — used by the week7 TEIR runtime for parallel iteration axes, and by
# the norm benchmark to set the thread count for the Sprint-5 scaling study.
# Detected here (not in a subdirectory) so every target sees the result.
#
# AppleClang supports OpenMP but ships neither libomp nor the -fopenmp driver
# flag, so FindOpenMP needs explicit hints. The standard Homebrew prefixes are
# probed directly rather than via `brew --prefix`, which is not always on the
# PATH of the shell running CMake. If libomp is absent (e.g. a bare CI runner)
# OpenMP simply stays off: the runtime's HAS_OMP fallback runs the same loop
# nests sequentially, so the build and the tests never break.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE)
    foreach(LIBOMP_PREFIX "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
        if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP)

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