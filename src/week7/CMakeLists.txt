# Standard find for CI
find_package(OpenMP)

add_library(week7_lib TeirRuntime.cpp)

# Specifically add the Homebrew paths found on Edward for local build
target_include_directories(week7_lib PRIVATE /opt/homebrew/opt/libomp/include)
target_link_libraries(week7_lib PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
# Enable SME/SVE features
if(NOT MSVC)
    target_compile_options(week7_lib PRIVATE -march=armv9-a+sme)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(week7_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(week7_lib PRIVATE week6_lib)

add_executable(bench_week7 ../../apps/main_week7.cpp)
target_link_libraries(bench_week7 PRIVATE week7_lib)

add_executable(test_week7 ../../tests/test_week7.cpp)
target_link_libraries(test_week7 PRIVATE week7_lib Catch2::Catch2WithMain)