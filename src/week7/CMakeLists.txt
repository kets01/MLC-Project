
# OpenMP is detected once in the root CMakeLists (see the note there).

add_library(week7_lib TeirRuntime.cpp TeirParser.cpp)

if(NOT MSVC)
    target_compile_options(week7_lib PRIVATE -march=armv9-a+sme)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(week7_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# load_teir() resolves bare filenames (e.g. "matmul.teir") against the
# real data/ directory, independent of whatever working directory the
# test runner or bench binary is invoked from.
target_compile_definitions(week7_lib PRIVATE MLC_PROJECT_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

target_link_libraries(week7_lib PRIVATE week6_lib norm_lib)

add_executable(bench_week7 ../../apps/main_week7.cpp)
target_link_libraries(bench_week7 PRIVATE week7_lib)

add_executable(test_week7 ../../tests/test_week7.cpp)
target_link_libraries(test_week7 PRIVATE week7_lib Catch2::Catch2WithMain)

# Every other week registers its tests with CTest; week7 never did, so the
# TEIR tests were invisible to `ctest` (and therefore to CI).
add_test(NAME Week7Tests COMMAND test_week7)