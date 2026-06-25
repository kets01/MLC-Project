add_library(norm_lib reference.cpp layer_norm_ssve.S layer_norm_ssve.cpp)
# Only the assembly file needs the SME march flag.
# Applying it to C++ sources causes the compiler to auto-vectorize with
# regular SVE, which SIGILLs on Apple Silicon (SVE is only exposed in
# streaming mode; non-streaming SVE is absent on M-series chips).
set_source_files_properties(layer_norm_ssve.S
    PROPERTIES COMPILE_OPTIONS "-march=armv9-a+sme")

add_executable(test_norm ../../tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE norm_lib week3_lib Catch2::Catch2WithMain)

add_executable(main_norm ../../apps/main_norm.cpp)
target_link_libraries(main_norm PRIVATE norm_lib)

add_test(NAME NormTests COMMAND test_norm)
