add_library(norm_lib
    reference.cpp
    norm_ssve.cpp       # C++ SME guard + extern-C bridge
    rms_norm_ssve.S     # hand-written SSVE kernel (Sprint 2)
)

# -march=armv9-a+sme: lets the assembler accept SME/SVE mnemonics
# (smstart, smstop, whilelo, fmla, faddv, incw, …).
# Matches the pattern used by week3_lib and week6_lib.
target_compile_options(norm_lib PRIVATE -march=armv9-a+sme)

# norm_ssve.cpp includes week3/utility.hpp (cpu_supports_sme).
target_include_directories(norm_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_norm ../../tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE norm_lib Catch2::Catch2WithMain)

add_executable(main_norm ../../apps/main_norm.cpp)
target_link_libraries(main_norm PRIVATE norm_lib)

add_test(NAME NormTests COMMAND test_norm)
