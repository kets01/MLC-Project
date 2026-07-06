add_library(norm_lib
    reference.cpp
    norm_ssve.cpp       # C++ SME guard + extern-C bridge
    rms_norm_ssve.S     # hand-written SSVE kernel (Sprint 2)
)

# rms_norm_ssve.S already carries a .arch armv9-a+sme directive, so the
# inline assembler knows the ISA without a library-wide -march flag.
# We must NOT apply -march=armv9-a+sme to reference.cpp: the compiler would
# auto-vectorize with SVE, but non-streaming SVE is unavailable on Apple
# Silicon (M4 exposes SVE only inside a streaming region).
# So we scope -march only to the two files that actually need it.
set_source_files_properties(
    norm_ssve.cpp
    rms_norm_ssve.S
    PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme"
)

# norm_ssve.cpp includes week3/utility.hpp (cpu_supports_sme).
target_include_directories(norm_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_norm ../../tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE norm_lib Catch2::Catch2WithMain)

add_executable(main_norm ../../apps/main_norm.cpp)
target_link_libraries(main_norm PRIVATE norm_lib)

add_test(NAME NormTests COMMAND test_norm)
