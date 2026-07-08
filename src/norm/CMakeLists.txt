add_library(norm_lib
    reference.cpp
    norm_ssve.cpp           # C++ SME guard + extern-C bridge
    rms_norm_ssve.S         # V0: baseline SSVE kernel
    rms_norm_ssve_v1.S      # V1: FRSQRTE+FRSQRTS (no FSQRT/FDIV)
    rms_norm_ssve_v2.S      # V2: V1 + pre-computed 1/N
    rms_norm_ssve_v3.S      # V3: V2 + 2x column-loop unroll
    rms_norm_ssve_v4.S      # V4: V2 + 4 independent FMLA accumulators
    rms_norm_ssve_v5.S      # V5: V4 + software-pipelined loads
    rms_norm_ssve_v6.S      # V6: 4-row-block contiguity grouping
    bw_probe_ssve.S         # Sprint 2a: streaming-mode roofline probe
)

# rms_norm_ssve*.S carry a .arch armv9-a+sme directive themselves, but
# CMake still needs COMPILE_FLAGS so the assembler driver accepts the file.
# We must NOT apply -march=armv9-a+sme to reference.cpp: the compiler would
# auto-vectorize with SVE, but non-streaming SVE is unavailable on Apple
# Silicon (M4 exposes SVE only inside a streaming region).
set_source_files_properties(
    norm_ssve.cpp
    rms_norm_ssve.S
    rms_norm_ssve_v1.S
    rms_norm_ssve_v2.S
    rms_norm_ssve_v3.S
    rms_norm_ssve_v4.S
    rms_norm_ssve_v5.S
    rms_norm_ssve_v6.S
    bw_probe_ssve.S
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
