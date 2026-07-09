add_library(norm_lib
    reference.cpp
    norm_ssve.cpp               # C++ SME guard + extern-C bridge
    jit_norm.cpp                # Sprint 4: mini_jit::Norm JIT generator
    layer_norm_ssve.S           # Sprint 2c: LayerNorm SSVE kernel V0 (Mariza)
    layer_norm_ssve_v1.S        # V1: FRSQRTE+NR replaces FSQRT+FDIV
    layer_norm_ssve_v2.S        # V2: V1 + pre-computed 1/N
    layer_norm_ssve_v4.S        # V4: V2 + 4 independent accumulators (both passes)
    layer_norm_ssve_v5.S        # V5: V4 + software-pipelined loads
    layer_norm_ssve_v6.S        # V6: 4-row-block contiguity grouping
    layer_norm_ssve_welford.S   # Welford: online single-pass mean+var
    layer_norm_za.S             # Sprint 3: ZA-tile residency (3R+1W -> 1R+1W fusion, gated)
    rms_norm_ssve.S             # V0: baseline SSVE kernel
    rms_norm_ssve_v1.S          # V1: FRSQRTE+FRSQRTS (no FSQRT/FDIV)
    rms_norm_ssve_v2.S          # V2: V1 + pre-computed 1/N
    rms_norm_ssve_v3.S          # V3: V2 + 2x column-loop unroll
    rms_norm_ssve_v4.S          # V4: V2 + 4 independent FMLA accumulators
    rms_norm_ssve_v5.S          # V5: V4 + software-pipelined loads
    rms_norm_ssve_v6.S          # V6: 4-row-block contiguity grouping
    rms_norm_za.S               # Sprint 3: ZA-tile residency (1R+1W fusion)
    bw_probe_ssve.S             # Sprint 2a: streaming-mode roofline probe
)

# layer_norm_ssve.S and rms_norm_ssve*.S carry a .arch armv9-a+sme directive
# themselves, but CMake still needs COMPILE_FLAGS so the assembler driver
# accepts the file.  norm_ssve.cpp gets the flag so the compiler resolves any
# SME-related intrinsics it may transitively see.
# We must NOT apply -march=armv9-a+sme to reference.cpp: the compiler would
# auto-vectorize with SVE, but non-streaming SVE is unavailable on Apple
# Silicon (M4 exposes SVE only inside a streaming region).
set_source_files_properties(
    norm_ssve.cpp
    layer_norm_ssve.S
    layer_norm_ssve_v1.S
    layer_norm_ssve_v2.S
    layer_norm_ssve_v4.S
    layer_norm_ssve_v5.S
    layer_norm_ssve_v6.S
    layer_norm_ssve_welford.S
    layer_norm_za.S
    rms_norm_ssve.S
    rms_norm_ssve_v1.S
    rms_norm_ssve_v2.S
    rms_norm_ssve_v3.S
    rms_norm_ssve_v4.S
    rms_norm_ssve_v5.S
    rms_norm_ssve_v6.S
    rms_norm_za.S
    bw_probe_ssve.S
    PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme"
)

# norm_ssve.cpp includes week3/utility.hpp (cpu_supports_sme).
target_include_directories(norm_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# jit_norm.cpp emits via week6 InstGen into a week5 JitEngine buffer
# (jit_engine.hpp is header-only; InstGen needs the week6 library).
target_link_libraries(norm_lib PRIVATE week6_lib)

add_executable(test_norm ../../tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE norm_lib week3_lib Catch2::Catch2WithMain)

add_executable(main_norm ../../apps/main_norm.cpp)
target_link_libraries(main_norm PRIVATE norm_lib)

add_test(NAME NormTests COMMAND test_norm)
