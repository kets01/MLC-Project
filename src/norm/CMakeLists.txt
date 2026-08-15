add_library(norm_lib
    reference.cpp
    stability.cpp               # Sprint 7a: FP32 variance estimators (analysis only)
    norm_ssve.cpp               # C++ SME guard + extern-C bridge
    jit_norm.cpp                # Sprint 4: mini_jit::Norm JIT generator
    layer_norm_ssve.S           # Sprint 2c: LayerNorm SSVE kernel V0 (Mariza)
    layer_norm_ssve_v1.S        # V1: FRSQRTE+NR replaces FSQRT+FDIV
    layer_norm_ssve_v2.S        # V2: V1 + pre-computed 1/N
    layer_norm_ssve_v4.S        # V4: V2 + 4 independent accumulators (both passes)
    layer_norm_ssve_v5.S        # V5: V4 + software-pipelined loads
    layer_norm_ssve_v6.S        # V6: 4-row-block contiguity grouping
    layer_norm_ssve_v7.S        # Sprint 6: V6 + SME2 multi-vector LD1W/ST1W
    layer_norm_ssve_welford.S   # Welford: online single-pass mean+var
    layer_norm_za.S             # Sprint 3: ZA-tile residency (3R+1W -> 1R+1W fusion, gated)
    layer_norm_za_sme2.S        # Sprint 6: ZA residency via multi-vector MOVA
    rms_norm_ssve.S             # V0: baseline SSVE kernel
    rms_norm_ssve_v1.S          # V1: FRSQRTE+FRSQRTS (no FSQRT/FDIV)
    rms_norm_ssve_v2.S          # V2: V1 + pre-computed 1/N
    rms_norm_ssve_v3.S          # V3: V2 + 2x column-loop unroll
    rms_norm_ssve_v4.S          # V4: V2 + 4 independent FMLA accumulators
    rms_norm_ssve_v5.S          # V5: V4 + software-pipelined loads
    rms_norm_ssve_v6.S          # V6: 4-row-block contiguity grouping
    rms_norm_ssve_v7.S          # Sprint 6: V6 + SME2 multi-vector LD1W/ST1W
    rms_norm_ssve_v7x2.S        # Sprint 6: 2-vector control for V7
    rms_norm_za.S               # Sprint 3: ZA-tile residency (1R+1W fusion)
    rms_norm_za_sme2.S          # Sprint 6: ZA residency via multi-vector MOVA
    bw_probe_ssve.S             # Sprint 2a: streaming-mode roofline probe
    smstart_probe.S             # Sprint 7b: streaming-transition cost probe
)

# The .S files carry their own .arch directive, but CMake still needs
# COMPILE_FLAGS so the assembler driver accepts them; norm_ssve.cpp gets the
# flag so the compiler resolves any SME intrinsics it transitively sees.
#
# reference.cpp must NOT get it: the compiler would auto-vectorize with
# non-streaming SVE, which Apple Silicon does not expose outside a streaming
# region.
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
    smstart_probe.S
    PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme"
)

# The SME2 variants need +sme2 to ASSEMBLE — a build-time capability of the
# assembler, independent of the build host, so CI (no SME at all) still
# compiles them.  Whether they RUN is the runtime cpu_supports_sme2() check.
set_source_files_properties(
    rms_norm_ssve_v7.S
    rms_norm_ssve_v7x2.S
    rms_norm_za_sme2.S
    layer_norm_za_sme2.S
    layer_norm_ssve_v7.S
    PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme2"
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

# main_norm also drives the TEIR path and sets the OpenMP thread count.
add_executable(main_norm ../../apps/main_norm.cpp)
# week7_lib already carries norm_lib as a static-library link dependency;
# listing norm_lib again here makes Apple's ld warn about a duplicate.
target_link_libraries(main_norm PRIVATE week7_lib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(main_norm PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME NormTests COMMAND test_norm)
