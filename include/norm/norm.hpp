#pragma once
#include <cstdint>

namespace mini_jit::norm {

// Canonical kernel interface (decision A): shared by the C++ reference,
// the mini_jit::Norm JIT generator, and the TEIR registration.
//
// Layout: column-major, explicit leading dimension.  data[row + col*ld]
// Normalized axis: N (each of the M rows is normalized independently).
// gamma, beta: per-feature scale/shift vectors [N elements].

// LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
// Two-pass: pass 1 computes mean and variance, pass 2 normalizes.
void layer_norm_ref(const float* a,
                    float*       b,
                    const float* gamma,
                    const float* beta,
                    int64_t      m,
                    int64_t      n,
                    int64_t      ld_a,
                    int64_t      ld_b,
                    float        epsilon);

// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
// Single-pass: no mean subtraction, no beta.
void rms_norm_ref(const float* a,
                  float*       b,
                  const float* gamma,
                  int64_t      m,
                  int64_t      n,
                  int64_t      ld_a,
                  int64_t      ld_b,
                  float        epsilon);

// LayerNorm — hand-written SSVE kernel (Sprint 2c).
// Same signature as layer_norm_ref; verified against it in tests.
// Returns immediately (no-op) when SME is absent so the caller can skip.
void layer_norm_ssve(const float* a,
                     float*       b,
                     const float* gamma,
                     const float* beta,
                     int64_t      m,
                     int64_t      n,
                     int64_t      ld_a,
                     int64_t      ld_b,
                     float        epsilon);

// RMSNorm — hand-written Streaming SVE kernel (Sprint 2, V0 baseline).
// Same interface as rms_norm_ref; requires cpu_supports_sme() == true.
// Returns immediately (no-op) when SME is absent so the caller can skip.
void rms_norm_ssve(const float* a,
                   float*       b,
                   const float* gamma,
                   int64_t      m,
                   int64_t      n,
                   int64_t      ld_a,
                   int64_t      ld_b,
                   float        epsilon);

// V1: replace FSQRT+FDIV with FRSQRTE+FRSQRTS (reciprocal sqrt estimate + NR).
void rms_norm_ssve_v1(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V2: V1 + pre-compute 1/N once before the outer loop (eliminates inner FDIV).
void rms_norm_ssve_v2(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V3: V2 + 2x column-loop unroll in both passes.
void rms_norm_ssve_v3(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V4: V2 + four independent FMLA accumulator chains in the reduction pass
// (memory-level-parallelism lever; Sprint 2b round two).
void rms_norm_ssve_v4(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V5: V4 + explicit software-pipelining of the reduction loads (group B
// loads issued ahead of group A's FMLAs, rotating; Sprint 2b round two).
void rms_norm_ssve_v5(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V6: contiguity blocking — four consecutive VL-row blocks per group, so
// each column touch is 256 B contiguous (4x denser DRAM access); one
// accumulator per block keeps V4's ILP with the reference's summation order.
void rms_norm_ssve_v6(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// Sprint 2a roofline probe (NOT a norm kernel): STREAM-style scale-add
// d[i] = s[i] + 1.0f executed in streaming mode with contiguous LD1W/ST1W —
// measures the single-core bandwidth ceiling the SSVE kernels can actually
// reach (the compiler-vectorized C++ probe runs in NEON mode, a different
// execution mode and therefore a different ceiling).
// No-op when cpu_supports_sme() == false.
void bw_probe_ssve(float*       d,
                   const float* s,
                   int64_t      n);

} // namespace mini_jit::norm
