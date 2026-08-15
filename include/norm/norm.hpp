#pragma once
#include <cstdint>

// Canonical kernel interface (decision A): shared by the C++ reference, the
// mini_jit::Norm JIT generator, and the TEIR registration.
//
// Layout: column-major with an explicit leading dimension, data[row + col*ld].
// The normalized axis is N — each of the M rows is normalized independently.
// gamma and beta are per-feature vectors of N elements.

namespace mini_jit::norm {

// ===========================================================================
// PUBLIC API — use these unless you specifically want one named kernel.
//
// These dispatchers ALWAYS compute a correct result, on any CPU:
//     FEAT_SME2 -> V7 (SME2 multi-vector)
//     FEAT_SME  -> V6 (SSVE, 4-row-block contiguity)
//     otherwise -> the scalar C++ reference
//
// The ISA-specific entry points below used to `return;` silently when SME was
// absent, so a caller could not tell success from a no-op and would go on to
// read an untouched buffer.  That is why this layer exists.
// ===========================================================================

void layer_norm(const float* a,
                float*       b,
                const float* gamma,
                const float* beta,
                int64_t      m,
                int64_t      n,
                int64_t      ld_a,
                int64_t      ld_b,
                float        epsilon);

void rms_norm(const float* a,
              float*       b,
              const float* gamma,
              int64_t      m,
              int64_t      n,
              int64_t      ld_a,
              int64_t      ld_b,
              float        epsilon);

// Which implementation this host selects: "SME2 (V7)", "SME (V6)" or the
// scalar reference.  Printed in the benchmark's provenance block so a reported
// number always says which code ran.
const char* norm_dispatch_target();

// ===========================================================================
// ISA-SPECIFIC ENTRY POINTS
//
// PRECONDITION: cpu_supports_sme(), and cpu_supports_sme2() for the SME2
// variants.  Calling one without the required feature prints a diagnostic
// naming the function and aborts — it never returns an uncomputed buffer, and
// never quietly substitutes a different variant.  Guard the call, or use the
// dispatchers above, which are the layer allowed to fall back.
// ===========================================================================

// LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
// Two reduction stages (mean, then variance) plus output generation, so the
// baseline performs THREE traversals of the input.
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
// One reduction stage plus output generation, so TWO traversals — one fewer
// than LayerNorm, which is the structural source of its throughput advantage.
void rms_norm_ref(const float* a,
                  float*       b,
                  const float* gamma,
                  int64_t      m,
                  int64_t      n,
                  int64_t      ld_a,
                  int64_t      ld_b,
                  float        epsilon);

// --- LayerNorm kernels.  All share layer_norm_ref's signature. -------------

// V0: hand-written SSVE baseline, three predicated passes.
void layer_norm_ssve(const float* a,
                     float*       b,
                     const float* gamma,
                     const float* beta,
                     int64_t      m,
                     int64_t      n,
                     int64_t      ld_a,
                     int64_t      ld_b,
                     float        epsilon);

// V1: FRSQRTE + one Newton-Raphson step replaces FSQRT+FDIV for inv_std.
void layer_norm_ssve_v1(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V2: V1 + 1/N precomputed once, removing both per-block FDIVs.
void layer_norm_ssve_v2(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V4: V2 + four independent accumulator chains in both reduction passes.
void layer_norm_ssve_v4(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V5: V4 + software-pipelined loads in both reduction passes.
void layer_norm_ssve_v5(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V6 (incumbent): 4-row-block contiguity grouping — all three passes run on a
// group of four consecutive VL-row blocks, making each column touch 256
// contiguous bytes; four accumulators keep V4's ILP.
void layer_norm_ssve_v6(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// Welford: online single-pass (mean, M2).  Cuts traffic 3R+1W -> 2R+1W but
// needs a scalar FDIV per column, which serialises the loop; measured slower
// than V6, and kept as the ablation row that shows it.
void layer_norm_ssve_welford(const float* a,
                             float*       b,
                             const float* gamma,
                             const float* beta,
                             int64_t      m,
                             int64_t      n,
                             int64_t      ld_a,
                             int64_t      ld_b,
                             float        epsilon);

// V7: V6 with SME2 multi-vector loads/stores.  Same traffic and same summation
// order as V6, so it must be bit-identical to it.
// PRECONDITION: cpu_supports_sme2(); aborts otherwise, it does NOT fall back
// to V6 — a benchmark labelled V7 must never silently measure V6.
void layer_norm_ssve_v7(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// ZA residency rebuilt on SME2 multi-vector MOVA.  LayerNorm stages 3 MOVAs
// per element to RMSNorm's 2, so it has proportionally more to gain from the
// 4:1 fold.  PRECONDITION: cpu_supports_sme2(); aborts otherwise.
void layer_norm_za_sme2(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// ZA residency: x is staged in ZA during the mean pass and reused from ZA for
// both the variance and normalize passes (3R+1W -> 1R+1W) whenever a row fits
// (N <= 4*SVL); wider rows take a streaming three-pass fallback.  A measured
// negative result — see the kernel header.
void layer_norm_za(const float* a,
                   float*       b,
                   const float* gamma,
                   const float* beta,
                   int64_t      m,
                   int64_t      n,
                   int64_t      ld_a,
                   int64_t      ld_b,
                   float        epsilon);

// --- RMSNorm kernels.  All share rms_norm_ref's signature. -----------------

// V0: hand-written SSVE baseline, two predicated passes.
void rms_norm_ssve(const float* a,
                   float*       b,
                   const float* gamma,
                   int64_t      m,
                   int64_t      n,
                   int64_t      ld_a,
                   int64_t      ld_b,
                   float        epsilon);

// V1: FRSQRTE + Newton-Raphson replaces FSQRT+FDIV for inv_rms.
void rms_norm_ssve_v1(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V2: V1 + 1/N precomputed once before the outer loop.
void rms_norm_ssve_v2(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V3: V2 + 2x column-loop unroll.
void rms_norm_ssve_v3(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V4: V2 + four independent FMLA accumulator chains.
void rms_norm_ssve_v4(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V5: V4 + software-pipelined reduction loads.
void rms_norm_ssve_v5(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V6 (incumbent): 4-row-block contiguity grouping — 256 B per column touch,
// one accumulator per block, which keeps V4's ILP at the reference's
// summation order.
void rms_norm_ssve_v6(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V7: V6 with SME2 multi-vector loads/stores — same traffic and arithmetic,
// fewer instructions, so a single-variable test of whether V6 is issue-bound
// or memory-bound.
// PRECONDITION: cpu_supports_sme2(); aborts otherwise, it does NOT fall back.
void rms_norm_ssve_v7(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V7x2: the 2-vector CONTROL for V7 — identical work and identical 256 B per
// column, issued as two 2-vector accesses.  Measurement variant; not emitted
// by the JIT.  PRECONDITION: cpu_supports_sme2(); aborts otherwise.
void rms_norm_ssve_v7x2(const float* a,
                        float*       b,
                        const float* gamma,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// ZA residency on SME2 multi-vector MOVA: four columns staged per MOVA instead
// of one.  PRECONDITION: cpu_supports_sme2(); aborts otherwise.
void rms_norm_za_sme2(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// ZA residency: pass 2 reads x from ZA instead of memory (2R+1W -> 1R+1W)
// whenever a row fits (N <= 4*SVL); wider rows take a streaming two-pass
// fallback.  A measured negative result — see the kernel header.
void rms_norm_za(const float* a,
                 float*       b,
                 const float* gamma,
                 int64_t      m,
                 int64_t      n,
                 int64_t      ld_a,
                 int64_t      ld_b,
                 float        epsilon);

// --- Instruments (not norm kernels) ----------------------------------------

// Roofline probe: STREAM-style d[i] = s[i] + 1.0f in streaming mode with
// contiguous LD1W/ST1W — the single-core ceiling the SSVE kernels can actually
// reach.  The compiler-vectorized C++ probe runs in NEON, a different mode and
// therefore a different ceiling.
void bw_probe_ssve(float*       d,
                   const float* s,
                   int64_t      n);

// Streaming-transition probes.  Every "fixed per-call cost" before these was
// inferred from a linear fit's intercept, and that intercept once absorbed an
// unrelated syscall bug; these measure the transition directly.
//
//   pairs   — iters x { smstart ; smstop }        (SM + ZA, what ZA kernels pay)
//   sm_only — iters x { smstart sm ; smstop sm }  (what the SSVE kernels pay)
//   empty   — the same loop with no transition; the control that is subtracted
void smstart_probe_pairs(int64_t iters);
void smstart_probe_sm_only(int64_t iters);
void smstart_probe_empty(int64_t iters);

// The streaming vector length in FP32 lanes, via RDSVL (decision D: never
// hard-code SVL).  RDSVL needs no streaming mode, so there is no PSTATE
// transition.  Returns 0 without SME rather than a guessed default — silently
// returning 16 would reintroduce the hard-coded SVL decision D forbids.
// V6/V7 process 4 VL-row blocks per group, so 4 * svl_fp32_lanes() is the
// kernels' row granularity.
int64_t svl_fp32_lanes();

} // namespace mini_jit::norm
