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

// V1: replace FSQRT+FDIV for inv_std with FRSQRTE + one Newton-Raphson step
// (SEL+FRSQRTE+FRSQRTS+FMUL); passes 1 and 3 are unchanged from V0.
void layer_norm_ssve_v1(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V2: V1 + pre-compute 1/N once (eliminates per-block FDIV in both reduction
// passes; adopts RMSNorm-style ADDVL/INCW outer loop).
void layer_norm_ssve_v2(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V4: V2 + four independent accumulator chains in BOTH reduction passes
// (mean and variance) to expose instruction-level parallelism.
void layer_norm_ssve_v4(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V5: V4 + software-pipelined loads in both reduction passes (B-group loads
// issued ahead of A-group FP compute; expected ~0% gain on M4 OOO core).
void layer_norm_ssve_v5(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// V6: 4-row-block contiguity grouping — all three passes run on each group of
// four consecutive VL-row blocks before advancing, making each column touch a
// 256-byte contiguous burst; four independent accumulators maintain ILP.
void layer_norm_ssve_v6(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// Welford: Welford online algorithm replaces the two reduction passes with a
// single column sweep computing per-lane (mean, M2) simultaneously.  Reduces
// traffic from 3R+1W to 2R+1W but requires a scalar FDIV per column to
// compute 1/(j+1), which serialises the column loop and is expected to be
// SLOWER than V2 for all practical N.
void layer_norm_ssve_welford(const float* a,
                             float*       b,
                             const float* gamma,
                             const float* beta,
                             int64_t      m,
                             int64_t      n,
                             int64_t      ld_a,
                             int64_t      ld_b,
                             float        epsilon);

// V7: V6 with SME2 multi-vector loads/stores — all three passes fold their
// four consecutive VL-row block accesses into one 4-vector LD1W (and pass 3's
// four stores into one 4-vector ST1W).  Same traffic and same summation order
// as V6, so it must be bit-identical to it; built to test whether the RMSNorm
// V7 result is norm-agnostic (Sprint 6).  Requires FEAT_SME2; the wrapper
// dispatches to V6 when it is absent.
void layer_norm_ssve_v7(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// LayerNorm ZA residency rebuilt on SME2 multi-vector MOVA (Sprint 6), the
// counterpart of rms_norm_za_sme2.  LayerNorm stages 3 MOVAs per element to
// RMSNorm's 2, so it has proportionally more MOVA cost for the 4:1 fold to
// remove.  Requires FEAT_SME2; falls back to V6 when absent.
void layer_norm_za_sme2(const float* a,
                        float*       b,
                        const float* gamma,
                        const float* beta,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// LayerNorm — hand-written SME/ZA kernel (Sprint 3, gated).  Full 3-pass ZA
// residency: x is staged in ZA during the mean pass and reused from ZA for
// BOTH the variance pass and the normalize pass (3R+1W -> 1R+1W, vs the SSVE
// winner V6's 3R+1W) whenever a row fits in ZA (N <= 4*SVL); wider rows fall
// back to a correct streaming three-pass.  The reduction stays SSVE
// (context.md §5).  Same interface as layer_norm_ssve_v6; requires
// cpu_supports_sme() == true, no-op otherwise.
void layer_norm_za(const float* a,
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

// V7: V6 with SME2 multi-vector loads/stores — the group loop's four
// consecutive VL-row block accesses become ONE 4-vector LD1W (and one
// 4-vector ST1W in pass 2).  Same traffic, same arithmetic, fewer
// instructions: a single-variable test of whether V6 is instruction-issue
// bound or memory bound (Sprint 6).  Requires FEAT_SME2; the wrapper
// dispatches to V6 when it is absent, so this is safe to call anywhere.
void rms_norm_ssve_v7(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// V7x2: the 2-vector CONTROL for V7 (Sprint 6).  Identical work and identical
// 256 B per column, issued as two 2-vector accesses instead of one 4-vector
// access — isolates bytes-of-demand-per-instruction as the only variable, to
// discriminate memory-level-parallelism from prefetcher explanations for V7's
// DRAM win.  Measurement variant; not emitted by the JIT.
void rms_norm_ssve_v7x2(const float* a,
                        float*       b,
                        const float* gamma,
                        int64_t      m,
                        int64_t      n,
                        int64_t      ld_a,
                        int64_t      ld_b,
                        float        epsilon);

// ZA residency rebuilt on SME2 multi-vector MOVA (Sprint 6).  Same 1R+1W
// residency idea as rms_norm_za, but four columns are staged per MOVA instead
// of one, folding ZA traffic 4:1 — built because a direct measurement showed
// MOVA is issue-bound (4.00x from the 4-vector form), which invalidates the
// premise of the Sprint-3 decision to leave ZA alone.  Requires FEAT_SME2;
// falls back to V6 when absent.
void rms_norm_za_sme2(const float* a,
                      float*       b,
                      const float* gamma,
                      int64_t      m,
                      int64_t      n,
                      int64_t      ld_a,
                      int64_t      ld_b,
                      float        epsilon);

// RMSNorm — hand-written SME/ZA kernel (Sprint 3).  Uses the ZA tiles as an
// on-core residency buffer so pass 2 reads x from ZA instead of re-reading it
// from memory (1R+1W vs the SSVE winner's 2R+1W) whenever a row fits in ZA
// (N <= 4*SVL); wider rows fall back to a correct streaming two-pass.  The
// reduction stays SSVE (context.md §5).  Same interface as rms_norm_ssve;
// requires cpu_supports_sme() == true, no-op otherwise.
void rms_norm_za(const float* a,
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

// Sprint 7b streaming-transition probes (NOT norm kernels).  Every "fixed
// per-call cost" so far was INFERRED from a linear fit's intercept; Sprint 4
// showed that intercept had absorbed an unrelated syscall bug.  These measure
// the transition directly so the intercept can be decomposed instead.
//
//   pairs   — iters x { smstart ; smstop }        (SM + ZA, what ZA kernels pay)
//   sm_only — iters x { smstart sm ; smstop sm }  (what the SSVE kernels pay)
//   empty   — the identical loop with no transition; the control whose time is
//             subtracted, so the result is the transition and not the loop.
//
// No-ops when cpu_supports_sme() == false.
void smstart_probe_pairs(int64_t iters);
void smstart_probe_sm_only(int64_t iters);
void smstart_probe_empty(int64_t iters);

// The streaming vector length in FP32 lanes, queried at runtime via RDSVL
// (decision D: never hard-code SVL).  RDSVL does not require streaming mode,
// so this is a plain call with no PSTATE transition.  Returns 0 without SME.
// V6/V7 process 4 VL-row blocks per group, so 4 * svl_fp32_lanes() is the
// kernels' row granularity — 64 on the M4's 512-bit SVL, derived not literal.
int64_t svl_fp32_lanes();

} // namespace mini_jit::norm
