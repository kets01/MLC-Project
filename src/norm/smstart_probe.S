// smstart_probe.S — what the streaming-mode transition actually costs.
//
// Every per-call figure in this project was previously INFERRED from the
// intercept of a linear fit.  That instrument had already failed once: Sprint
// 2a read t0 ~ 1 us/call from it, and ~70% of that turned out to be an
// uncached cpu_supports_sme() sysctl in the wrapper.  An inferred quantity had
// absorbed an unrelated bug, so this measures the transition directly.
//
// Method: two entry points differing in exactly one thing — a loop of
// `iters` x {smstart; smstop} against the identical loop with the pair
// removed.  The difference over `iters` is the transition with loop overhead
// subtracted; branch structure, counter and alignment are otherwise the same.
//
// Both bare-form (SM + ZA) and SM-only variants are measured: the ZA kernels
// pay the first, the SSVE kernels the second.  They came out equal, so
// managing PSTATE.ZA is free.
//
// ABI: entering or leaving streaming mode zeroes d8-d15, which AAPCS64 makes
// callee-saved.  These probes save all eight even though they name none — a
// mova microbenchmark that skipped this reported `inf`.
//
// C signatures:
//   void smstart_probe_pairs(int64_t iters);    // x0
//   void smstart_probe_sm_only(int64_t iters);  // x0
//   void smstart_probe_empty(int64_t iters);    // x0
//   int64_t svl_fp32_lanes(void);               // -> x0

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(smstart_probe_pairs)
.global SYMBOL(smstart_probe_empty)
.global SYMBOL(smstart_probe_sm_only)
.global SYMBOL(svl_fp32_lanes)

.text
.align 4

// ---------------------------------------------------------------------------
// svl_fp32_lanes -> x0: streaming vector length in FP32 lanes.
//
// Decision D says never hard-code SVL.  RDSVL returns it in bytes WITHOUT
// entering streaming mode, so there is no smstart/smstop pair and no
// d8-d15 hazard; divide by 4 for lanes.
// ---------------------------------------------------------------------------
SYMBOL(svl_fp32_lanes):
    rdsvl   x0, #1          // x0 = SVL in bytes
    lsr     x0, x0, #2      // /4 -> FP32 lanes
    ret

// --- save/restore d8-d15 (64 bytes) -----------------------------------------
.macro SAVE_D8_D15
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
.endm

.macro RESTORE_D8_D15
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
.endm

// smstart_probe_pairs: iters x { smstart ; smstop }   (SM + ZA)
SYMBOL(smstart_probe_pairs):
    SAVE_D8_D15
    cbz     x0, 2f
1:
    smstart
    smstop
    subs    x0, x0, #1
    b.ne    1b
2:
    RESTORE_D8_D15
    ret

// ---------------------------------------------------------------------------
// smstart_probe_sm_only: iters x { smstart sm ; smstop sm }.  This is what
// the SSVE kernels pay; the pairs above are what the ZA kernels pay.
// kernels pay.  Measuring both shows whether managing PSTATE.ZA costs extra.
// ---------------------------------------------------------------------------
SYMBOL(smstart_probe_sm_only):
    SAVE_D8_D15
    cbz     x0, 2f
1:
    smstart sm
    smstop  sm
    subs    x0, x0, #1
    b.ne    1b
2:
    RESTORE_D8_D15
    ret

// smstart_probe_empty: the control — the identical loop without the
// transition, so subtracting its time leaves the transition itself.
SYMBOL(smstart_probe_empty):
    SAVE_D8_D15
    cbz     x0, 2f
1:
    subs    x0, x0, #1
    b.ne    1b
2:
    RESTORE_D8_D15
    ret
