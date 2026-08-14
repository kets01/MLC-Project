// smstart_probe.S — Sprint 7b: the cost of the streaming-mode transition itself.
//
// WHY THIS EXISTS
// ---------------
// Every "fixed per-call cost" figure in this project so far was INFERRED, by
// fitting t(N) = t0 + b*N to a sweep and reading the intercept.  Sprint 2a did
// that and reported t0 ~ 1 us/call.  Sprint 4 then found roughly 70 % of that
// intercept was not streaming cost at all but an uncached `cpu_supports_sme()`
// sysctl the wrapper performed on every call.  An inferred quantity absorbed a
// bug that had nothing to do with the thing being measured.
//
// This probe measures the transition DIRECTLY instead of inferring it, so the
// two numbers can be compared and the intercept can finally be decomposed.
//
// METHOD
// ------
// Two entry points that differ in exactly one thing:
//
//   smstart_probe_pairs(iters)  — a loop of `iters` x { smstart; smstop }
//   smstart_probe_empty(iters)  — the same loop with the pair removed
//
// The difference of the two timings, divided by `iters`, is the cost of one
// smstart/smstop pair with the loop overhead subtracted out.  Everything else
// about the two paths is identical: same branch structure, same counter, same
// alignment directive, same prologue/epilogue.
//
// Both are bare `smstart`/`smstop` (SM + ZA), matching what the ZA kernels do.
// The SM-only forms are measured separately by the caller passing the variant.
//
// ABI NOTE (the bug this project has hit repeatedly — Sprint 4 #6/#8,
// Sprint 5's prerequisite, Sprint 6 §1.1): entering or leaving streaming mode
// ZEROES d8-d15, which alias the low 128 bits of z8-z15 and which AAPCS64
// makes callee-saved.  A probe that does not save them corrupts its caller,
// and Sprint 6 Part 3 #5 records a `mova` microbenchmark that reported `inf`
// for exactly this reason.  This file therefore saves and restores all eight
// even though it never names them — the hazard comes from the transition, not
// from the register allocation.
//
// C signatures:
//   void smstart_probe_pairs(int64_t iters);   // x0
//   void smstart_probe_empty(int64_t iters);   // x0
//   void smstart_probe_sm_only(int64_t iters); // x0

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
// svl_fp32_lanes: the streaming vector length, in FP32 lanes.
//
//   int64_t svl_fp32_lanes(void);   // -> x0
//
// Decision D says never hard-code SVL; query it.  Until now nothing on the C++
// side could, so the harness would have had to write "64 rows per group" as a
// literal.  RDSVL is the instruction meant for exactly this: it returns the
// streaming vector length in bytes WITHOUT entering streaming mode, so no
// smstart/smstop pair (and no d8-d15 hazard) is involved.  Divide by 4 for
// FP32 lanes.  On the M4, SVL = 512 bits = 64 bytes -> 16 lanes.
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

// ---------------------------------------------------------------------------
// smstart_probe_pairs: iters x { smstart ; smstop }   (SM + ZA)
// ---------------------------------------------------------------------------
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
// smstart_probe_sm_only: iters x { smstart sm ; smstop sm }
//
// The SSVE norm kernels use the SM-only forms (they never touch ZA), so this
// is the variant that matters for V6/V7; the pairs above are what the ZA
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

// ---------------------------------------------------------------------------
// smstart_probe_empty: the identical loop WITHOUT the transition.
//
// This is the control.  Subtracting its time removes the loop's own branch and
// counter cost, so what remains is attributable to the transition rather than
// to the harness measuring a loop.
// ---------------------------------------------------------------------------
SYMBOL(smstart_probe_empty):
    SAVE_D8_D15
    cbz     x0, 2f
1:
    subs    x0, x0, #1
    b.ne    1b
2:
    RESTORE_D8_D15
    ret
