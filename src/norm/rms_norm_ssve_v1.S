// rms_norm_ssve_v1.S — Sprint 2 ablation V1
//
// Change vs V0 (rms_norm_ssve.S):
//   inv_rms replaces FSQRT + 1.0/FDIV with FRSQRTE + one Newton-Raphson step.
//   Everything else is identical to V0.
//
// NR sequence for 1/sqrt(a), where a = mean_square + eps:
//   e = FRSQRTE(a)        — estimate, e ~= 1/sqrt(a)
//   t = e * a             — NR intermediate
//   c = FRSQRTS(e, t)     — correction: c = (3 - e*t) / 2
//   inv = e * c           — refined inv_rms, FP32-accurate after one step
//
// Inactive-lane safety: after the reduction, inactive Z2 lanes hold 0.0.
// FRSQRTE is unpredicated so it sees 0 there, producing +inf.  The NR
// steps propagate NaN into those lanes.  The final FMUL is predicated (p1/m)
// so inactive Z4 lanes are never updated, and pass-2 only reads Z4 under
// p1/m, so junk never reaches output.  FP traps are masked by default FPCR.
// We fill inactive Z2 lanes with eps via SEL before FRSQRTE to keep all
// intermediates finite and avoid setting FPSR exception flags.
//
// C signature (identical to V0):
//   void rms_norm_ssve_v1(const float *a,     // x0
//                         float       *b,     // x1
//                         const float *gamma, // x2
//                         int64_t      m,     // x3
//                         int64_t      n,     // x4
//                         int64_t      ld_a,  // x5
//                         int64_t      ld_b,  // x6
//                         float        eps);  // s0

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(rms_norm_ssve_v1)

.text
.align 4

SYMBOL(rms_norm_ssve_v1):

    // Prologue — frame extended to 80 bytes (was 64) to fit a dedicated eps
    // stash slot at [sp,#64] alongside the original layout.
    stp  x19, x20, [sp, #-80]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    str  x25,      [sp, #48]
    str  d8,       [sp, #56]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    lsl   x24, x5, #2
    lsl   x25, x6, #2
    fmov  s8,  s0
    str   s0,  [sp, #64]       // dedicated eps stash, stored BEFORE smstart
                                // (streaming-mode entry clobbers D8 too, not just
                                // D9-D15 — memory survives smstart, no register
                                // does; verified empirically)

    cbz   x22, .Lrow_done_v1

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #64]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, eps, ...}

    mov    x0, #0

    // -----------------------------------------------------------------------
    // Outer loop — one iteration per block of VL rows.
    // -----------------------------------------------------------------------
.Lblock_loop_v1:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done_v1

    // === Pass 1: sum of squares =============================================
    mov    z2.s, #0
    mov    x8,   x19
    mov    x9,   x23

.Lred_col_loop_v1:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s,   p1/m, z0.s, z0.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_col_loop_v1

    // === Compute inv_rms (V1 change) ========================================
    //
    // V0 used: FSQRT z2 + FMOV z4,1.0 + FDIV z4,z4,z2  (~24 cycles latency)
    // V1 uses: FRSQRTE + one NR step                     (~12 cycles latency)
    //
    // Division by N is still a vector FDIV here — removed in V2.
    scvtf  s0, x23
    dup    z3.s, z0.s[0]
    fdiv   z2.s, p1/m, z2.s, z3.s     // z2 /= N  (mean square)

    dup    z3.s, z8.s[0]               // z3 = {eps, eps, ...}
    fadd   z2.s, p1/m, z2.s, z3.s     // z2 += eps  (active lanes only)
    sel    z2.s, p1, z2.s, z3.s       // z2[inactive] = eps (safe for FRSQRTE)

    frsqrte z4.s, z2.s                 // z4 ~= 1/sqrt(z2)  (estimate, all lanes)
    fmul    z3.s, z4.s, z2.s           // z3 = z4 * z2  (NR intermediate)
    frsqrts z3.s, z4.s, z3.s          // z3 = (3 - z4*z3) / 2  (NR correction)
    fmul    z4.s, p1/m, z4.s, z3.s    // z4[active] = refined inv_rms

    // === Pass 2: normalize + scale =========================================
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Lnrm_col_loop_v1:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x10]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_loop_v1

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lblock_loop_v1

.Lblock_done_v1:
    smstop sm

.Lrow_done_v1:
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
