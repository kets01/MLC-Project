// rms_norm_ssve_v5.S — Sprint 2b ablation V5
//
// Change vs V4 (4 independent FMLA accumulator chains):
//   Pass 1 is explicitly software-pipelined: the NEXT group of four column
//   loads (register set B: Z27-Z30) is issued BEFORE the current group's
//   FMLAs (register set A: Z0/Z24-Z26) consume their data, rotating A/B
//   each step.  The load stream therefore always runs one 4-column group
//   ahead of the compute in program order.
//
// Hypothesis (written before measuring): ~0 gain on the M4.  V4 already
// broke the accumulator dependency chain; issuing loads early in PROGRAM
// order should add nothing on a deep out-of-order core whose register
// renaming already hoists them.  V5 exists to test whether V4's win was
// limited by rename/issue depth — if it was, explicit scheduling helps;
// if not, the ≈0 result closes the "more outstanding loads" family
// (including PRFW prefetch, which works the same lever one step earlier).
//
// Pipeline structure (x9 counts columns not yet LOADED):
//   preload A                      (consumes 4)
//   loop:  load B ; FMLA A         (steady state, alternating)
//          load A ; FMLA B
//   tails: FMLA the group left in flight, then 0-3 single columns.
//
// Everything after the reduction (combine, inv_rms, pass 2) is identical
// to V4.
//
// C signature (identical to V0-V4):
//   void rms_norm_ssve_v5(const float *a,     // x0
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

.global SYMBOL(rms_norm_ssve_v5)

.text
.align 4

SYMBOL(rms_norm_ssve_v5):

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
    lsl   x24, x5, #2          // stride_a in bytes
    lsl   x25, x6, #2          // stride_b in bytes
    fmov  s8,  s0
    str   s0,  [sp, #64]       // dedicated eps stash, stored BEFORE smstart
                                // (streaming-mode entry clobbers D8 too, not
                                // just D9-D15 — memory survives smstart, no
                                // register does; verified empirically)

    cbz   x22, .Lrow_done_v5

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #64]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]

    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1
    dup   z5.s, z0.s[0]        // z5 = {1/N, ...}

    lsl   x13, x24, #1         // x13 = 2*stride_a
    mov   x0, #0

.Lblock_loop_v5:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done_v5

    // === Pass 1: pipelined 4-chain sum of squares ===========================
    mov    z2.s,  #0
    mov    z16.s, #0
    mov    z17.s, #0
    mov    z18.s, #0
    mov    x8,   x19
    mov    x9,   x23           // columns not yet loaded

    cmp    x9, #4
    b.lo   .Lred_rem_v5

    add    x12, x8, x24

    // Preload group A.
    ld1w   {z0.s},  p1/z, [x8]
    ld1w   {z24.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    ld1w   {z25.s}, p1/z, [x8]
    ld1w   {z26.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    sub    x9,  x9,  #4
    cmp    x9,  #4
    b.lo   .Lpipe_tail_a_v5

.Lpipe_loop_v5:
    // Load group B, then consume group A.
    ld1w   {z27.s}, p1/z, [x8]
    ld1w   {z28.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    ld1w   {z29.s}, p1/z, [x8]
    ld1w   {z30.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    fmla   z2.s,  p1/m, z0.s,  z0.s
    fmla   z16.s, p1/m, z24.s, z24.s
    fmla   z17.s, p1/m, z25.s, z25.s
    fmla   z18.s, p1/m, z26.s, z26.s
    sub    x9,  x9,  #4
    cmp    x9,  #4
    b.lo   .Lpipe_tail_b_v5

    // Load group A, then consume group B.
    ld1w   {z0.s},  p1/z, [x8]
    ld1w   {z24.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    ld1w   {z25.s}, p1/z, [x8]
    ld1w   {z26.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    fmla   z2.s,  p1/m, z27.s, z27.s
    fmla   z16.s, p1/m, z28.s, z28.s
    fmla   z17.s, p1/m, z29.s, z29.s
    fmla   z18.s, p1/m, z30.s, z30.s
    sub    x9,  x9,  #4
    cmp    x9,  #4
    b.hs   .Lpipe_loop_v5
    b      .Lpipe_tail_a_v5    // group A loaded but not yet consumed

.Lpipe_tail_b_v5:
    fmla   z2.s,  p1/m, z27.s, z27.s
    fmla   z16.s, p1/m, z28.s, z28.s
    fmla   z17.s, p1/m, z29.s, z29.s
    fmla   z18.s, p1/m, z30.s, z30.s
    b      .Lrem_chk_v5

.Lpipe_tail_a_v5:
    fmla   z2.s,  p1/m, z0.s,  z0.s
    fmla   z16.s, p1/m, z24.s, z24.s
    fmla   z17.s, p1/m, z25.s, z25.s
    fmla   z18.s, p1/m, z26.s, z26.s

.Lrem_chk_v5:
    cbz    x9, .Lred_done_v5

    // === Remainder: 0-3 trailing columns ====================================
.Lred_rem_v5:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s,   p1/m, z0.s, z0.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_rem_v5

.Lred_done_v5:
    fadd   z16.s, z16.s, z17.s
    fadd   z2.s,  z2.s,  z18.s
    fadd   z2.s,  z2.s,  z16.s

    // === inv_rms — identical to V2/V4 =======================================
    fmul   z2.s, p1/m, z2.s, z5.s

    dup    z3.s, z8.s[0]
    fadd   z2.s, p1/m, z2.s, z3.s
    sel    z2.s, p1, z2.s, z3.s

    frsqrte z4.s, z2.s
    fmul    z3.s, z4.s, z2.s
    frsqrts z3.s, z4.s, z3.s
    fmul    z4.s, p1/m, z4.s, z3.s

    // === Pass 2: normalize + scale — identical to V2/V4 =====================
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Lnrm_col_loop_v5:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x10]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_loop_v5

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lblock_loop_v5

.Lblock_done_v5:
    smstop sm

.Lrow_done_v5:
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
