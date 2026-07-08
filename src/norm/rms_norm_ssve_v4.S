// rms_norm_ssve_v4.S — Sprint 2b ablation V4
//
// Change vs V2 (the incumbent structure: FRSQRTE+NR, pre-computed 1/N):
//   Pass 1 (sum of squares) uses FOUR independent FMLA accumulators
//   (Z2, Z16, Z17, Z18), processing four columns per iteration, combined
//   with two FADDs at the end before the inv_rms step.
//
// Hypothesis (the memory-level-parallelism lever): V0-V3 accumulate into a
// single Z register, so every FMLA waits on the previous FMLA's result
// (~3-4 cycle latency chain) and the load stream can run at most one
// column ahead per chain step.  Four independent chains let the OOO core
// keep four column loads in flight behind independent FMAs.  This is
// distinct from V3's unroll, which kept ONE chain and only cut branch
// overhead (measured: no gain).
//
// Addressing: two pointers (X8, X12 = X8 + stride), each advanced by
// 2*stride per pair of loads — avoids the LD1W scalar+scalar form whose
// element-vs-byte index scaling caused the Sprint-2 debug-log bug #1.
//
// Remainder: N % 4 trailing columns fall into a single-accumulator loop
// (identical to V2's) accumulating into Z2.
//
// Inactive-lane safety: all four accumulators start as MOV #0 (all lanes)
// and FMLA is p1/m, so inactive lanes stay 0 through the FADD combine —
// the same invariant V1/V2 rely on before the SEL/FRSQRTE sequence.
//
// C signature (identical to V0-V3):
//   void rms_norm_ssve_v4(const float *a,     // x0
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

.global SYMBOL(rms_norm_ssve_v4)

.text
.align 4

SYMBOL(rms_norm_ssve_v4):

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
    fmov  s8,  s0              // save eps into callee-saved S8
    str   s0,  [sp, #64]       // dedicated eps stash, stored BEFORE smstart
                                // (streaming-mode entry clobbers D8 too, not
                                // just D9-D15 — memory survives smstart, no
                                // register does; verified empirically)

    cbz   x22, .Lrow_done_v4

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #64]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, eps, ...}

    fmov  s0, #1.0
    scvtf s1, x23              // s1 = float(N)
    fdiv  s0, s0, s1           // s0 = 1.0 / N  (one-time scalar divide)
    dup   z5.s, z0.s[0]        // z5 = {1/N, 1/N, ...}

    lsl   x13, x24, #1         // x13 = 2*stride_a (constant; no calls, so
                               // the scratch register survives the function)
    mov   x0, #0

    // -----------------------------------------------------------------------
    // Outer loop — one iteration per block of VL rows.
    // -----------------------------------------------------------------------
.Lblock_loop_v4:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done_v4

    // === Pass 1: sum of squares, 4 independent accumulator chains ==========
    mov    z2.s,  #0
    mov    z16.s, #0
    mov    z17.s, #0
    mov    z18.s, #0
    mov    x8,   x19
    mov    x9,   x23

    cmp    x9, #4
    b.lo   .Lred_rem_v4

    add    x12, x8, x24        // x12 = column pointer, one stride ahead

.Lred_col4_v4:
    ld1w   {z0.s},  p1/z, [x8]
    ld1w   {z24.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    ld1w   {z25.s}, p1/z, [x8]
    ld1w   {z26.s}, p1/z, [x12]
    add    x8,  x8,  x13
    add    x12, x12, x13
    fmla   z2.s,  p1/m, z0.s,  z0.s
    fmla   z16.s, p1/m, z24.s, z24.s
    fmla   z17.s, p1/m, z25.s, z25.s
    fmla   z18.s, p1/m, z26.s, z26.s
    sub    x9,  x9,  #4
    cmp    x9,  #4
    b.hs   .Lred_col4_v4

    cbz    x9, .Lred_done_v4

    // === Remainder: 0-3 trailing columns, single chain into Z2 =============
.Lred_rem_v4:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s,   p1/m, z0.s, z0.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_rem_v4

.Lred_done_v4:
    // Combine the four chains (inactive lanes are 0 in all four).
    fadd   z16.s, z16.s, z17.s
    fadd   z2.s,  z2.s,  z18.s
    fadd   z2.s,  z2.s,  z16.s

    // === Compute inv_rms — identical to V2 ==================================
    fmul   z2.s, p1/m, z2.s, z5.s     // z2 *= 1/N  (mean square)

    dup    z3.s, z8.s[0]               // z3 = {eps, eps, ...}
    fadd   z2.s, p1/m, z2.s, z3.s     // z2 += eps  (active lanes)
    sel    z2.s, p1, z2.s, z3.s       // z2[inactive] = eps (safe for FRSQRTE)

    frsqrte z4.s, z2.s                 // z4 ~= 1/sqrt(z2)
    fmul    z3.s, z4.s, z2.s           // z3 = z4*z2 (NR prep)
    frsqrts z3.s, z4.s, z3.s          // z3 = (3 - z4*z3) / 2
    fmul    z4.s, p1/m, z4.s, z3.s    // z4[active] = inv_rms

    // === Pass 2: normalize + scale — identical to V2 ========================
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Lnrm_col_loop_v4:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x10]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_loop_v4

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lblock_loop_v4

.Lblock_done_v4:
    smstop sm

.Lrow_done_v4:
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
