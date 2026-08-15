// rms_norm_ssve_v4.S — RMSNorm V4: four independent FMLA accumulator chains.
//
// V0-V3 accumulate into one Z register, so each FMLA waits ~3-4 cycles on the
// previous one and the load stream can only run one column ahead.  Four
// chains (z2/z16/z17/z18, four columns per iteration, combined by two FADDs)
// let the OOO core keep four column loads in flight.  Measured +27-32% at
// N >= 2048 — the single chain WAS the bottleneck, which is also why V3's
// unrolling bought nothing.
//
// Two pointers (x8, x12 = x8 + stride) each advance by 2*stride per pair, to
// avoid the LD1W scalar+scalar form and its element-vs-byte index scaling.
// N % 4 trailing columns fall into a single-accumulator remainder loop.
//
// Numerics: the four partial sums combine in tree order rather than the
// reference's sequential order, so V4/V5 are verified at kTolReassoc (2e-5),
// documented rather than silently widening the gate.  V6 removes the issue.
//
// C signature: identical to rms_norm_ssve (rms_norm_ssve.S).
//   (a, b, gamma, m, n, ld_a, ld_b) in x0-x6, eps in s0.

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
    stp  x19, x20, [sp, #-128]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    str  x25,      [sp, #48]
    str  d8,       [sp, #56]
    str  d9,       [sp, #64]
    str  d10,      [sp, #72]
    str  d11,      [sp, #80]
    str  d12,      [sp, #88]
    str  d13,      [sp, #96]
    str  d14,      [sp, #104]
    str  d15,      [sp, #112]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    lsl   x24, x5, #2          // stride_a in bytes
    lsl   x25, x6, #2          // stride_b in bytes
    fmov  s8,  s0              // save eps into callee-saved S8
    str   s0,  [sp, #120]       // dedicated eps stash, stored BEFORE smstart

    cbz   x22, .Lrow_done_v4

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #120]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, eps, ...}

    fmov  s0, #1.0
    scvtf s1, x23              // s1 = float(N)
    fdiv  s0, s0, s1           // s0 = 1.0 / N  (one-time scalar divide)
    dup   z5.s, z0.s[0]        // z5 = {1/N, 1/N, ...}

    lsl   x13, x24, #1         // x13 = 2*stride_a (constant; no calls, so
                               // the scratch register survives the function)
    mov   x0, #0

    // Outer loop — one iteration per block of VL rows.
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
    ldr  d9,       [sp, #64]
    ldr  d10,      [sp, #72]
    ldr  d11,      [sp, #80]
    ldr  d12,      [sp, #88]
    ldr  d13,      [sp, #96]
    ldr  d14,      [sp, #104]
    ldr  d15,      [sp, #112]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #128
    ret
