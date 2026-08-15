// rms_norm_ssve_v3.S — RMSNorm V3: V2 with both column loops unrolled 2x.
//
// A peel handles odd N so the main loop always sees pairs.  Measured gain
// ~0: the out-of-order core already overlaps sequential column loads, and
// unrolling does not break the single accumulator chain (V4 does, and wins).
//
// Addressing note: LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2] scales Xm as an
// ELEMENT count, so the second column uses x5/x6 (elements) and not the
// byte strides in x24/x25.
//
// C signature: identical to rms_norm_ssve (rms_norm_ssve.S).
//   (a, b, gamma, m, n, ld_a, ld_b) in x0-x6, eps in s0.

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(rms_norm_ssve_v3)

.text
.align 4

SYMBOL(rms_norm_ssve_v3):

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
    lsl   x24, x5, #2
    lsl   x25, x6, #2
    fmov  s8,  s0
    str   s0,  [sp, #120]   // dedicated eps stash, stored BEFORE smstart

    cbz   x22, .Lrow_done_v3

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #120]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, eps, ...}
    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1           // s0 = 1.0 / N
    dup   z5.s, z0.s[0]        // z5 = {1/N, ...}

    mov    x0, #0

    // Outer loop — one iteration per block of VL rows.
.Lblock_loop_v3:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done_v3

    // === Pass 1: sum of squares, 2x unrolled ================================
    mov    z2.s, #0
    mov    x8,   x19
    mov    x9,   x23       // x9 = column countdown = N

    // Peel one column when N is odd so the main loop always sees even count.
    tbz    x9, #0, .Lpass1_main_v3
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s,   p1/m, z0.s, z0.s
    add    x8, x8, x24
    subs   x9, x9, #1
    cbz    x9, .Lpass1_done_v3   // N was 1

.Lpass1_main_v3:               // x9 is even and >= 2
    ld1w   {z0.s}, p1/z, [x8]
    ld1w   {z6.s}, p1/z, [x8, x5, lsl #2]  // col c+1: x8 + ld_a_elems*4
    fmla   z2.s, p1/m, z0.s, z0.s
    fmla   z2.s, p1/m, z6.s, z6.s
    add    x8, x8, x24, lsl #1      // advance 2 columns
    subs   x9, x9, #2
    b.ne   .Lpass1_main_v3

.Lpass1_done_v3:

    // === Compute inv_rms (same as V2) =======================================
    fmul   z2.s, p1/m, z2.s, z5.s

    dup    z3.s, z8.s[0]
    fadd   z2.s, p1/m, z2.s, z3.s
    sel    z2.s, p1, z2.s, z3.s

    frsqrte z4.s, z2.s
    fmul    z3.s, z4.s, z2.s
    frsqrts z3.s, z4.s, z3.s
    fmul    z4.s, p1/m, z4.s, z3.s

    // === Pass 2: normalize + scale, 2x unrolled =============================
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23        // x11 = column countdown = N

    // Peel one column when N is odd.
    tbz    x11, #0, .Lpass2_main_v3
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x10]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    cbz    x11, .Lpass2_done_v3    // N was 1

.Lpass2_main_v3:               // x11 is even and >= 2
    ld1w   {z0.s}, p1/z, [x8]
    ld1w   {z6.s}, p1/z, [x8, x5, lsl #2]   // A col c+1: x8 + ld_a_elems*4
    ld1rw  {z1.s}, p0/z, [x10]       // gamma[c]
    ld1rw  {z7.s}, p0/z, [x10, #4]   // gamma[c+1]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z6.s, p1/m, z6.s, z7.s
    fmul   z0.s, p1/m, z0.s, z4.s
    fmul   z6.s, p1/m, z6.s, z4.s
    st1w   {z0.s}, p1, [x9]
    st1w   {z6.s}, p1, [x9, x6, lsl #2]     // B col c+1: x9 + ld_b_elems*4
    add    x8,  x8,  x24, lsl #1
    add    x9,  x9,  x25, lsl #1
    add    x10, x10, #8               // gamma += 2 floats
    subs   x11, x11, #2
    b.ne   .Lpass2_main_v3

.Lpass2_done_v3:

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lblock_loop_v3

.Lblock_done_v3:
    smstop sm

.Lrow_done_v3:
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
