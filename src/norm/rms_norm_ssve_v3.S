// rms_norm_ssve_v3.S — Sprint 2 ablation V3
//
// Change vs V2:
//   Both inner column loops (pass 1 and pass 2) are unrolled by 2.
//   Each iteration loads two consecutive columns, letting the out-of-order
//   core overlap two independent memory streams and halving branch overhead.
//
//   When N is odd, a single-column peel executes first so the main loop
//   always processes pairs.
//
// Additional registers vs V2:
//   Z6 — second column of A (pass 1 + pass 2)
//   Z7 — second gamma broadcast (pass 2)
//
// Addressing for the second column:
//   LD1W  {Z6.S}, P1/Z, [X8, X24]   — load A col c+1  (byte offset x24 = ld_a_bytes)
//   LD1RW {Z7.S}, P0/Z, [X10, #4]   — broadcast gamma[c+1] (#4 = 1 float = 4 bytes)
//   ST1W  {Z6.S}, P1,   [X9,  X25]  — store B col c+1  (byte offset x25 = ld_b_bytes)
//
// Pointer advance for pairs:
//   ADD Xd, Xn, Xm, LSL #1 — adds 2*Xm in one instruction.
//
// C signature (identical to V0/V1/V2):
//   void rms_norm_ssve_v3(const float *a,     // x0
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

.global SYMBOL(rms_norm_ssve_v3)

.text
.align 4

SYMBOL(rms_norm_ssve_v3):

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
    str   s0,  [sp, #64]   // dedicated eps stash, stored BEFORE smstart
                            // (streaming-mode entry clobbers D8 too, not just
                            // D9-D15 — memory survives smstart, no register
                            // does; verified empirically)

    cbz   x22, .Lrow_done_v3

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #64]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, eps, ...}
    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1           // s0 = 1.0 / N
    dup   z5.s, z0.s[0]        // z5 = {1/N, ...}

    mov    x0, #0

    // -----------------------------------------------------------------------
    // Outer loop — one iteration per block of VL rows.
    // -----------------------------------------------------------------------
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
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
