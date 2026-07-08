// layer_norm_ssve_v4.S — LayerNorm SSVE Sprint 2c ablation V4
//
// Change vs V2: four independent accumulator chains in BOTH reduction passes
// (pass 1 = mean, pass 2 = variance) to expose instruction-level parallelism.
// A 4-column main loop issues loads for columns j, j+1, j+2, j+3 in each
// iteration and dispatches them to four separate FADD/FMLA chains (z8/z16/z17/z18
// for mean, z9/z10/z11/z12 for variance).  After each pass the four partial
// sums are folded into one register before the final multiply.
// Pass 3 (normalize + scale + shift) is unchanged from V2.
//
// C signature: same as layer_norm_ssve_v2.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v4)
.text
.align 4

// Register map (post-prologue, constant through all passes):
//   x19 = a chunk base   x20 = b chunk base
//   x21 = gamma          x22 = beta
//   x23 = m              x24 = n
//   x25 = stride_a(B)    x26 = stride_b(B)
//   d8  = eps            x0  = row counter
//   z4  = [1/N,...]      z7  = [eps,...]
//
// Per pass (volatile):
//   x8  = walking ptr A          x12 = walking ptr B (A + stride)
//   x9  = column counter         x13 = 2 * stride_a
//   Pass-1 accumulators : z8(A) z16(B) z17(C) z18(D) → z8 = mean
//   Pass-2 accumulators : z9(A) z10(B) z11(C) z12(D) → z9 = var+eps
//   z5  = inv_std   z3 = NR scratch
//   Pass-3 data / gamma / beta : z0 / z1 / z2

SYM(layer_norm_ssve_v4):

    stp  x19, x20, [sp, #-80]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    str  d8,       [sp, #64]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, x5
    lsl   x25, x6, #2
    lsl   x26, x7, #2
    fmov  s8, s0
    str   s0, [sp, #72]            // dedicated eps stash, stored BEFORE smstart
                                    // (streaming-mode entry clobbers D8 too, not
                                    // just D9-D15 — memory survives smstart, no
                                    // register does; verified empirically)

    cbz   x23, .Ldone_v4

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #72]            // eps reloaded from the dedicated stack slot
    dup   z7.s, z0.s[0]

    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1
    dup   z4.s, z0.s[0]            // z4 = [1/N,...]

    lsl   x13, x25, #1             // x13 = 2 * stride_a (constant)
    mov   x0, #0

.Lblock_v4:
    whilelo p1.s, x0, x23
    b.none  .Lblkdone_v4

    // ── Pass 1: mean (4-chain) ─────────────────────────────────────
    mov   z8.s,  #0
    mov   z16.s, #0
    mov   z17.s, #0
    mov   z18.s, #0

    mov   x8, x19
    mov   x9, x24

    cmp   x9, #4
    b.lo  .Lp1rem_v4

    add   x12, x8, x25             // ptr B = ptr A + one stride

.Lp1col4_v4:
    ld1w  {z0.s},  p1/z, [x8]
    ld1w  {z24.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z25.s}, p1/z, [x8]
    ld1w  {z26.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    fadd  z8.s,  p1/m, z8.s,  z0.s
    fadd  z16.s, p1/m, z16.s, z24.s
    fadd  z17.s, p1/m, z17.s, z25.s
    fadd  z18.s, p1/m, z18.s, z26.s
    sub   x9, x9, #4
    cmp   x9, #4
    b.hs  .Lp1col4_v4

    cbz   x9, .Lp1done_v4

.Lp1rem_v4:
    ld1w  {z0.s}, p1/z, [x8]
    fadd  z8.s, p1/m, z8.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp1rem_v4

.Lp1done_v4:
    fadd  z16.s, z16.s, z17.s
    fadd  z8.s,  z8.s,  z18.s
    fadd  z8.s,  z8.s,  z16.s     // z8 = total sum
    fmul  z8.s, p1/m, z8.s, z4.s  // z8 = mean

    // ── Pass 2: variance (4-chain) ────────────────────────────────
    mov   z9.s,  #0
    mov   z10.s, #0
    mov   z11.s, #0
    mov   z12.s, #0

    mov   x8, x19
    mov   x9, x24

    cmp   x9, #4
    b.lo  .Lp2rem_v4

    add   x12, x8, x25

.Lp2col4_v4:
    ld1w  {z0.s},  p1/z, [x8]
    ld1w  {z24.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z25.s}, p1/z, [x8]
    ld1w  {z26.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    fsub  z0.s,  p1/m, z0.s,  z8.s
    fsub  z24.s, p1/m, z24.s, z8.s
    fsub  z25.s, p1/m, z25.s, z8.s
    fsub  z26.s, p1/m, z26.s, z8.s
    fmla  z9.s,  p1/m, z0.s,  z0.s
    fmla  z10.s, p1/m, z24.s, z24.s
    fmla  z11.s, p1/m, z25.s, z25.s
    fmla  z12.s, p1/m, z26.s, z26.s
    sub   x9, x9, #4
    cmp   x9, #4
    b.hs  .Lp2col4_v4

    cbz   x9, .Lp2done_v4

.Lp2rem_v4:
    ld1w  {z0.s}, p1/z, [x8]
    fsub  z0.s, p1/m, z0.s, z8.s
    fmla  z9.s, p1/m, z0.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp2rem_v4

.Lp2done_v4:
    fadd  z10.s, z10.s, z11.s
    fadd  z9.s,  z9.s,  z12.s
    fadd  z9.s,  z9.s,  z10.s     // z9 = total sq_sum
    fmul  z9.s, p1/m, z9.s, z4.s  // z9 = variance
    fadd  z9.s, p1/m, z9.s, z7.s  // var + eps

    sel     z9.s, p1, z9.s, z7.s
    frsqrte z5.s, z9.s
    fmul    z3.s, z5.s, z9.s
    frsqrts z3.s, z5.s, z3.s
    fmul    z5.s, p1/m, z5.s, z3.s // z5 = inv_std

    // ── Pass 3: normalize + scale + shift ─────────────────────────
    mov   x8,  x19
    mov   x9,  x20
    mov   x10, x21
    mov   x11, x22
    mov   x12, x24

.Lp3_v4:
    ld1w  {z0.s}, p1/z, [x8]
    ld1rw {z1.s}, p1/z, [x10]
    ld1rw {z2.s}, p1/z, [x11]
    fsub  z0.s, p1/m, z0.s, z8.s
    fmul  z0.s, p1/m, z0.s, z5.s
    fmul  z0.s, p1/m, z0.s, z1.s
    fadd  z0.s, p1/m, z0.s, z2.s
    st1w  {z0.s}, p1, [x9]
    add   x8,  x8,  x25
    add   x9,  x9,  x26
    add   x10, x10, #4
    add   x11, x11, #4
    subs  x12, x12, #1
    b.ne  .Lp3_v4

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lblock_v4

.Lblkdone_v4:
    smstop sm

.Ldone_v4:
    ldr  d8,       [sp, #64]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
