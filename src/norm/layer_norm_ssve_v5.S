// layer_norm_ssve_v5.S — LayerNorm SSVE Sprint 2c ablation V5
//
// Change vs V4: software-pipelined loads in both reduction passes.
// Within each pass a group of four columns (z0/z24/z25/z26) is preloaded
// before the steady-state loop; the loop issues the next four loads
// (z27/z28/z29/z30) while the FP compute for the current group executes.
// After the FP ops, the B group is moved into A positions via four MOV
// z.d instructions before the next iteration.  The expected throughput gain
// on an M4 out-of-order core is ~0 % because the hardware already reorders
// memory and FP operations.  The ablation is included for completeness.
// Pass 3 is unchanged from V2.
//
// C signature: same as layer_norm_ssve_v2.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v5)
.text
.align 4

SYM(layer_norm_ssve_v5):

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

    cbz   x23, .Ldone_v5

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #64]
    dup   z7.s, z0.s[0]

    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1
    dup   z4.s, z0.s[0]            // z4 = [1/N,...]

    lsl   x13, x25, #1             // x13 = 2 * stride_a (constant)
    mov   x0, #0

.Lblock_v5:
    whilelo p1.s, x0, x23
    b.none  .Lblkdone_v5

    // ── Pass 1: mean (4-chain, pipelined loads) ─────────────────────
    mov   z8.s,  #0
    mov   z16.s, #0
    mov   z17.s, #0
    mov   z18.s, #0

    mov   x8, x19
    mov   x9, x24

    cmp   x9, #4
    b.lo  .Lp1rem_v5

    add   x12, x8, x25             // ptr B = ptr A + one stride

    // Preload group A (cols 0–3)
    ld1w  {z0.s},  p1/z, [x8]
    ld1w  {z24.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z25.s}, p1/z, [x8]
    ld1w  {z26.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    sub   x9, x9, #4

    cmp   x9, #4
    b.lo  .Lp1tail_v5

    // Steady state: load group B while processing group A
.Lp1steady_v5:
    ld1w  {z27.s}, p1/z, [x8]
    ld1w  {z28.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z29.s}, p1/z, [x8]
    ld1w  {z30.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    fadd  z8.s,  p1/m, z8.s,  z0.s
    fadd  z16.s, p1/m, z16.s, z24.s
    fadd  z17.s, p1/m, z17.s, z25.s
    fadd  z18.s, p1/m, z18.s, z26.s
    sub   x9, x9, #4
    mov   z0.d,  z27.d
    mov   z24.d, z28.d
    mov   z25.d, z29.d
    mov   z26.d, z30.d
    cmp   x9, #4
    b.hs  .Lp1steady_v5

    // Flush last loaded group A
.Lp1tail_v5:
    fadd  z8.s,  p1/m, z8.s,  z0.s
    fadd  z16.s, p1/m, z16.s, z24.s
    fadd  z17.s, p1/m, z17.s, z25.s
    fadd  z18.s, p1/m, z18.s, z26.s

    // Remainder: 1–3 cols not covered by a full group
.Lp1rem_v5:
    cbz   x9, .Lp1done_v5
.Lp1remloop_v5:
    ld1w  {z0.s}, p1/z, [x8]
    fadd  z8.s, p1/m, z8.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp1remloop_v5

.Lp1done_v5:
    fadd  z16.s, z16.s, z17.s
    fadd  z8.s,  z8.s,  z18.s
    fadd  z8.s,  z8.s,  z16.s     // z8 = total mean sum
    fmul  z8.s, p1/m, z8.s, z4.s  // z8 = mean

    // ── Pass 2: variance (4-chain, pipelined loads) ─────────────────
    // z8 = mean (held constant through pass 2)
    mov   z9.s,  #0
    mov   z10.s, #0
    mov   z11.s, #0
    mov   z12.s, #0

    mov   x8, x19
    mov   x9, x24

    cmp   x9, #4
    b.lo  .Lp2rem_v5

    add   x12, x8, x25

    // Preload group A
    ld1w  {z0.s},  p1/z, [x8]
    ld1w  {z24.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z25.s}, p1/z, [x8]
    ld1w  {z26.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    sub   x9, x9, #4

    cmp   x9, #4
    b.lo  .Lp2tail_v5

.Lp2steady_v5:
    ld1w  {z27.s}, p1/z, [x8]
    ld1w  {z28.s}, p1/z, [x12]
    add   x8,  x8,  x13
    add   x12, x12, x13
    ld1w  {z29.s}, p1/z, [x8]
    ld1w  {z30.s}, p1/z, [x12]
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
    mov   z0.d,  z27.d
    mov   z24.d, z28.d
    mov   z25.d, z29.d
    mov   z26.d, z30.d
    cmp   x9, #4
    b.hs  .Lp2steady_v5

.Lp2tail_v5:
    fsub  z0.s,  p1/m, z0.s,  z8.s
    fsub  z24.s, p1/m, z24.s, z8.s
    fsub  z25.s, p1/m, z25.s, z8.s
    fsub  z26.s, p1/m, z26.s, z8.s
    fmla  z9.s,  p1/m, z0.s,  z0.s
    fmla  z10.s, p1/m, z24.s, z24.s
    fmla  z11.s, p1/m, z25.s, z25.s
    fmla  z12.s, p1/m, z26.s, z26.s

.Lp2rem_v5:
    cbz   x9, .Lp2done_v5
.Lp2remloop_v5:
    ld1w  {z0.s}, p1/z, [x8]
    fsub  z0.s, p1/m, z0.s, z8.s
    fmla  z9.s, p1/m, z0.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp2remloop_v5

.Lp2done_v5:
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

.Lp3_v5:
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
    b.ne  .Lp3_v5

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lblock_v5

.Lblkdone_v5:
    smstop sm

.Ldone_v5:
    ldr  d8,       [sp, #64]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
