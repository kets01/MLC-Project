// layer_norm_ssve_v6.S — LayerNorm SSVE Sprint 2c ablation V6
//
// Change vs V4: 4-row-block contiguity grouping.  When at least 4 VL-row
// blocks remain, all three passes (mean, variance, normalize) are run for
// that group of 4 blocks before moving to the next group.  Within each pass,
// each column j is touched by four back-to-back LD1W/ST1W instructions at
// [ptr], [ptr,#1,mul vl], [ptr,#2,mul vl], [ptr,#3,mul vl], producing a
// 4×VL-row contiguous burst that the DRAM prefetcher can stream efficiently.
// Four independent FP accumulators (one per block) in the reduction passes
// also expose ILP.  Residency across passes is improved for working sets
// that fit in L2 (all three passes on one 4-block chunk complete before
// touching a different chunk).  For m not divisible by 4×VL, the remaining
// 1–3 blocks are processed by the V2-style predicated single-block tail.
// C signature: same as layer_norm_ssve_v2.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v6)
.text
.align 4

// Register map (constant post-prologue):
//   x19 = a-chunk block-0 base    x20 = b-chunk block-0 base
//   x21 = gamma                   x22 = beta
//   x23 = m (rows)                x24 = n (cols)
//   x25 = stride_a (B)            x26 = stride_b (B)
//   d8  = eps                     x14 = 4*VL (elements, set after SMSTART)
//   z4  = [1/N,...]               z7  = [eps,...]
//
// Group path (p0 = all-true, 4 blocks per iteration):
//   z8..z11  = mean accumulators/values for blocks 0..3
//   z12..z15 = variance accumulators for blocks 0..3
//   z16      = NR scratch
//   z17..z20 = inv_std for blocks 0..3
//   z0..z3   = data   z5 = gamma broadcast   z6 = beta broadcast
//
// Tail path (p1 = whilelo, V2-style single block):
//   z8 = mean acc/value    z9 = var acc    z5 = inv_std

SYM(layer_norm_ssve_v6):

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

    cbz   x23, .Ldone_v6

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #72]            // eps reloaded from the dedicated stack slot
    dup   z7.s, z0.s[0]

    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1
    dup   z4.s, z0.s[0]            // z4 = [1/N,...]

    cntw  x14
    lsl   x14, x14, #2             // x14 = 4*VL (elements)
    mov   x0, #0

    // ──────────────────────────────────────────────────────────────
    // Group loop: process 4 VL-row blocks per iteration
    // ──────────────────────────────────────────────────────────────
.Lgroup_v6:
    add   x15, x0, x14             // x15 = x0 + 4*VL
    cmp   x15, x23
    b.hi  .Ltail_v6                // fewer than 4 full blocks remain

    // ── Group pass 1: mean (p0, 4 accumulators) ──────────────────
    mov   z8.s,  #0
    mov   z9.s,  #0
    mov   z10.s, #0
    mov   z11.s, #0

    mov   x8, x19
    mov   x9, x24

.Lgp1col_v6:
    ld1w  {z0.s}, p0/z, [x8]
    ld1w  {z1.s}, p0/z, [x8, #1, mul vl]
    ld1w  {z2.s}, p0/z, [x8, #2, mul vl]
    ld1w  {z3.s}, p0/z, [x8, #3, mul vl]
    fadd  z8.s,  p0/m, z8.s,  z0.s
    fadd  z9.s,  p0/m, z9.s,  z1.s
    fadd  z10.s, p0/m, z10.s, z2.s
    fadd  z11.s, p0/m, z11.s, z3.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lgp1col_v6

    fmul  z8.s,  p0/m, z8.s,  z4.s // mean_0
    fmul  z9.s,  p0/m, z9.s,  z4.s // mean_1
    fmul  z10.s, p0/m, z10.s, z4.s // mean_2
    fmul  z11.s, p0/m, z11.s, z4.s // mean_3

    // ── Group pass 2: variance + inv_std ─────────────────────────
    mov   z12.s, #0
    mov   z13.s, #0
    mov   z14.s, #0
    mov   z15.s, #0

    mov   x8, x19
    mov   x9, x24

.Lgp2col_v6:
    ld1w  {z0.s}, p0/z, [x8]
    ld1w  {z1.s}, p0/z, [x8, #1, mul vl]
    ld1w  {z2.s}, p0/z, [x8, #2, mul vl]
    ld1w  {z3.s}, p0/z, [x8, #3, mul vl]
    fsub  z0.s, p0/m, z0.s, z8.s
    fsub  z1.s, p0/m, z1.s, z9.s
    fsub  z2.s, p0/m, z2.s, z10.s
    fsub  z3.s, p0/m, z3.s, z11.s
    fmla  z12.s, p0/m, z0.s, z0.s
    fmla  z13.s, p0/m, z1.s, z1.s
    fmla  z14.s, p0/m, z2.s, z2.s
    fmla  z15.s, p0/m, z3.s, z3.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lgp2col_v6

    // var + eps for each block
    fmul  z12.s, p0/m, z12.s, z4.s
    fadd  z12.s, p0/m, z12.s, z7.s
    fmul  z13.s, p0/m, z13.s, z4.s
    fadd  z13.s, p0/m, z13.s, z7.s
    fmul  z14.s, p0/m, z14.s, z4.s
    fadd  z14.s, p0/m, z14.s, z7.s
    fmul  z15.s, p0/m, z15.s, z4.s
    fadd  z15.s, p0/m, z15.s, z7.s

    // FRSQRTE+NR per block (no SEL: p0 has no inactive lanes)
    frsqrte z17.s, z12.s
    fmul    z16.s, z17.s, z12.s
    frsqrts z16.s, z17.s, z16.s
    fmul    z17.s, z17.s, z16.s    // inv_std_0

    frsqrte z18.s, z13.s
    fmul    z16.s, z18.s, z13.s
    frsqrts z16.s, z18.s, z16.s
    fmul    z18.s, z18.s, z16.s    // inv_std_1

    frsqrte z19.s, z14.s
    fmul    z16.s, z19.s, z14.s
    frsqrts z16.s, z19.s, z16.s
    fmul    z19.s, z19.s, z16.s    // inv_std_2

    frsqrte z20.s, z15.s
    fmul    z16.s, z20.s, z15.s
    frsqrts z16.s, z20.s, z16.s
    fmul    z20.s, z20.s, z16.s    // inv_std_3

    // ── Group pass 3: normalize + scale + shift ───────────────────
    mov   x8,  x19
    mov   x9,  x20
    mov   x10, x21
    mov   x11, x22
    mov   x12, x24

.Lgp3col_v6:
    ld1w  {z0.s}, p0/z, [x8]
    ld1w  {z1.s}, p0/z, [x8, #1, mul vl]
    ld1w  {z2.s}, p0/z, [x8, #2, mul vl]
    ld1w  {z3.s}, p0/z, [x8, #3, mul vl]
    ld1rw {z5.s}, p0/z, [x10]
    ld1rw {z6.s}, p0/z, [x11]
    fsub  z0.s, p0/m, z0.s, z8.s
    fsub  z1.s, p0/m, z1.s, z9.s
    fsub  z2.s, p0/m, z2.s, z10.s
    fsub  z3.s, p0/m, z3.s, z11.s
    fmul  z0.s, p0/m, z0.s, z17.s
    fmul  z1.s, p0/m, z1.s, z18.s
    fmul  z2.s, p0/m, z2.s, z19.s
    fmul  z3.s, p0/m, z3.s, z20.s
    fmul  z0.s, p0/m, z0.s, z5.s
    fmul  z1.s, p0/m, z1.s, z5.s
    fmul  z2.s, p0/m, z2.s, z5.s
    fmul  z3.s, p0/m, z3.s, z5.s
    fadd  z0.s, p0/m, z0.s, z6.s
    fadd  z1.s, p0/m, z1.s, z6.s
    fadd  z2.s, p0/m, z2.s, z6.s
    fadd  z3.s, p0/m, z3.s, z6.s
    st1w  {z0.s}, p0, [x9]
    st1w  {z1.s}, p0, [x9, #1, mul vl]
    st1w  {z2.s}, p0, [x9, #2, mul vl]
    st1w  {z3.s}, p0, [x9, #3, mul vl]
    add   x8,  x8,  x25
    add   x9,  x9,  x26
    add   x10, x10, #4
    add   x11, x11, #4
    subs  x12, x12, #1
    b.ne  .Lgp3col_v6

    addvl x19, x19, #4
    addvl x20, x20, #4
    add   x0, x0, x14
    b     .Lgroup_v6

    // ──────────────────────────────────────────────────────────────
    // Tail: 1–3 remaining blocks, predicated (V2-style, p1)
    // ──────────────────────────────────────────────────────────────
.Ltail_v6:
    whilelo p1.s, x0, x23
    b.none  .Ldone_v6

    // Pass 1: mean
    mov   z8.s, #0
    mov   x8, x19
    mov   x9, x24

.Ltp1_v6:
    ld1w  {z0.s}, p1/z, [x8]
    fadd  z8.s, p1/m, z8.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Ltp1_v6

    fmul  z8.s, p1/m, z8.s, z4.s   // mean

    // Pass 2: variance + inv_std
    mov   z9.s, #0
    mov   x8, x19
    mov   x9, x24

.Ltp2_v6:
    ld1w  {z0.s}, p1/z, [x8]
    fsub  z0.s, p1/m, z0.s, z8.s
    fmla  z9.s, p1/m, z0.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Ltp2_v6

    fmul  z9.s, p1/m, z9.s, z4.s
    fadd  z9.s, p1/m, z9.s, z7.s

    sel     z9.s, p1, z9.s, z7.s
    frsqrte z5.s, z9.s
    fmul    z3.s, z5.s, z9.s
    frsqrts z3.s, z5.s, z3.s
    fmul    z5.s, p1/m, z5.s, z3.s // z5 = inv_std

    // Pass 3: normalize + scale + shift
    mov   x8,  x19
    mov   x9,  x20
    mov   x10, x21
    mov   x11, x22
    mov   x12, x24

.Ltp3_v6:
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
    b.ne  .Ltp3_v6

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Ltail_v6

.Ldone_v6:
    smstop sm

    ldr  d8,       [sp, #64]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
