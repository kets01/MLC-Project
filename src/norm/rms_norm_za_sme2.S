// rms_norm_za_sme2.S — ZA residency rebuilt on SME2 multi-vector MOVA.
//
// Sprint 3 left ZA alone on the reasoning that even a 2x mova speed-up would
// only reach a tie with V6.  That rested on an untested factor: mova had never
// been characterised as ISSUE-bound vs ZA-PORT-bound.  Measured directly (no
// memory traffic, 16 vectors either way): single-vector 3.88 G vec/s,
// 4-vector 15.50 = exactly 4.00x.  It is issue-bound and the fold is 4:1, so
// the skip's own arithmetic did not hold and the kernel was built.
//
// Result: +62% over the Sprint-3 ZA kernel, and still 37% behind V7.  The
// residual cause is different from the original diagnosis — these kernels
// touch ONE SVL-row block per iteration (64 B per column) against V6/V7's
// four (256 B), so what binds is access density, not mova.
//
// vs rms_norm_za: four columns are staged per chunk, folding ZA traffic 4:1
// in both directions (2 MOVAs per 4 columns instead of 8).  Loads and stores
// stay single-vector — columns are strided in column-major layout.
//
// ADDRESSING, verified rather than assumed: the group form `za.s[w8,0,vgx4]`
// takes slice w8 from EACH of the four 32-bit tiles, so a group is 4 vectors,
// the cursor runs 0..SVL_S-1 and steps by ONE, not four.  (Stepping by four
// wraps silently at N > SVL; the N=17/31 tests pin it.)  A standalone probe
// wrote two groups, clobbered every Z register, read both back: 0/128 wrong.
//
// REGISTER TRAP this kernel is designed around: the multi-vector form requires
// its slice cursor in W8-W11, and writing W8 zeroes the upper half of X8 — the
// week-6 GEMM defect in miniature.  Pointers therefore live in x12-x15 and x8
// is never used as one.
//
// Numerics: the four columns of a chunk accumulate into ONE accumulator with
// four sequential FMLAs, so summation stays in strict column order.
// Capacity: 16 groups x 4 vectors = 64, so the resident path covers
// N <= 4*SVL; N % 4 pads the final chunk with zeroed vectors rather than
// reading out of bounds, and the padded slices are never read back.
//
// C signature: identical to rms_norm_ssve (rms_norm_ssve.S).
//   (a, b, gamma, m, n, ld_a, ld_b) in x0-x6, eps in s0.

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme2

.global SYMBOL(rms_norm_za_sme2)

.text
.align 4

// Register map (post-prologue):
//   x19 a-block base   x20 b-block base   x21 gamma   x22 M   x23 N
//   x24 stride_a (B)   x25 stride_b (B)   x0  row index
//   x12 walking input ptr        x13 walking output ptr
//   x14 walking gamma ptr        x15 column counter
//   x9  SVL              x10 4*SVL       w8  ZA slice cursor (W8-W11 only)
//   z0-z3 staging quad (4-aligned group)   z4 sum-of-squares accumulator
//   z5 {1/N}   z6 inv_rms   z7 gamma bcast   z8 {eps}   z9 NR scratch
//   p0 all-true            p1 row predicate

SYMBOL(rms_norm_za_sme2):

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
    str   s0,  [sp, #120]        // eps stash, BEFORE smstart

    cbz   x22, .Lrow_done_zs

    smstart                      // bare: PSTATE.SM *and* PSTATE.ZA
    ptrue  p0.s, all

    ldr    s0, [sp, #120]
    dup    z8.s, z0.s[0]         // z8 = {eps,...}
    fmov   s0, #1.0
    scvtf  s1, x23
    fdiv   s0, s0, s1
    dup    z5.s, z0.s[0]         // z5 = {1/N,...}

    cntw   x9
    lsl    x10, x9, #2           // 4*SVL = residency capacity in columns
    mov    x0, #0

    cmp    x23, x10
    b.hi   .Lfallback_zs

// ===========================================================================
// ZA-resident fast path (N <= 4*SVL): 1R + 1W, multi-vector MOVA staging.
// ===========================================================================
.Lza_block_zs:
    whilelo p1.s, x0, x22
    b.none  .Ldone_zs

    // ---- pass 1: read x once, accumulate, stage 4 columns per MOVA -------
    mov    z4.s, #0
    mov    x12, x19
    mov    x15, x23
    mov    w8,  #0

.Lchunk_zs:
    cmp    x15, #4
    b.lt   .Lchunk_rem_zs

    ld1w   {z0.s}, p1/z, [x12]
    add    x12, x12, x24
    ld1w   {z1.s}, p1/z, [x12]
    add    x12, x12, x24
    ld1w   {z2.s}, p1/z, [x12]
    add    x12, x12, x24
    ld1w   {z3.s}, p1/z, [x12]
    add    x12, x12, x24
    fmla   z4.s, p1/m, z0.s, z0.s     // sequential: reference summation order
    fmla   z4.s, p1/m, z1.s, z1.s
    fmla   z4.s, p1/m, z2.s, z2.s
    fmla   z4.s, p1/m, z3.s, z3.s
    mova   za.s[w8, 0, vgx4], { z0.s - z3.s }
    add    w8,  w8,  #1
    sub    x15, x15, #4
    b      .Lchunk_zs

    // 1-3 trailing columns: zero-pad the quad, never read past column N-1.
.Lchunk_rem_zs:
    cbz    x15, .Lpass2_zs
    mov    z0.s, #0
    mov    z1.s, #0
    mov    z2.s, #0
    mov    z3.s, #0

    ld1w   {z0.s}, p1/z, [x12]
    fmla   z4.s, p1/m, z0.s, z0.s
    subs   x15, x15, #1
    b.eq   .Lstage_rem_zs
    add    x12, x12, x24
    ld1w   {z1.s}, p1/z, [x12]
    fmla   z4.s, p1/m, z1.s, z1.s
    subs   x15, x15, #1
    b.eq   .Lstage_rem_zs
    add    x12, x12, x24
    ld1w   {z2.s}, p1/z, [x12]
    fmla   z4.s, p1/m, z2.s, z2.s

.Lstage_rem_zs:
    mova   za.s[w8, 0, vgx4], { z0.s - z3.s }

    // ---- inv_rms = 1/sqrt(sumsq/N + eps) --------------------------------
.Lpass2_zs:
    fmul    z4.s, p1/m, z4.s, z5.s
    fadd    z4.s, p1/m, z4.s, z8.s
    sel     z4.s, p1, z4.s, z8.s      // inactive lanes -> eps, never rsqrt(0)
    frsqrte z6.s, z4.s
    fmul    z9.s, z6.s, z4.s
    frsqrts z9.s, z6.s, z9.s
    fmul    z6.s, z6.s, z9.s          // z6 = inv_rms

    // ---- pass 2: read x back from ZA, scale, write once ------------------
    mov    x13, x20
    mov    x14, x21
    mov    x15, x23
    mov    w8,  #0

.Lchunk2_zs:
    cmp    x15, #4
    b.lt   .Lchunk2_rem_zs

    mova   { z0.s - z3.s }, za.s[w8, 0, vgx4]

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z0.s, p1/m, z0.s, z7.s
    fmul   z0.s, p1/m, z0.s, z6.s
    st1w   {z0.s}, p1, [x13]
    add    x13, x13, x25
    add    x14, x14, #4

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z1.s, p1/m, z1.s, z7.s
    fmul   z1.s, p1/m, z1.s, z6.s
    st1w   {z1.s}, p1, [x13]
    add    x13, x13, x25
    add    x14, x14, #4

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z2.s, p1/m, z2.s, z7.s
    fmul   z2.s, p1/m, z2.s, z6.s
    st1w   {z2.s}, p1, [x13]
    add    x13, x13, x25
    add    x14, x14, #4

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z3.s, p1/m, z3.s, z7.s
    fmul   z3.s, p1/m, z3.s, z6.s
    st1w   {z3.s}, p1, [x13]
    add    x13, x13, x25
    add    x14, x14, #4

    add    w8,  w8,  #1
    sub    x15, x15, #4
    b      .Lchunk2_zs

.Lchunk2_rem_zs:
    cbz    x15, .Lblock_next_zs
    mova   { z0.s - z3.s }, za.s[w8, 0, vgx4]

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z0.s, p1/m, z0.s, z7.s
    fmul   z0.s, p1/m, z0.s, z6.s
    st1w   {z0.s}, p1, [x13]
    subs   x15, x15, #1
    b.eq   .Lblock_next_zs
    add    x13, x13, x25
    add    x14, x14, #4

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z1.s, p1/m, z1.s, z7.s
    fmul   z1.s, p1/m, z1.s, z6.s
    st1w   {z1.s}, p1, [x13]
    subs   x15, x15, #1
    b.eq   .Lblock_next_zs
    add    x13, x13, x25
    add    x14, x14, #4

    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z2.s, p1/m, z2.s, z7.s
    fmul   z2.s, p1/m, z2.s, z6.s
    st1w   {z2.s}, p1, [x13]

.Lblock_next_zs:
    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lza_block_zs

// ===========================================================================
// Fallback (N > 4*SVL): streaming two-pass, no ZA.
// ===========================================================================
.Lfallback_zs:
    whilelo p1.s, x0, x22
    b.none  .Ldone_zs

    mov    z4.s, #0
    mov    x12, x19
    mov    x15, x23
.Lfb_red_zs:
    ld1w   {z0.s}, p1/z, [x12]
    fmla   z4.s, p1/m, z0.s, z0.s
    add    x12, x12, x24
    subs   x15, x15, #1
    b.ne   .Lfb_red_zs

    fmul    z4.s, p1/m, z4.s, z5.s
    fadd    z4.s, p1/m, z4.s, z8.s
    sel     z4.s, p1, z4.s, z8.s
    frsqrte z6.s, z4.s
    fmul    z9.s, z6.s, z4.s
    frsqrts z9.s, z6.s, z9.s
    fmul    z6.s, z6.s, z9.s

    mov    x12, x19
    mov    x13, x20
    mov    x14, x21
    mov    x15, x23
.Lfb_nrm_zs:
    ld1w   {z0.s}, p1/z, [x12]
    ld1rw  {z7.s}, p0/z, [x14]
    fmul   z0.s, p1/m, z0.s, z7.s
    fmul   z0.s, p1/m, z0.s, z6.s
    st1w   {z0.s}, p1, [x13]
    add    x12, x12, x24
    add    x13, x13, x25
    add    x14, x14, #4
    subs   x15, x15, #1
    b.ne   .Lfb_nrm_zs

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lfallback_zs

.Ldone_zs:
    smstop

.Lrow_done_zs:
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
