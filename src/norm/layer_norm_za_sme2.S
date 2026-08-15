// layer_norm_za_sme2.S — LayerNorm ZA residency on SME2 multi-vector MOVA.
//
// The counterpart of rms_norm_za_sme2, built for the same reason Sprint 3
// built layer_norm_za rather than inferring from RMSNorm: the two norms stage
// a different amount through ZA, so the 4:1 fold has different work to remove.
//   RMSNorm  : 2 movas/element (1 store in pass 1, 1 load in pass 2)
//   LayerNorm: 3 movas/element (1 store, then loads for variance and normalize)
// LayerNorm pays a 50% larger mova tax, so it should gain MORE from the fold.
// Measured +114% against RMSNorm's +62% — and still 24% behind V7.
//
// vs layer_norm_za: four columns staged per chunk, so each of the three passes
// issues ONE multi-vector MOVA per four columns instead of four.
//
// ADDRESSING (verified, see rms_norm_za_sme2.S): `za.s[w8,0,vgx4]` takes slice
// w8 from each of the four tiles, so a group is 4 vectors, the cursor steps by
// ONE, and capacity is 16 groups x 4 = 4*SVL columns.  Stepping by four wraps
// silently at N > SVL — found by test, pinned by the N=17/31 cases.
//
// REGISTER NOTE: the multi-vector form needs its cursor in W8-W11, and writing
// W8 zeroes the upper half of X8, so x8 is never used as a pointer here.
//
// Numerics: within a chunk the four columns accumulate into ONE accumulator
// with sequential FADD/FMLA, so summation stays in strict column order.
//
// C signature: identical to layer_norm_ssve (layer_norm_ssve.S).
//   (a, b, gamma, beta, m, n, ld_a, ld_b) in x0-x7, eps in s0.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.arch armv9-a+sme2

.global SYM(layer_norm_za_sme2)

.text
.align 4

// Register map (post-prologue):
//   x19 a base   x20 b base   x21 gamma   x22 beta   x23 M   x24 N
//   x25 stride_a (B)          x26 stride_b (B)       x0  row index
//   x12 input ptr   x13 output ptr   x14 gamma ptr   x15 beta ptr
//   x16 column counter    x9 SVL    x10 4*SVL    w8 ZA slice cursor
//   z0-z3 staging quad    z4 mean acc/value   z5 {1/N}   z6 variance acc
//   z7 inv_std   z8 {eps}   z9 NR scratch   z10 gamma bcast   z11 beta bcast
//   p0 all-true           p1 row predicate

SYM(layer_norm_za_sme2):

    stp  x19, x20, [sp, #-144]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    str  d8,       [sp, #64]
    str  d9,       [sp, #72]
    str  d10,      [sp, #80]
    str  d11,      [sp, #88]
    str  d12,      [sp, #96]
    str  d13,      [sp, #104]
    str  d14,      [sp, #112]
    str  d15,      [sp, #120]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, x5
    lsl   x25, x6, #2
    lsl   x26, x7, #2
    str   s0, [sp, #128]          // eps stash, BEFORE smstart

    cbz   x23, .Lret_lzs

    smstart                       // bare: PSTATE.SM *and* PSTATE.ZA
    ptrue p0.s, all

    ldr   s0, [sp, #128]
    dup   z8.s, z0.s[0]           // z8 = {eps,...}
    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1
    dup   z5.s, z0.s[0]           // z5 = {1/N,...}

    cntw  x9
    lsl   x10, x9, #2             // 4*SVL = residency capacity in columns
    mov   x0, #0

    cmp   x24, x10
    b.hi  .Lfallback_lzs

// ===========================================================================
// ZA-resident fast path (N <= 4*SVL): 3R+1W -> 1R+1W, multi-vector staging.
// ===========================================================================
.Lza_block_lzs:
    whilelo p1.s, x0, x23
    b.none  .Ldone_lzs

    // ---- pass 1: read x ONCE, accumulate mean, stage 4 columns per MOVA ---
    mov   z4.s, #0
    mov   x12, x19
    mov   x16, x24
    mov   w8,  #0

.Lm_chunk_lzs:
    cmp   x16, #4
    b.lt  .Lm_rem_lzs
    ld1w  {z0.s}, p1/z, [x12]
    add   x12, x12, x25
    ld1w  {z1.s}, p1/z, [x12]
    add   x12, x12, x25
    ld1w  {z2.s}, p1/z, [x12]
    add   x12, x12, x25
    ld1w  {z3.s}, p1/z, [x12]
    add   x12, x12, x25
    fadd  z4.s, p1/m, z4.s, z0.s      // sequential: reference order
    fadd  z4.s, p1/m, z4.s, z1.s
    fadd  z4.s, p1/m, z4.s, z2.s
    fadd  z4.s, p1/m, z4.s, z3.s
    mova  za.s[w8, 0, vgx4], { z0.s - z3.s }
    add   w8,  w8,  #1
    sub   x16, x16, #4
    b     .Lm_chunk_lzs

.Lm_rem_lzs:
    cbz   x16, .Lmean_done_lzs
    mov   z0.s, #0
    mov   z1.s, #0
    mov   z2.s, #0
    mov   z3.s, #0
    ld1w  {z0.s}, p1/z, [x12]
    fadd  z4.s, p1/m, z4.s, z0.s
    subs  x16, x16, #1
    b.eq  .Lm_stage_lzs
    add   x12, x12, x25
    ld1w  {z1.s}, p1/z, [x12]
    fadd  z4.s, p1/m, z4.s, z1.s
    subs  x16, x16, #1
    b.eq  .Lm_stage_lzs
    add   x12, x12, x25
    ld1w  {z2.s}, p1/z, [x12]
    fadd  z4.s, p1/m, z4.s, z2.s
.Lm_stage_lzs:
    mova  za.s[w8, 0, vgx4], { z0.s - z3.s }

.Lmean_done_lzs:
    fmul  z4.s, p1/m, z4.s, z5.s      // z4 = mean

    // ---- pass 2: read x back from ZA (no memory), accumulate variance ----
    mov   z6.s, #0
    mov   x16, x24
    mov   w8,  #0

.Lv_chunk_lzs:
    cmp   x16, #4
    b.lt  .Lv_rem_lzs
    mova  { z0.s - z3.s }, za.s[w8, 0, vgx4]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmla  z6.s, p1/m, z0.s, z0.s
    fsub  z1.s, p1/m, z1.s, z4.s
    fmla  z6.s, p1/m, z1.s, z1.s
    fsub  z2.s, p1/m, z2.s, z4.s
    fmla  z6.s, p1/m, z2.s, z2.s
    fsub  z3.s, p1/m, z3.s, z4.s
    fmla  z6.s, p1/m, z3.s, z3.s
    add   w8,  w8,  #1
    sub   x16, x16, #4
    b     .Lv_chunk_lzs

.Lv_rem_lzs:
    cbz   x16, .Lvar_done_lzs
    mova  { z0.s - z3.s }, za.s[w8, 0, vgx4]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmla  z6.s, p1/m, z0.s, z0.s
    subs  x16, x16, #1
    b.eq  .Lvar_done_lzs
    fsub  z1.s, p1/m, z1.s, z4.s
    fmla  z6.s, p1/m, z1.s, z1.s
    subs  x16, x16, #1
    b.eq  .Lvar_done_lzs
    fsub  z2.s, p1/m, z2.s, z4.s
    fmla  z6.s, p1/m, z2.s, z2.s

.Lvar_done_lzs:
    fmul    z6.s, p1/m, z6.s, z5.s
    fadd    z6.s, p1/m, z6.s, z8.s
    sel     z6.s, p1, z6.s, z8.s      // inactive lanes -> eps, never rsqrt(0)
    frsqrte z7.s, z6.s
    fmul    z9.s, z7.s, z6.s
    frsqrts z9.s, z7.s, z9.s
    fmul    z7.s, z7.s, z9.s          // z7 = inv_std

    // ---- pass 3: read x back a SECOND time, normalize, write once --------
    mov   x13, x20
    mov   x14, x21
    mov   x15, x22
    mov   x16, x24
    mov   w8,  #0

.Ln_chunk_lzs:
    cmp   x16, #4
    b.lt  .Ln_rem_lzs
    mova  { z0.s - z3.s }, za.s[w8, 0, vgx4]

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmul  z0.s, p1/m, z0.s, z7.s
    fmul  z0.s, p1/m, z0.s, z10.s
    fadd  z0.s, p1/m, z0.s, z11.s
    st1w  {z0.s}, p1, [x13]
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z1.s, p1/m, z1.s, z4.s
    fmul  z1.s, p1/m, z1.s, z7.s
    fmul  z1.s, p1/m, z1.s, z10.s
    fadd  z1.s, p1/m, z1.s, z11.s
    st1w  {z1.s}, p1, [x13]
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z2.s, p1/m, z2.s, z4.s
    fmul  z2.s, p1/m, z2.s, z7.s
    fmul  z2.s, p1/m, z2.s, z10.s
    fadd  z2.s, p1/m, z2.s, z11.s
    st1w  {z2.s}, p1, [x13]
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z3.s, p1/m, z3.s, z4.s
    fmul  z3.s, p1/m, z3.s, z7.s
    fmul  z3.s, p1/m, z3.s, z10.s
    fadd  z3.s, p1/m, z3.s, z11.s
    st1w  {z3.s}, p1, [x13]
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    add   w8,  w8,  #1
    sub   x16, x16, #4
    b     .Ln_chunk_lzs

.Ln_rem_lzs:
    cbz   x16, .Lblock_next_lzs
    mova  { z0.s - z3.s }, za.s[w8, 0, vgx4]

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmul  z0.s, p1/m, z0.s, z7.s
    fmul  z0.s, p1/m, z0.s, z10.s
    fadd  z0.s, p1/m, z0.s, z11.s
    st1w  {z0.s}, p1, [x13]
    subs  x16, x16, #1
    b.eq  .Lblock_next_lzs
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z1.s, p1/m, z1.s, z4.s
    fmul  z1.s, p1/m, z1.s, z7.s
    fmul  z1.s, p1/m, z1.s, z10.s
    fadd  z1.s, p1/m, z1.s, z11.s
    st1w  {z1.s}, p1, [x13]
    subs  x16, x16, #1
    b.eq  .Lblock_next_lzs
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4

    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z2.s, p1/m, z2.s, z4.s
    fmul  z2.s, p1/m, z2.s, z7.s
    fmul  z2.s, p1/m, z2.s, z10.s
    fadd  z2.s, p1/m, z2.s, z11.s
    st1w  {z2.s}, p1, [x13]

.Lblock_next_lzs:
    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lza_block_lzs

// ===========================================================================
// Fallback (N > 4*SVL): streaming three-pass, no ZA.
// ===========================================================================
.Lfallback_lzs:
    whilelo p1.s, x0, x23
    b.none  .Ldone_lzs

    mov   z4.s, #0
    mov   x12, x19
    mov   x16, x24
.Lfb_mean_lzs:
    ld1w  {z0.s}, p1/z, [x12]
    fadd  z4.s, p1/m, z4.s, z0.s
    add   x12, x12, x25
    subs  x16, x16, #1
    b.ne  .Lfb_mean_lzs
    fmul  z4.s, p1/m, z4.s, z5.s

    mov   z6.s, #0
    mov   x12, x19
    mov   x16, x24
.Lfb_var_lzs:
    ld1w  {z0.s}, p1/z, [x12]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmla  z6.s, p1/m, z0.s, z0.s
    add   x12, x12, x25
    subs  x16, x16, #1
    b.ne  .Lfb_var_lzs

    fmul    z6.s, p1/m, z6.s, z5.s
    fadd    z6.s, p1/m, z6.s, z8.s
    sel     z6.s, p1, z6.s, z8.s
    frsqrte z7.s, z6.s
    fmul    z9.s, z7.s, z6.s
    frsqrts z9.s, z7.s, z9.s
    fmul    z7.s, z7.s, z9.s

    mov   x12, x19
    mov   x13, x20
    mov   x14, x21
    mov   x15, x22
    mov   x16, x24
.Lfb_nrm_lzs:
    ld1w  {z0.s},  p1/z, [x12]
    ld1rw {z10.s}, p0/z, [x14]
    ld1rw {z11.s}, p0/z, [x15]
    fsub  z0.s, p1/m, z0.s, z4.s
    fmul  z0.s, p1/m, z0.s, z7.s
    fmul  z0.s, p1/m, z0.s, z10.s
    fadd  z0.s, p1/m, z0.s, z11.s
    st1w  {z0.s}, p1, [x13]
    add   x12, x12, x25
    add   x13, x13, x26
    add   x14, x14, #4
    add   x15, x15, #4
    subs  x16, x16, #1
    b.ne  .Lfb_nrm_lzs

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lfallback_lzs

.Ldone_lzs:
    smstop

.Lret_lzs:
    ldr  d8,       [sp, #64]
    ldr  d9,       [sp, #72]
    ldr  d10,      [sp, #80]
    ldr  d11,      [sp, #88]
    ldr  d12,      [sp, #96]
    ldr  d13,      [sp, #104]
    ldr  d14,      [sp, #112]
    ldr  d15,      [sp, #120]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #144
    ret
