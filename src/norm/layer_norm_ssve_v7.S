// layer_norm_ssve_v7.S — LayerNorm V7: V6 with SME2 multi-vector accesses.
//
// Built as the DISCRIMINATING experiment for RMSNorm V7's surprising +17%.
// Two mechanisms explain that win and predict opposite outcomes here:
//   - "fewer instructions retire faster" -> LayerNorm folds MORE accesses per
//     element (three passes, not two), so it should gain MORE;
//   - "relieved memory-level parallelism" -> LayerNorm does more FP work per
//     byte and is less memory-starved, so it should gain LESS.
// Pre-registered prediction: less than RMSNorm's +17%.  Measured +2.7%, which
// falsifies the instruction-count explanation.  (A later 2-vector control then
// falsified the MLP one too; see rms_norm_ssve_v7x2.S.)
//
// The change itself is only instruction count: V6 already touches four
// consecutive VL-row blocks per column, so each pass folds 4x LD1W into one
// 4-vector ld1w and pass 3 folds 4x ST1W into one st1w.  Same addresses, same
// traffic, same summation order — V7 must be BIT-IDENTICAL to V6.
//
// NO TAIL PATH: `whilelo pn8.s, x0, x23, vlx4` describes min(M - x0, 4*VL)
// active elements across the group, so one body serves full and partial
// groups.  V6's separate tail touched 64 B per column instead of 256 B, i.e.
// the low access density V6 exists to fix.
//
// Inactive-lane safety: loads are pn8/z, so a fully-inactive block ends with
// mean 0 and variance 0 -> +eps, which FRSQRTE handles finitely; the store
// uses the same counter.
//
// Requires FEAT_SME2; the C++ bridge dispatches to V6 when it is absent.
//
// C signature: identical to layer_norm_ssve (layer_norm_ssve.S).
//   (a, b, gamma, beta, m, n, ld_a, ld_b) in x0-x7, eps in s0.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.arch armv9-a+sme2

.global SYM(layer_norm_ssve_v7)
.text
.align 4

// Register map (constant post-prologue) — identical to V6, which already
// keeps the data in a consecutive 4-aligned quad (z0-z3), so the multi-vector
// forms drop in without reallocating anything:
//   x19 = a-chunk block-0 base    x20 = b-chunk block-0 base
//   x21 = gamma                   x22 = beta
//   x23 = m (rows)                x24 = n (cols)
//   x25 = stride_a (B)            x26 = stride_b (B)
//   x14 = 4*VL (elements, set after SMSTART)
//   z4  = [1/N,...]               z7  = [eps,...]
//   z8..z11  = mean accumulators/values for blocks 0..3
//   z12..z15 = variance accumulators for blocks 0..3
//   z16      = NR scratch         z17..z20 = inv_std for blocks 0..3
//   z0..z3   = data quad          z5 = gamma broadcast   z6 = beta broadcast
//   p0  = all-true SVE predicate  pn8 = all-true counter predicate (SME2)
// Tail path (p1 = whilelo, V2-style single block, plain SVE):
//   z8 = mean acc/value    z9 = var acc    z5 = inv_std

SYM(layer_norm_ssve_v7):

    // Prologue — 144-byte frame saving the FULL d8-d15 callee-saved range
    // (AAPCS64).  smstart/smstop zero all of d8-d15, and this kernel uses
    // z8-z15 as its accumulators, so the save is doubly required.
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
    str   s0, [sp, #128]           // dedicated eps stash, stored BEFORE smstart

    cbz   x23, .Lret_v7            // m == 0: never enter the streaming region,
                                    // so skip the smstop as well

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #128]
    dup   z7.s, z0.s[0]            // z7 = [eps,...]

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
.Lgroup_v7:
    whilelo pn8.s, x0, x23, vlx4   // all-true while >= 4*VL rows remain,
    b.none  .Ldone_v7              // partial for the last group, none at end

    // ── Group pass 1: mean (one 4-vector load per column) ────────
    mov   z8.s,  #0
    mov   z9.s,  #0
    mov   z10.s, #0
    mov   z11.s, #0

    mov   x8, x19
    mov   x9, x24

.Lgp1col_v7:
    ld1w  {z0.s - z3.s}, pn8/z, [x8]
    fadd  z8.s,  p0/m, z8.s,  z0.s
    fadd  z9.s,  p0/m, z9.s,  z1.s
    fadd  z10.s, p0/m, z10.s, z2.s
    fadd  z11.s, p0/m, z11.s, z3.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lgp1col_v7

    fmul  z8.s,  p0/m, z8.s,  z4.s // mean_0
    fmul  z9.s,  p0/m, z9.s,  z4.s // mean_1
    fmul  z10.s, p0/m, z10.s, z4.s // mean_2
    fmul  z11.s, p0/m, z11.s, z4.s // mean_3

    // ── Group pass 2: centered variance (the stable two-pass form) ─
    mov   z12.s, #0
    mov   z13.s, #0
    mov   z14.s, #0
    mov   z15.s, #0

    mov   x8, x19
    mov   x9, x24

.Lgp2col_v7:
    ld1w  {z0.s - z3.s}, pn8/z, [x8]
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
    b.ne  .Lgp2col_v7

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

.Lgp3col_v7:
    ld1w  {z0.s - z3.s}, pn8/z, [x8]
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
    st1w  {z0.s - z3.s}, pn8, [x9]
    add   x8,  x8,  x25
    add   x9,  x9,  x26
    add   x10, x10, #4
    add   x11, x11, #4
    subs  x12, x12, #1
    b.ne  .Lgp3col_v7

    addvl x19, x19, #4
    addvl x20, x20, #4
    add   x0, x0, x14
    b     .Lgroup_v7

.Ldone_v7:
    smstop sm

.Lret_v7:
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
