// layer_norm_ssve_v1.S — LayerNorm V1: FRSQRTE + one Newton-Raphson step.
//
// Change vs V0: inv_std drops FSQRT + FMOV(1.0) + FDIV for
//   e = FRSQRTE(var+eps); t = e*(var+eps); c = FRSQRTS(e,t); inv_std = e*c
// Passes 1 and 3 are unchanged.  Measured ~0, same verdict as RMSNorm V1:
// reciprocal-sqrt latency is not on the critical path.
//
// Inactive z9 lanes hold 0 after the variance reduction, and FRSQRTE is
// unpredicated, so a SEL fills them with eps first — otherwise FRSQRTE(0) is
// +inf and the NR step propagates NaN.  The final FMUL is p1/m regardless.
//
// C signature: identical to layer_norm_ssve (layer_norm_ssve.S).
//   (a, b, gamma, beta, m, n, ld_a, ld_b) in x0-x7, eps in s0.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v1)
.text
.align 2

// Register map (as V0, plus z3 as NR scratch):
//   x8 SVL lanes   x9 r_base   x10/x12/x13/x17 walking a/b/gamma/beta ptrs
//   x11 col index  x14/x15 column strides   x16 SVL*4   x19/x20 chunk bases
//   z0 data  z1 gamma  z2 beta  z3 NR scratch  z5 inv_std
//   z6 {n,...}  z7 {eps,...}  z8 mean  z9 var+eps   p1 row predicate

SYM(layer_norm_ssve_v1):

// ── Prologue ──────────────────────────────────────────────────────────────────
// Frame grown 48 -> 112 bytes to save the FULL d8-d15 callee-saved range
// (AAPCS64) — see layer_norm_ssve.S for the full rationale: smstart/smstop
// zero all of d8-d15, and z8/z9 are this kernel's accumulators.
    stp  x29, x30, [sp, #-112]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    str  d8,  [sp, #48]
    str  d9,  [sp, #56]
    str  d10, [sp, #64]
    str  d11, [sp, #72]
    str  d12, [sp, #80]
    str  d13, [sp, #88]
    str  d14, [sp, #96]
    str  d15, [sp, #104]
    str  s0,  [sp, #32]            // save epsilon before smstart

    smstart sm

// ── One-time setup ────────────────────────────────────────────────────────────
    cntw x8
    lsl  x14, x6, #2               // stride_col_a = ld_a * 4
    lsl  x15, x7, #2               // stride_col_b = ld_b * 4
    lsl  x16, x8, #2               // SVL * 4

    ldr  s22, [x29, #32]
    dup  z7.s, z22.s[0]            // z7 = [eps, ..., eps]

    scvtf s21, x5
    dup  z6.s, z21.s[0]            // z6 = [n, ..., n]

    mov  x9,  xzr
    mov  x19, x0
    mov  x20, x1

// ── Outer loop ────────────────────────────────────────────────────────────────
Louter_v1:
    cmp  x9, x4
    b.ge Louter_done_v1

    whilelo p1.s, x9, x4

// ════════════════════════════════════════════════════════════════════════════
// PASS 1 — per-row sum → mean   (identical to V0)
// ════════════════════════════════════════════════════════════════════════════
    fmov z8.s, #0.0
    mov  x10, x19
    mov  x11, xzr

Lp1col_v1:
    cmp  x11, x5
    b.ge Lp1col_done_v1

    ld1w {z0.s}, p1/z, [x10]
    fadd z8.s, p1/m, z8.s, z0.s

    add  x10, x10, x14
    add  x11, x11, #1
    b    Lp1col_v1

Lp1col_done_v1:
    fdiv z8.s, p1/m, z8.s, z6.s   // z8 = mean

// ════════════════════════════════════════════════════════════════════════════
// PASS 2 — per-row variance → inv_std via FRSQRTE + one NR step  (V1 change)
// ════════════════════════════════════════════════════════════════════════════
    fmov z9.s, #0.0
    mov  x10, x19
    mov  x11, xzr

Lp2col_v1:
    cmp  x11, x5
    b.ge Lp2col_done_v1

    ld1w {z0.s}, p1/z, [x10]
    fsub z0.s, p1/m, z0.s, z8.s   // x - mean
    fmul z0.s, p1/m, z0.s, z0.s   // (x - mean)^2
    fadd z9.s, p1/m, z9.s, z0.s   // accumulate variance

    add  x10, x10, x14
    add  x11, x11, #1
    b    Lp2col_v1

Lp2col_done_v1:
    fdiv z9.s, p1/m, z9.s, z6.s   // variance = sum / n
    fadd z9.s, p1/m, z9.s, z7.s   // var + eps  (active lanes only)

    // Inactive z9 lanes are 0 and FRSQRTE is unpredicated; FRSQRTE(0) = +inf.
    sel  z9.s, p1, z9.s, z7.s     // z9[inactive] = eps

    frsqrte z5.s, z9.s             // z5 = estimate e
    fmul    z3.s, z5.s, z9.s       // z3 = t = e * (var+eps)
    frsqrts z3.s, z5.s, z3.s       // z3 = c = (3 - e*t) / 2
    fmul    z5.s, p1/m, z5.s, z3.s // z5[active] = inv_std = e * c

// ════════════════════════════════════════════════════════════════════════════
// PASS 3 — normalise, scale, shift, store   (identical to V0)
// ════════════════════════════════════════════════════════════════════════════
    mov  x10, x19
    mov  x12, x20
    mov  x13, x2
    mov  x17, x3
    mov  x11, xzr

Lp3col_v1:
    cmp  x11, x5
    b.ge Lp3col_done_v1

    ld1w  {z0.s}, p1/z, [x10]
    fsub  z0.s, p1/m, z0.s, z8.s  // x - mean
    fmul  z0.s, p1/m, z0.s, z5.s  // x_hat = (x-mean)*inv_std
    ld1rw {z1.s}, p1/z, [x13]
    fmul  z0.s, p1/m, z0.s, z1.s  // x_hat * gamma
    ld1rw {z2.s}, p1/z, [x17]
    fadd  z0.s, p1/m, z0.s, z2.s  // + beta
    st1w  {z0.s}, p1, [x12]

    add  x10, x10, x14
    add  x12, x12, x15
    add  x13, x13, #4
    add  x17, x17, #4
    add  x11, x11, #1
    b    Lp3col_v1

Lp3col_done_v1:

    add  x19, x19, x16
    add  x20, x20, x16
    add  x9,  x9,  x8
    b    Louter_v1

Louter_done_v1:

// ── Epilogue ──────────────────────────────────────────────────────────────────
    smstop sm

    ldr  d8,  [sp, #48]
    ldr  d9,  [sp, #56]
    ldr  d10, [sp, #64]
    ldr  d11, [sp, #72]
    ldr  d12, [sp, #80]
    ldr  d13, [sp, #88]
    ldr  d14, [sp, #96]
    ldr  d15, [sp, #104]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #112
    ret
