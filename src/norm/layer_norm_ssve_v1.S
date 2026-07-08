// layer_norm_ssve_v1.S — LayerNorm SSVE Sprint 2c ablation V1
//
// Change vs V0 (layer_norm_ssve.S):
//   inv_std replaces FSQRT + FMOV(1.0) + FDIV with FRSQRTE + one
//   Newton-Raphson step (FRSQRTS).
//   Everything else — prologue, passes 1 and 3, VLA predication — is
//   identical to V0.
//
// NR sequence for 1/sqrt(a), where a = var + eps  (z9 after pass 2):
//   SEL z9, p1, z9, z7     — fill inactive lanes with eps before unpredicated ops
//   e  = FRSQRTE(z9)       — estimate  e ≈ 1/sqrt(a)           (1 cycle)
//   t  = e * z9            — NR intermediate t = e·a
//   c  = FRSQRTS(e, t)     — correction    c = (3 − e·t) / 2
//   inv_std = e * c        — refined result, FP32-accurate after one step
//
// Inactive-lane safety: after the variance reduction, inactive z9 lanes hold
// 0.0 (never touched by p1/m or p1/z operations).  FRSQRTE is unpredicated
// and would give +inf for those 0-valued lanes; NR would then produce NaN.
// The SEL fills those lanes with eps (z7) before FRSQRTE, keeping all
// intermediates finite and preventing FPSR exception-flag side effects.
// The final FMUL is predicated (p1/m), so inactive z5 lanes are never
// updated and the junk from eps-based NR never reaches pass 3.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v1)
.text
.align 2

// ─── Calling convention (AArch64) ────────────────────────────────────────────
//  x0  = a       (input  matrix, column-major, leading dim ld_a)
//  x1  = b       (output matrix, column-major, leading dim ld_b)
//  x2  = gamma   (scale vector, N floats, contiguous)
//  x3  = beta    (shift vector, N floats, contiguous)
//  x4  = m       (number of rows)
//  x5  = n       (feature size)
//  x6  = ld_a    (leading dimension of a)
//  x7  = ld_b    (leading dimension of b)
//  s0  = epsilon (first FP argument register)
//
// ─── Register map ────────────────────────────────────────────────────────────
//  x8         SVL in FP32 elements
//  x9         r_base — outer loop: first row of current chunk
//  x10        walking column pointer in a
//  x11        col index
//  x12        walking pointer in b   (pass 3)
//  x13        walking pointer in gamma (pass 3)
//  x14        stride_col_a = ld_a * 4
//  x15        stride_col_b = ld_b * 4
//  x16        SVL * 4
//  x17        walking pointer in beta (pass 3)
//  x19        a_chunk_base (callee-saved)
//  x20        b_chunk_base (callee-saved)
//
//  z0         data register (reused across passes)
//  z1         gamma broadcast (pass 3)
//  z2         beta  broadcast (pass 3)
//  z3         NR scratch  (NEW in V1: replaces idle z3 from V0)
//  z5         per-row inv_std (result of pass 2)
//  z6         [n, n, ..., n]
//  z7         [eps, eps, ...]
//  z8         per-row mean (accumulator pass 1, reused in passes 2+3)
//  z9         per-row var+eps accumulator (pass 2)
//
//  p1         row predicate: p1[j] = 1 iff (r_base + j) < m

SYM(layer_norm_ssve_v1):

// ── Prologue ──────────────────────────────────────────────────────────────────
    stp  x29, x30, [sp, #-48]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
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

// Fill inactive lanes with eps before the unpredicated FRSQRTE.
// Inactive z9 lanes are 0 after the p1/m operations above; FRSQRTE(0) = +inf.
// SEL z9.s, p1, z9.s, z7.s replaces inactive lanes with eps so FRSQRTE
// sees a valid positive input everywhere and no exception flags are raised.
    sel  z9.s, p1, z9.s, z7.s     // z9[inactive] = eps

// Newton-Raphson reciprocal square root:
//   e = FRSQRTE(var+eps)           — ~8-bit accurate estimate of 1/sqrt(a)
//   t = e * (var+eps)              — NR intermediate: should equal sqrt(a)
//   c = FRSQRTS(e, t)             — correction factor (3 - e*t) / 2
//   inv_std = e * c               — one NR step reaches full FP32 accuracy
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

    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #48
    ret
