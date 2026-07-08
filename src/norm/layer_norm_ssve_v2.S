// layer_norm_ssve_v2.S — LayerNorm SSVE Sprint 2c ablation V2
//
// Change vs V1: pre-compute 1/N once (scalar FDIV before the outer loop,
// broadcast into z4).  Both per-block vector FDIVs for mean and variance
// become vector FMULs by z4.  Passes 1–3 and the FRSQRTE+NR inv_std step
// are otherwise identical to V1.
//
// Adopts the RMSNorm-style prologue (ADDVL/INCW outer loop, all arguments
// moved to callee-saved registers before SMSTART) for consistency with the
// rest of the ablation series.
//
// C signature:
//   void layer_norm_ssve_v2(const float *a, float *b,
//                           const float *gamma, const float *beta,
//                           int64_t m, int64_t n,
//                           int64_t ld_a, int64_t ld_b, float eps);
//
// Register map (post-prologue):
//   x19 = a chunk base   x20 = b chunk base
//   x21 = gamma          x22 = beta
//   x23 = m              x24 = n
//   x25 = stride_a(B)    x26 = stride_b(B)
//   d8  = eps            x0  = row counter
//   z4  = [1/N,...]      z7  = [eps,...]
//   z8  = mean acc       z9  = var acc
//   z5  = inv_std        z3  = NR scratch

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_v2)
.text
.align 4

SYM(layer_norm_ssve_v2):

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

    cbz   x23, .Ldone_v2

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #64]
    dup   z7.s, z0.s[0]            // z7 = [eps,...]

    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1               // s0 = 1/N  (one-time scalar FDIV)
    dup   z4.s, z0.s[0]            // z4 = [1/N,...]

    mov   x0, #0

.Lblock_v2:
    whilelo p1.s, x0, x23
    b.none  .Lblkdone_v2

    // ── Pass 1: mean ──────────────────────────────────────────────
    mov   z8.s, #0
    mov   x8, x19
    mov   x9, x24

.Lp1_v2:
    ld1w  {z0.s}, p1/z, [x8]
    fadd  z8.s, p1/m, z8.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp1_v2

    fmul  z8.s, p1/m, z8.s, z4.s  // mean = sum * 1/N

    // ── Pass 2: variance + inv_std ────────────────────────────────
    mov   z9.s, #0
    mov   x8, x19
    mov   x9, x24

.Lp2_v2:
    ld1w  {z0.s}, p1/z, [x8]
    fsub  z0.s, p1/m, z0.s, z8.s
    fmla  z9.s, p1/m, z0.s, z0.s
    add   x8, x8, x25
    subs  x9, x9, #1
    b.ne  .Lp2_v2

    fmul  z9.s, p1/m, z9.s, z4.s  // var = sq_sum * 1/N
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

.Lp3_v2:
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
    b.ne  .Lp3_v2

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lblock_v2

.Lblkdone_v2:
    smstop sm

.Ldone_v2:
    ldr  d8,       [sp, #64]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
