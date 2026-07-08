// rms_norm_za.S — Sprint 3: RMSNorm with ZA-tile residency (Ketsia)
//
// The lever ZA offers a bandwidth-bound RMSNorm (context.md §5, ROADMAP
// Sprint 3 "honest expectation-setting"):
//
//   RMSNorm must see a row's ENTIRE sum-of-squares before it can normalise,
//   so the SSVE winner (V6) re-reads x in pass 2 -> 2R+1W of traffic.  That
//   second read is only avoidable if the whole row-block stays resident on
//   core between the two passes.  ZA gives ~4 KB of extra on-core storage
//   (four 32-bit tiles ZA0..ZA3, each SVL_S x SVL_S = 16x16 FP32 on the M4)
//   BEYOND the 32 Z-registers — exactly enough to hold one SVL-row x (<=4*SVL)
//   -col block.  We use it as a residency buffer so pass 2 reads x from ZA,
//   not from memory: 1R+1W instead of 2R+1W.
//
//   The REDUCTION stays SSVE (fmla in Z-registers, per context.md §5 — a
//   horizontal sum-of-squares maps poorly onto the ZA matrix accumulator).
//   ZA is pure staging: it holds x, it does not compute the statistic.
//
// Structure per SVL-row block (rows are contiguous in column-major memory,
// so one SVL-vector load grabs SVL consecutive rows of one column):
//   pass 1  (1 memory read): for each column   z = ld1w x;  acc += z*z;
//                            mova z -> ZA slice  (stage for pass 2)
//   inv_rms = 1/sqrt(acc/N + eps)   (FRSQRTE + one Newton step, as in V6)
//   pass 2  (1 memory write): for each column  z = mova ZA slice;
//                             z *= gamma[col];  z *= inv_rms;  st1w z
//
// The ZA-resident fast path only applies while the row fits in ZA
// (N <= 4*SVL, derived at runtime — decision D, never a literal 64).  For
// wider rows the block cannot stay resident; we fall back to a correct
// streaming two-pass (2R+1W) so the kernel is right for every shape.  That
// fallback is NOT the performance story — V6 remains the large-N incumbent;
// rms_norm_za is measured for the small-N regime where residency can pay.
//
// A single sum-of-squares accumulator (z2) runs strictly in column order
// across all four tile sections, so the summation order matches the scalar
// reference exactly — no reassociation tolerance needed (kTol = 1e-5).
//
// C signature (identical to the SSVE kernels):
//   void rms_norm_za(const float *a,     // x0
//                    float       *b,     // x1
//                    const float *gamma, // x2
//                    int64_t      m,     // x3
//                    int64_t      n,     // x4
//                    int64_t      ld_a,  // x5
//                    int64_t      ld_b,  // x6
//                    float        eps);  // s0

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(rms_norm_za)

.text
.align 4

// -----------------------------------------------------------------------------
// ZA_LOAD t — pass-1 body for ZA tile <t> (t in 0..3).
//   Consumes up to SVL columns from the running input pointer x8, accumulates
//   x*x into z2, and stages each column into a vertical slice of tile t.
//   x15 = columns still to process (counts down); x9 = SVL (elems/vector).
//   Slice-select register MUST be W12..W15 (SME constraint) -> we use w13.
// -----------------------------------------------------------------------------
.macro ZA_LOAD t
    cmp    x15, #0
    b.le   .Lload_skip\@         // this tile has no columns
    cmp    x15, x9
    csel   x14, x9, x15, ge      // cols_in_tile = min(SVL, remaining)
    mov    w13, #0
.Lload_lp\@:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s, p1/m, z0.s, z0.s
    mova   za\t\()v.s[w13, 0], p1/m, z0.s
    add    x8, x8, x24           // next column (stride_a bytes)
    add    w13, w13, #1
    cmp    w13, w14
    b.ne   .Lload_lp\@
    sub    x15, x15, x14
.Lload_skip\@:
.endm

// -----------------------------------------------------------------------------
// ZA_STORE t — pass-2 body for ZA tile <t>.
//   Reads x back from tile t (no memory read), applies gamma[col] and inv_rms
//   (z4), and stores.  x10 = output pointer, x11 = gamma pointer.
// -----------------------------------------------------------------------------
.macro ZA_STORE t
    cmp    x15, #0
    b.le   .Lstore_skip\@
    cmp    x15, x9
    csel   x14, x9, x15, ge
    mov    w13, #0
.Lstore_lp\@:
    mova   z0.s, p1/m, za\t\()v.s[w13, 0]   // x from ZA residency buffer
    ld1rw  {z1.s}, p0/z, [x11]              // gamma[col] broadcast
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s           // * inv_rms
    st1w   {z0.s}, p1, [x10]
    add    x10, x10, x25          // next output column (stride_b bytes)
    add    x11, x11, #4           // next gamma element
    add    w13, w13, #1
    cmp    w13, w14
    b.ne   .Lstore_lp\@
    sub    x15, x15, x14
.Lstore_skip\@:
.endm

SYMBOL(rms_norm_za):

    // --- Prologue: preserve callee-saved GPRs and d8 (eps stash) -----------
    stp  x19, x20, [sp, #-64]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    str  x25,      [sp, #48]
    str  d8,       [sp, #56]

    mov   x19, x0                // a
    mov   x20, x1                // b
    mov   x21, x2                // gamma
    mov   x22, x3                // M
    mov   x23, x4                // N
    lsl   x24, x5, #2            // stride_a in bytes
    lsl   x25, x6, #2            // stride_b in bytes
    fmov  s8,  s0                // stash eps in d8 (survives smstart)

    cbz   x22, .Lrow_done        // M == 0: nothing to do, no streaming region

    // smstart enables PSTATE.SM *and* PSTATE.ZA (both needed here); ZA state
    // is entirely written before it is read, so no explicit "zero {za}".
    smstart
    ptrue p0.s, all

    // eps broadcast (z8) and 1/N broadcast (z5), computed once.
    ldr    s0, [sp, #56]         // eps from the d8 slot
    dup    z8.s, z0.s[0]
    fmov   s0, #1.0
    scvtf  s1, x23
    fdiv   s0, s0, s1
    dup    z5.s, z0.s[0]         // z5 = {1/N, ...}

    cntw   x9                    // x9 = SVL (FP32 elements per vector)
    mov    x0, #0                // x0 = row index

    // Choose the path: ZA-resident only while a row fits in ZA (N <= 4*SVL).
    lsl    x10, x9, #2           // 4*SVL
    cmp    x23, x10
    b.hi   .Lfallback

    // =======================================================================
    // ZA-resident fast path (N <= 4*SVL): 1R+1W via ZA residency.
    // =======================================================================
.Lza_block:
    whilelo p1.s, x0, x22        // active rows in this SVL-row block
    b.none  .Ldone

    // --- Pass 1: load x, reduce sum-of-squares, stage x into ZA ------------
    mov    z2.s, #0              // per-lane sum-of-squares accumulator
    mov    x8,  x19              // input pointer, column 0 of this block
    mov    x15, x23              // columns remaining
    ZA_LOAD 0
    ZA_LOAD 1
    ZA_LOAD 2
    ZA_LOAD 3

    // --- inv_rms = 1/sqrt(sumsq/N + eps), per lane (FRSQRTE + 1 NR step) ---
    fmul    z2.s, p1/m, z2.s, z5.s
    fadd    z2.s, p1/m, z2.s, z8.s
    sel     z2.s, p1, z2.s, z8.s   // inactive lanes -> eps (avoid rsqrt(0))
    frsqrte z4.s, z2.s
    fmul    z3.s, z4.s, z2.s
    frsqrts z3.s, z4.s, z3.s
    fmul    z4.s, z4.s, z3.s       // z4 = inv_rms

    // --- Pass 2: read x from ZA, apply gamma + inv_rms, store --------------
    mov    x10, x20              // output pointer, column 0
    mov    x11, x21              // gamma pointer, column 0
    mov    x15, x23              // columns remaining
    ZA_STORE 0
    ZA_STORE 1
    ZA_STORE 2
    ZA_STORE 3

    addvl  x19, x19, #1          // next SVL-row block (+SVL floats, contiguous)
    addvl  x20, x20, #1
    incw   x0
    b      .Lza_block

    // =======================================================================
    // Fallback (N > 4*SVL): correct streaming two-pass, 2R+1W (no ZA).
    // =======================================================================
.Lfallback:
    whilelo p1.s, x0, x22
    b.none  .Ldone

    // pass 1: sum of squares
    mov    z2.s, #0
    mov    x8,  x19
    mov    x12, x23
.Lfb_red:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s, p1/m, z0.s, z0.s
    add    x8,  x8, x24
    subs   x12, x12, #1
    b.ne   .Lfb_red

    fmul    z2.s, p1/m, z2.s, z5.s
    fadd    z2.s, p1/m, z2.s, z8.s
    sel     z2.s, p1, z2.s, z8.s
    frsqrte z4.s, z2.s
    fmul    z3.s, z4.s, z2.s
    frsqrts z3.s, z4.s, z3.s
    fmul    z4.s, z4.s, z3.s

    // pass 2: re-read x, normalise, store
    mov    x8,  x19
    mov    x10, x20
    mov    x11, x21
    mov    x12, x23
.Lfb_nrm:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x11]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x10]
    add    x8,  x8,  x24
    add    x10, x10, x25
    add    x11, x11, #4
    subs   x12, x12, #1
    b.ne   .Lfb_nrm

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lfallback

.Ldone:
    smstop

.Lrow_done:
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #64
    ret
