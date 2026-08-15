// layer_norm_za.S — LayerNorm staged through the SME ZA array.  A measured,
// explained NEGATIVE result, kept as an ablation row.
//
// RMSNorm's ZA prototype already lost (39-65% slower than V6, mova-bound),
// but LayerNorm has the larger residency headroom — 3R+1W, so TWO extra reads
// are available to eliminate rather than one — so the gate was resolved by
// measuring rather than inferring.
//
// This goes further than the sketched "3R->2R" partial fusion: x is staged in
// ZA once during the mean pass and reused from ZA for BOTH the variance and
// normalize passes, a true 3R+1W -> 1R+1W (50% traffic cut vs RMSNorm's 33%).
// That is the most decisive form of the test, because it costs THREE movas per
// element to RMSNorm's two.
//
// Pre-registered expectation: also a loss, and by a LARGER margin, since the
// bigger traffic cut has to clear a proportionally bigger mova tax.  Measured
// 51-68% slower than V6 — confirming mova throughput, not DRAM bandwidth, as
// the norm-agnostic constraint.
//
// The REDUCTION stays SSVE (context.md §5): both the mean sum and the sum of
// squared deviations accumulate in Z registers; ZA is pure residency staging.
//
// Same ZA constraints as rms_norm_za: tile number is an instruction immediate
// (four macro expansions), slice-select must be W12-W15 (w13), bare
// smstart/smstop for PSTATE.SM + PSTATE.ZA, every slice written before read.
// Fast path requires N <= 4*SVL (from cntw, decision D); wider rows take a
// correct streaming three-pass fallback.  Single accumulators run in strict
// column order, matching the reference.
//
// C signature: identical to layer_norm_ssve (layer_norm_ssve.S).
//   (a, b, gamma, beta, m, n, ld_a, ld_b) in x0-x7, eps in s0.

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(layer_norm_za)

.text
.align 4

// -----------------------------------------------------------------------------
// ZA_MEAN t — pass-1 body for ZA tile <t> (t in 0..3).
//   Consumes up to SVL columns from the running input pointer x8, accumulates
//   x into z4 (mean_acc), and stages each column into a vertical slice of
//   tile t. x15 = columns still to process (counts down); x9 = SVL.
//   Slice-select register MUST be W12..W15 (SME constraint) -> we use w13.
// -----------------------------------------------------------------------------
.macro ZA_MEAN t
    cmp    x15, #0
    b.le   .Lmean_skip\@         // this tile has no columns
    cmp    x15, x9
    csel   x14, x9, x15, ge      // cols_in_tile = min(SVL, remaining)
    mov    w13, #0
.Lmean_lp\@:
    ld1w   {z0.s}, p1/z, [x8]
    fadd   z4.s, p1/m, z4.s, z0.s
    mova   za\t\()v.s[w13, 0], p1/m, z0.s
    add    x8, x8, x25           // next column (stride_a bytes)
    add    w13, w13, #1
    cmp    w13, w14
    b.ne   .Lmean_lp\@
    sub    x15, x15, x14
.Lmean_skip\@:
.endm

// -----------------------------------------------------------------------------
// ZA_VAR t — pass-2 body for ZA tile <t>.
//   Reads x back from tile t (no memory read), centers on the mean (z4), and
//   accumulates the squared deviation into z6 (var_acc).
// -----------------------------------------------------------------------------
.macro ZA_VAR t
    cmp    x15, #0
    b.le   .Lvar_skip\@
    cmp    x15, x9
    csel   x14, x9, x15, ge
    mov    w13, #0
.Lvar_lp\@:
    mova   z0.s, p1/m, za\t\()v.s[w13, 0]   // x from ZA residency buffer
    fsub   z0.s, p1/m, z0.s, z4.s           // z0 = x - mean (in place)
    fmla   z6.s, p1/m, z0.s, z0.s           // var_acc += centered^2
    add    w13, w13, #1
    cmp    w13, w14
    b.ne   .Lvar_lp\@
    sub    x15, x15, x14
.Lvar_skip\@:
.endm

// -----------------------------------------------------------------------------
// ZA_NORM t — pass-3 body for ZA tile <t>.
//   Reads x back from tile t a SECOND time (still no memory read), applies
//   the centered/scaled/shifted normalize, and stores. x10 = output pointer,
//   x11 = gamma pointer, x12 = beta pointer.
// -----------------------------------------------------------------------------
.macro ZA_NORM t
    cmp    x15, #0
    b.le   .Lnorm_skip\@
    cmp    x15, x9
    csel   x14, x9, x15, ge
    mov    w13, #0
.Lnorm_lp\@:
    mova   z0.s, p1/m, za\t\()v.s[w13, 0]   // x from ZA residency buffer
    fsub   z0.s, p1/m, z0.s, z4.s           // x - mean
    fmul   z0.s, p1/m, z0.s, z7.s           // * inv_std
    ld1rw  {z1.s}, p0/z, [x11]              // gamma[col] broadcast
    ld1rw  {z2.s}, p0/z, [x12]              // beta[col] broadcast
    fmul   z0.s, p1/m, z0.s, z1.s
    fadd   z0.s, p1/m, z0.s, z2.s
    st1w   {z0.s}, p1, [x10]
    add    x10, x10, x26          // next output column (stride_b bytes)
    add    x11, x11, #4           // next gamma element
    add    x12, x12, #4           // next beta element
    add    w13, w13, #1
    cmp    w13, w14
    b.ne   .Lnorm_lp\@
    sub    x15, x15, x14
.Lnorm_skip\@:
.endm

SYMBOL(layer_norm_za):

    // --- Prologue: preserve callee-saved GPRs, d8, and a dedicated eps slot.
    // 80-byte frame: x19-x26 (four pairs, 64B) + d8 (8B) + eps stash (8B).
    // eps is stashed in memory rather than relied on via S8/D8 across smstart
    // (see rms_norm_ssve*.S / rms_norm_za.S Sprint-3 fix: streaming-mode
    // entry clobbers D8 too, not just D9-D15 -- verified empirically).
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

    mov   x19, x0                // a
    mov   x20, x1                // b
    mov   x21, x2                // gamma
    mov   x22, x3                // beta
    mov   x23, x4                // M
    mov   x24, x5                // N
    lsl   x25, x6, #2            // stride_a in bytes
    lsl   x26, x7, #2            // stride_b in bytes
    str   s0,  [sp, #128]         // dedicated eps stash, stored BEFORE smstart

    cbz   x23, .Lrow_done        // M == 0: nothing to do, no streaming region

    // smstart enables PSTATE.SM *and* PSTATE.ZA (both needed here); ZA state
    // is entirely written before it is read, so no explicit "zero {za}".
    smstart
    ptrue p0.s, all

    // eps broadcast (z29) and 1/N broadcast (z28), computed once.
    ldr    s0, [sp, #128]          // eps reloaded from the dedicated stack slot
    dup    z29.s, z0.s[0]

    fmov   s0, #1.0
    scvtf  s1, x24
    fdiv   s0, s0, s1
    dup    z28.s, z0.s[0]         // z28 = {1/N, ...}

    cntw   x9                     // x9 = SVL (FP32 elements per vector)
    mov    x0, #0                 // x0 = row index

    // Choose the path: ZA-resident only while a row fits in ZA (N <= 4*SVL).
    lsl    x10, x9, #2            // 4*SVL
    cmp    x24, x10
    b.hi   .Lfallback

    // =======================================================================
    // ZA-resident fast path (N <= 4*SVL): 1R+1W via full 3-pass ZA residency.
    // =======================================================================
.Lza_block:
    whilelo p1.s, x0, x23         // active rows in this SVL-row block
    b.none  .Ldone

    // --- Pass 1: load x, accumulate mean, stage x into ZA ------------------
    mov    z4.s, #0               // mean accumulator
    mov    x8,  x19                // input pointer, column 0 of this block
    mov    x15, x24                // columns remaining
    ZA_MEAN 0
    ZA_MEAN 1
    ZA_MEAN 2
    ZA_MEAN 3
    fmul   z4.s, p1/m, z4.s, z28.s // z4 = mean

    // --- Pass 2: read x from ZA, accumulate variance -----------------------
    mov    z6.s, #0               // variance accumulator
    mov    x15, x24
    ZA_VAR 0
    ZA_VAR 1
    ZA_VAR 2
    ZA_VAR 3

    // --- inv_std = 1/sqrt(var/N + eps), per lane (FRSQRTE + 1 NR step) -----
    fmul    z6.s, p1/m, z6.s, z28.s
    fadd    z6.s, p1/m, z6.s, z29.s
    sel     z6.s, p1, z6.s, z29.s   // inactive lanes -> eps (avoid rsqrt(0))
    frsqrte z7.s, z6.s
    fmul    z2.s, z7.s, z6.s
    frsqrts z2.s, z7.s, z2.s
    fmul    z7.s, z7.s, z2.s        // z7 = inv_std

    // --- Pass 3: read x from ZA (2nd time), normalize+scale+shift, store ---
    mov    x10, x20               // output pointer, column 0
    mov    x11, x21               // gamma pointer, column 0
    mov    x12, x22               // beta pointer, column 0
    mov    x15, x24                // columns remaining
    ZA_NORM 0
    ZA_NORM 1
    ZA_NORM 2
    ZA_NORM 3

    addvl  x19, x19, #1           // next SVL-row block (+SVL floats, contiguous)
    addvl  x20, x20, #1
    incw   x0
    b      .Lza_block

    // =======================================================================
    // Fallback (N > 4*SVL): correct streaming three-pass, 3R+1W (no ZA).
    // =======================================================================
.Lfallback:
    whilelo p1.s, x0, x23
    b.none  .Ldone

    // pass 1: mean
    mov    z4.s, #0
    mov    x8,  x19
    mov    x12, x24
.Lfb_mean:
    ld1w   {z0.s}, p1/z, [x8]
    fadd   z4.s, p1/m, z4.s, z0.s
    add    x8,  x8, x25
    subs   x12, x12, #1
    b.ne   .Lfb_mean
    fmul   z4.s, p1/m, z4.s, z28.s

    // pass 2: variance (re-read x)
    mov    z6.s, #0
    mov    x8,  x19
    mov    x12, x24
.Lfb_var:
    ld1w   {z0.s}, p1/z, [x8]
    fsub   z0.s, p1/m, z0.s, z4.s
    fmla   z6.s, p1/m, z0.s, z0.s
    add    x8,  x8, x25
    subs   x12, x12, #1
    b.ne   .Lfb_var

    fmul    z6.s, p1/m, z6.s, z28.s
    fadd    z6.s, p1/m, z6.s, z29.s
    sel     z6.s, p1, z6.s, z29.s
    frsqrte z7.s, z6.s
    fmul    z2.s, z7.s, z6.s
    frsqrts z2.s, z7.s, z2.s
    fmul    z7.s, z7.s, z2.s

    // pass 3: re-read x, normalize, store
    mov    x8,  x19
    mov    x10, x20
    mov    x11, x21
    mov    x13, x22
    mov    x12, x24
.Lfb_norm:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x11]
    ld1rw  {z2.s}, p0/z, [x13]
    fsub   z0.s, p1/m, z0.s, z4.s
    fmul   z0.s, p1/m, z0.s, z7.s
    fmul   z0.s, p1/m, z0.s, z1.s
    fadd   z0.s, p1/m, z0.s, z2.s
    st1w   {z0.s}, p1, [x10]
    add    x8,  x8,  x25
    add    x10, x10, x26
    add    x11, x11, #4
    add    x13, x13, #4
    subs   x12, x12, #1
    b.ne   .Lfb_norm

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Lfallback

.Ldone:
    smstop

.Lrow_done:
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
