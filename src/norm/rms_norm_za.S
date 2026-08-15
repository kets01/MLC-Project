// rms_norm_za.S — RMSNorm staged through the SME ZA array.  A measured,
// explained NEGATIVE result kept in the tree as an ablation row.
//
// The idea: RMSNorm must see a row's entire sum of squares before it can
// normalise, so V6 re-reads x in pass 2 (2R+1W).  That read is avoidable only
// if the row-block stays on core between passes.  ZA offers ~4 KB beyond the
// 32 Z-registers (four 32-bit tiles, SVL_S x SVL_S each), enough for one
// SVL-row x (<=4*SVL)-col block — so stage x in ZA during the reduction and
// read it back in pass 2: 1R+1W instead of 2R+1W.
//
// Verdict: 39-65% SLOWER than V6, at every footprint tested.  Each element
// costs two extra `mova` ops, and mova throughput — not memory — is what
// binds at these shapes; the kernel's own streaming fallback beats its ZA
// path.  V6's grouping also keeps pass 2's re-read cache-resident, so the DRAM
// read ZA "saves" was never a DRAM read.
//
// The REDUCTION stays SSVE (context.md §5): a horizontal sum of squares maps
// poorly onto a matrix accumulator, so ZA holds x and computes nothing.
//
// Constraints designed around: the ZA tile number is an instruction immediate
// (hence four macro expansions, not a loop), the slice-select register must be
// W12-W15 (w13 here), and bare smstart/smstop manage PSTATE.ZA as well as
// PSTATE.SM.  Every slice is written before it is read, so no `zero {za}` is
// needed, and ZA is disabled before return (no AAPCS64 lazy-save hazard).
//
// Fast path requires N <= 4*SVL (derived from cntw, decision D — never a
// literal 64); wider rows take a correct streaming two-pass fallback.  One
// accumulator runs in strict column order, matching the reference exactly.
//
// C signature: identical to rms_norm_ssve (rms_norm_ssve.S).
//   (a, b, gamma, m, n, ld_a, ld_b) in x0-x6, eps in s0.

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

    // --- Prologue: preserve callee-saved GPRs; frame extended to 80 bytes
    // (was 64) to fit a dedicated eps stash slot at [sp,#64]. ----------------
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

    mov   x19, x0                // a
    mov   x20, x1                // b
    mov   x21, x2                // gamma
    mov   x22, x3                // M
    mov   x23, x4                // N
    lsl   x24, x5, #2            // stride_a in bytes
    lsl   x25, x6, #2            // stride_b in bytes
    fmov  s8,  s0                // stash eps in d8 too (kept for consistency)
    str   s0,  [sp, #120]         // dedicated eps stash, stored BEFORE smstart

    cbz   x22, .Lrow_done        // M == 0: nothing to do, no streaming region

    // smstart enables PSTATE.SM *and* PSTATE.ZA (both needed here); ZA state
    // is entirely written before it is read, so no explicit "zero {za}".
    smstart
    ptrue p0.s, all

    // eps broadcast (z8) and 1/N broadcast (z5), computed once.
    ldr    s0, [sp, #120]          // eps reloaded from the dedicated stack slot
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
