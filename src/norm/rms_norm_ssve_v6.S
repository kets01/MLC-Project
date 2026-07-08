// rms_norm_ssve_v6.S — Sprint 2b ablation V6 (contiguity blocking)
//
// What the measurement said (see the Sprint-2b report section): the ROADMAP's
// original V6 premise — "make pass 2 re-read x from cache, not DRAM" — is
// already satisfied by the loop structure: the reuse distance between pass 1
// and pass 2 of one VL-row block is 64*N bytes (~128-256 KB at bench shapes),
// which is L2-resident by construction.  The measured DRAM-regime collapse
// (footprint > L2: 36 -> 18 -> 11 GiB/s) correlates with ACCESS DENSITY
// instead: each column touch reads 64 B out of every 4*M bytes, and the
// sparser that walk, the worse DRAM/prefetcher efficiency gets (at equal
// 32 MB footprint: M=512/N=8192 runs 31 GiB/s, M=2048/N=2048 runs 18 GiB/s).
//
// Change vs V4: process FOUR consecutive VL-row blocks per outer iteration.
//   - Each column touch becomes 4 consecutive LD1Ws ([x8, #k, MUL VL]) =
//     256 B contiguous, 4x denser DRAM access per pass.
//   - The four FMLA accumulators are now one PER ROW-BLOCK: still four
//     independent chains (keeps V4's ILP win), but each row's sum
//     accumulates sequentially over columns — the SAME summation order as
//     the scalar reference, so V6 needs no reassociation tolerance.
//   - inv_rms is computed for four blocks per group (amortized), and pass 2
//     shares each LD1RW gamma broadcast across the four blocks (4x fewer
//     gamma loads).
//
// Rows beyond the last full 4-block group fall back to a V2-style
// single-block predicated loop (at most 3 blocks + tail; perf shapes are
// group-dominated).  Group path needs no predication: all lanes active (p0).
//
// C signature (identical to V0-V5):
//   void rms_norm_ssve_v6(const float *a,     // x0
//                         float       *b,     // x1
//                         const float *gamma, // x2
//                         int64_t      m,     // x3
//                         int64_t      n,     // x4
//                         int64_t      ld_a,  // x5
//                         int64_t      ld_b,  // x6
//                         float        eps);  // s0

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(rms_norm_ssve_v6)

.text
.align 4

SYMBOL(rms_norm_ssve_v6):

    // Prologue — frame extended to 80 bytes (was 64) to fit a dedicated eps
    // stash slot at [sp,#64] alongside the original layout.
    stp  x19, x20, [sp, #-80]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    str  x25,      [sp, #48]
    str  d8,       [sp, #56]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    lsl   x24, x5, #2          // stride_a in bytes
    lsl   x25, x6, #2          // stride_b in bytes
    fmov  s8,  s0
    str   s0,  [sp, #64]       // dedicated eps stash, stored BEFORE smstart
                                // (streaming-mode entry clobbers D8 too, not
                                // just D9-D15 — memory survives smstart, no
                                // register does; verified empirically)

    cbz   x22, .Lrow_done_v6

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #64]       // eps reloaded from the dedicated stack slot
    dup    z8.s, z0.s[0]       // z8 = {eps, ...}

    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1
    dup   z5.s, z0.s[0]        // z5 = {1/N, ...}

    cntw  x14
    lsl   x14, x14, #2         // x14 = 4*VL rows per group
    mov   x0, #0               // x0 = row index

    // -----------------------------------------------------------------------
    // Group loop — four full VL-row blocks per iteration, no predication.
    // -----------------------------------------------------------------------
.Lgroup_loop_v6:
    add   x6, x0, x14
    cmp   x6, x22
    b.hi  .Ltail_blocks_v6     // fewer than 4*VL rows left

    // === Pass 1: one column at a time, 4 contiguous block loads ============
    mov    z2.s,  #0           // accumulator, block 0
    mov    z16.s, #0           // accumulator, block 1
    mov    z17.s, #0           // accumulator, block 2
    mov    z18.s, #0           // accumulator, block 3
    mov    x8,   x19
    mov    x9,   x23

.Lred_col_v6:
    ld1w   {z0.s},  p0/z, [x8]
    ld1w   {z24.s}, p0/z, [x8, #1, mul vl]
    ld1w   {z25.s}, p0/z, [x8, #2, mul vl]
    ld1w   {z26.s}, p0/z, [x8, #3, mul vl]
    fmla   z2.s,  p0/m, z0.s,  z0.s
    fmla   z16.s, p0/m, z24.s, z24.s
    fmla   z17.s, p0/m, z25.s, z25.s
    fmla   z18.s, p0/m, z26.s, z26.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_col_v6

    // === inv_rms for the four blocks (z2/z16/z17/z18 -> z4/z19/z20/z21) ====
    fmul   z2.s,  p0/m, z2.s,  z5.s
    fmul   z16.s, p0/m, z16.s, z5.s
    fmul   z17.s, p0/m, z17.s, z5.s
    fmul   z18.s, p0/m, z18.s, z5.s
    fadd   z2.s,  p0/m, z2.s,  z8.s
    fadd   z16.s, p0/m, z16.s, z8.s
    fadd   z17.s, p0/m, z17.s, z8.s
    fadd   z18.s, p0/m, z18.s, z8.s

    frsqrte z4.s,  z2.s
    fmul    z3.s,  z4.s, z2.s
    frsqrts z3.s,  z4.s, z3.s
    fmul    z4.s,  z4.s, z3.s          // z4 = inv_rms, block 0

    frsqrte z19.s, z16.s
    fmul    z3.s,  z19.s, z16.s
    frsqrts z3.s,  z19.s, z3.s
    fmul    z19.s, z19.s, z3.s         // z19 = inv_rms, block 1

    frsqrte z20.s, z17.s
    fmul    z3.s,  z20.s, z17.s
    frsqrts z3.s,  z20.s, z3.s
    fmul    z20.s, z20.s, z3.s         // z20 = inv_rms, block 2

    frsqrte z21.s, z18.s
    fmul    z3.s,  z21.s, z18.s
    frsqrts z3.s,  z21.s, z3.s
    fmul    z21.s, z21.s, z3.s         // z21 = inv_rms, block 3

    // === Pass 2: gamma broadcast shared across the four blocks =============
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Lnrm_col_v6:
    ld1rw  {z1.s}, p0/z, [x10]
    ld1w   {z0.s},  p0/z, [x8]
    ld1w   {z24.s}, p0/z, [x8, #1, mul vl]
    ld1w   {z25.s}, p0/z, [x8, #2, mul vl]
    ld1w   {z26.s}, p0/z, [x8, #3, mul vl]
    fmul   z0.s,  p0/m, z0.s,  z1.s
    fmul   z24.s, p0/m, z24.s, z1.s
    fmul   z25.s, p0/m, z25.s, z1.s
    fmul   z26.s, p0/m, z26.s, z1.s
    fmul   z0.s,  p0/m, z0.s,  z4.s
    fmul   z24.s, p0/m, z24.s, z19.s
    fmul   z25.s, p0/m, z25.s, z20.s
    fmul   z26.s, p0/m, z26.s, z21.s
    st1w   {z0.s},  p0, [x9]
    st1w   {z24.s}, p0, [x9, #1, mul vl]
    st1w   {z25.s}, p0, [x9, #2, mul vl]
    st1w   {z26.s}, p0, [x9, #3, mul vl]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_v6

    addvl  x19, x19, #4
    addvl  x20, x20, #4
    mov    x0,  x6
    b      .Lgroup_loop_v6

    // -----------------------------------------------------------------------
    // Tail — up to 3 full blocks + a partial block, V2-style predicated loop.
    // -----------------------------------------------------------------------
.Ltail_blocks_v6:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done_v6

    mov    z2.s, #0
    mov    x8,   x19
    mov    x9,   x23

.Ltail_red_v6:
    ld1w   {z0.s}, p1/z, [x8]
    fmla   z2.s,   p1/m, z0.s, z0.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Ltail_red_v6

    fmul   z2.s, p1/m, z2.s, z5.s
    dup    z3.s, z8.s[0]
    fadd   z2.s, p1/m, z2.s, z3.s
    sel    z2.s, p1, z2.s, z3.s

    frsqrte z4.s, z2.s
    fmul    z3.s, z4.s, z2.s
    frsqrts z3.s, z4.s, z3.s
    fmul    z4.s, p1/m, z4.s, z3.s

    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Ltail_nrm_v6:
    ld1w   {z0.s}, p1/z, [x8]
    ld1rw  {z1.s}, p0/z, [x10]
    fmul   z0.s, p1/m, z0.s, z1.s
    fmul   z0.s, p1/m, z0.s, z4.s
    st1w   {z0.s}, p1, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Ltail_nrm_v6

    addvl  x19, x19, #1
    addvl  x20, x20, #1
    incw   x0
    b      .Ltail_blocks_v6

.Lblock_done_v6:
    smstop sm

.Lrow_done_v6:
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
