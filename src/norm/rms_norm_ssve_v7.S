// rms_norm_ssve_v7.S — Sprint 6 ablation V7 (SME2 multi-vector load/store)
//
// THE HARDWARE CORRECTION THIS VARIANT RESTS ON: the project docs assumed
// SME1 throughout (context.md §5 "confirm against the M4's SME level"), but
// `sysctl hw.optional.arm.FEAT_SME2` reports 1 on the target M4.  Before any
// kernel was written, the 4-vector LD1W/ST1W forms were assembled AND
// executed inside a streaming region on the machine, so this is built on a
// verified capability rather than a datasheet reading (CLAUDE.md §10).
//
// THE LEVER.  V6's group loop already touches four consecutive VL-row blocks
// per column, addressed as [x8], [x8,#1,MUL VL], [x8,#2,MUL VL],
// [x8,#3,MUL VL] — that is *precisely* the operand shape of SME2's 4-vector
// contiguous load.  So V7 is V6 with:
//     4x LD1W  ->  1x  ld1w {z0.s-z3.s}, pn8/z, [x8]
//     4x ST1W  ->  1x  st1w {z0.s-z3.s}, pn8, [x9]
// Identical memory traffic, identical access pattern, identical arithmetic:
// the ONLY thing that changes is instruction count in the hot loops (pass 1
// drops 3 loads/column, pass 2 drops 3 loads + 3 stores/column).  That makes
// this a clean single-variable test of one question: is V6 instruction-issue
// bound, or memory bound?
//
// THE HYPOTHESIS, WRITTEN BEFORE MEASURING (ROADMAP Sprint 6 pre-registers
// the expectation that SME2 "does not raise the DRAM roofline"):
//   - True-DRAM shapes (256 MiB): V6 sustains ~20 GiB/s useful = ~30 GiB/s
//     moved = ~51% of the 59.5 GiB/s streaming roofline, and Sprint 2b showed
//     the binding constraint there is ACCESS DENSITY, not issue rate.  Folding
//     4 instructions into 1 should therefore buy ~0.
//   - Cache-resident shapes: V6 reaches ~38.6 GiB/s useful = ~58 GiB/s moved
//     = ~97% of the streaming roofline.  There is almost nothing left to win;
//     at best a small gain where per-column loop overhead still shows.
//   Expected verdict: ≈0 everywhere, i.e. a *confirmation* that V6 is memory-
//   bound rather than issue-bound.  A measured ≈0 here is a positive result:
//   it closes the instruction-issue family the way V5 closed the
//   outstanding-loads family, and it is the evidence for the report's claim
//   that the remaining gap is DRAM behaviour, not code density.
//
// NO TAIL PATH (Sprint 6 revision).  V6 needs a separate 1-3 block tail
// because its group path is unpredicated: whatever does not fill a group of
// four VL-row blocks falls into a single-block predicated loop that touches
// only 64 B per column instead of 256 B — i.e. it reverts to exactly the low
// access density V6 exists to fix.  That is a real cliff, not a rounding
// error: with OpenMP row-chunking it made 6- and 10-thread runs ~45 % slower
// than 4 and 8, because those chunk sizes are not multiples of 4*VL.
//
// SME2's predicate-as-counter removes the need for the second path entirely.
// `whilelo pn8.s, x0, x22, vlx4` describes min(M - x0, 4*VL) active elements
// ACROSS the four-vector group, so one loop body serves both the full groups
// (all-true counter) and the final partial group (partial counter).  The tail
// is not "multi-vector" — it is gone, and every column touch in the kernel is
// a 256 B multi-vector access.
//
// Inactive-lane safety: the loads are pn8/z (zeroing), so inactive lanes read
// 0, contribute 0 to the sum of squares, and a fully-inactive block ends with
// acc = 0 -> 0*(1/N) + eps = eps, which FRSQRTE handles finitely.  The store
// is predicated by the same counter, so nothing invalid reaches memory and no
// SEL is needed.
//
// The C++ bridge dispatches to V6 when FEAT_SME2 is absent, so the same
// binary still runs on M1/M2 (ROADMAP: "degrades to the SME1 kernel").
//
// Register map (group path):
//   z0-z3   data quad (must be a consecutive, 4-aligned Z group)
//   z4-z7   sum-of-squares accumulators, one per row block (V6's ILP, kept)
//   z8      {eps,...}          z16  gamma broadcast
//   z17     {1/N,...}          z18  Newton-Raphson scratch
//   z20-z23 inv_rms per block
//   p0      all-true SVE predicate (governs FP ops and the gamma broadcast)
//   pn8     row counter predicate, recomputed per group by WHILELO ...,VLx4
//
// C signature (identical to V0-V6):
//   void rms_norm_ssve_v7(const float *a,     // x0
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

.arch armv9-a+sme2

.global SYMBOL(rms_norm_ssve_v7)

.text
.align 4

SYMBOL(rms_norm_ssve_v7):

    // Prologue — 128-byte frame saving the FULL d8-d15 callee-saved range
    // (AAPCS64): smstart/smstop zero all of them.  Same layout as V6.
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
    lsl   x24, x5, #2          // stride_a in bytes
    lsl   x25, x6, #2          // stride_b in bytes
    str   s0,  [sp, #120]      // dedicated eps stash, stored BEFORE smstart
                                // (no register survives the PSTATE.SM change)

    cbz   x22, .Lrow_done_v7

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #120]
    dup    z8.s, z0.s[0]       // z8 = {eps, ...}

    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1
    dup   z17.s, z0.s[0]       // z17 = {1/N, ...}

    cntw  x14
    lsl   x14, x14, #2         // x14 = 4*VL rows per group
    mov   x0, #0               // x0 = row index

    // -----------------------------------------------------------------------
    // Single loop — four VL-row blocks per iteration, the last one partial.
    // WHILELO ...,VLx4 makes the group predicate all-true while >= 4*VL rows
    // remain and partial for the final group, so there is no separate tail.
    // -----------------------------------------------------------------------
.Lgroup_loop_v7:
    whilelo pn8.s, x0, x22, vlx4
    b.none  .Lblock_done_v7    // no rows left

    // === Pass 1: sum of squares, ONE 4-vector load per column ==============
    mov    z4.s, #0            // accumulator, block 0
    mov    z5.s, #0            // accumulator, block 1
    mov    z6.s, #0            // accumulator, block 2
    mov    z7.s, #0            // accumulator, block 3
    mov    x8,  x19
    mov    x9,  x23

.Lred_col_v7:
    ld1w   {z0.s - z3.s}, pn8/z, [x8]   // 4 blocks in one instruction
    fmla   z4.s, p0/m, z0.s, z0.s
    fmla   z5.s, p0/m, z1.s, z1.s
    fmla   z6.s, p0/m, z2.s, z2.s
    fmla   z7.s, p0/m, z3.s, z3.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_col_v7

    // === inv_rms for the four blocks (FRSQRTE + one Newton-Raphson step) ===
    fmul   z4.s, p0/m, z4.s, z17.s
    fmul   z5.s, p0/m, z5.s, z17.s
    fmul   z6.s, p0/m, z6.s, z17.s
    fmul   z7.s, p0/m, z7.s, z17.s
    fadd   z4.s, p0/m, z4.s, z8.s
    fadd   z5.s, p0/m, z5.s, z8.s
    fadd   z6.s, p0/m, z6.s, z8.s
    fadd   z7.s, p0/m, z7.s, z8.s

    frsqrte z20.s, z4.s
    fmul    z18.s, z20.s, z4.s
    frsqrts z18.s, z20.s, z18.s
    fmul    z20.s, z20.s, z18.s        // z20 = inv_rms, block 0

    frsqrte z21.s, z5.s
    fmul    z18.s, z21.s, z5.s
    frsqrts z18.s, z21.s, z18.s
    fmul    z21.s, z21.s, z18.s        // z21 = inv_rms, block 1

    frsqrte z22.s, z6.s
    fmul    z18.s, z22.s, z6.s
    frsqrts z18.s, z22.s, z18.s
    fmul    z22.s, z22.s, z18.s        // z22 = inv_rms, block 2

    frsqrte z23.s, z7.s
    fmul    z18.s, z23.s, z7.s
    frsqrts z18.s, z23.s, z18.s
    fmul    z23.s, z23.s, z18.s        // z23 = inv_rms, block 3

    // === Pass 2: one 4-vector load + one 4-vector store per column =========
    mov    x8,  x19
    mov    x9,  x20
    mov    x10, x21
    mov    x11, x23

.Lnrm_col_v7:
    ld1rw  {z16.s}, p0/z, [x10]         // gamma[col], shared by all 4 blocks
    ld1w   {z0.s - z3.s}, pn8/z, [x8]
    fmul   z0.s, p0/m, z0.s, z16.s
    fmul   z1.s, p0/m, z1.s, z16.s
    fmul   z2.s, p0/m, z2.s, z16.s
    fmul   z3.s, p0/m, z3.s, z16.s
    fmul   z0.s, p0/m, z0.s, z20.s
    fmul   z1.s, p0/m, z1.s, z21.s
    fmul   z2.s, p0/m, z2.s, z22.s
    fmul   z3.s, p0/m, z3.s, z23.s
    st1w   {z0.s - z3.s}, pn8, [x9]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_v7

    addvl  x19, x19, #4
    addvl  x20, x20, #4
    add    x0,  x0, x14
    b      .Lgroup_loop_v7

.Lblock_done_v7:
    smstop sm

.Lrow_done_v7:
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
