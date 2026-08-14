// rms_norm_ssve_v7x2.S — Sprint 6 CONTROL for V7 (2-vector instead of 4)
//
// PURPOSE: discriminate between the two explanations for V7's +17% DRAM win.
// V7 folds four single-vector accesses into ONE 4-vector access (256 B per
// load instruction).  This control does the identical work with TWO 2-vector
// accesses (128 B per load instruction).  Everything else is held constant:
// same four row-blocks per group, same 256 B of data touched per column, same
// addresses, same arithmetic, same summation order, no tail path.  The ONLY
// variable is bytes-of-demand exposed per load instruction.
//
//   V6      : 4 x 64 B  per column   (4 instructions)
//   V7x2    : 2 x 128 B per column   (2 instructions)   <- this kernel
//   V7      : 1 x 256 B per column   (1 instruction)
//
// PREDICTIONS, WRITTEN BEFORE MEASURING:
//   - If the win comes from RELIEVED MEMORY-LEVEL PARALLELISM (each in-flight
//     load instruction exposing more demand to the memory system), the gain
//     should scale with bytes-per-instruction, so V7x2 should land roughly
//     HALFWAY between V6 and V7.
//   - If the win comes from the memory system specifically liking one 256 B
//     burst (prefetcher/coalescing), V7x2 at 128 B should give LITTLE OR
//     NOTHING over V6.
// The two hypotheses predict measurably different results, which is the whole
// point of building the control rather than arguing about mechanism.
//
// Predication: blocks 0-1 are governed by a VLx2 counter built from the row
// index, blocks 2-3 by a second VLx2 counter built from row index + 2*VL.
// Together they cover exactly the rows the single VLx4 counter covers in V7,
// so partial groups behave identically and there is still no tail.
//
// This is a MEASUREMENT variant: it is not wired into the JIT and is not a
// candidate incumbent.

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme2

.global SYMBOL(rms_norm_ssve_v7x2)

.text
.align 4

SYMBOL(rms_norm_ssve_v7x2):

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

    cbz   x22, .Lrow_done_v7x2

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
    lsr   x15, x14, #1         // x15 = 2*VL (offset of blocks 2-3)
    mov   x0, #0               // x0 = row index

    // -----------------------------------------------------------------------
    // Single loop — four VL-row blocks per iteration, the last one partial.
    // WHILELO ...,VLx4 makes the group predicate all-true while >= 4*VL rows
    // remain and partial for the final group, so there is no separate tail.
    // -----------------------------------------------------------------------
.Lgroup_loop_v7x2:
    whilelo pn8.s, x0, x22, vlx2   // counter for blocks 0-1
    b.none  .Lblock_done_v7x2      // no rows left at all
    add     x13, x0, x15           // first row of blocks 2-3
    whilelo pn9.s, x13, x22, vlx2  // counter for blocks 2-3

    // === Pass 1: sum of squares, ONE 4-vector load per column ==============
    mov    z4.s, #0            // accumulator, block 0
    mov    z5.s, #0            // accumulator, block 1
    mov    z6.s, #0            // accumulator, block 2
    mov    z7.s, #0            // accumulator, block 3
    mov    x8,  x19
    mov    x9,  x23

.Lred_col_v7x2:
    ld1w   {z0.s - z1.s}, pn8/z, [x8]              // blocks 0-1, 128 B
    ld1w   {z2.s - z3.s}, pn9/z, [x8, #2, mul vl]  // blocks 2-3, 128 B
    fmla   z4.s, p0/m, z0.s, z0.s
    fmla   z5.s, p0/m, z1.s, z1.s
    fmla   z6.s, p0/m, z2.s, z2.s
    fmla   z7.s, p0/m, z3.s, z3.s
    add    x8,  x8,  x24
    subs   x9,  x9,  #1
    b.ne   .Lred_col_v7x2

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

.Lnrm_col_v7x2:
    ld1rw  {z16.s}, p0/z, [x10]         // gamma[col], shared by all 4 blocks
    ld1w   {z0.s - z1.s}, pn8/z, [x8]
    ld1w   {z2.s - z3.s}, pn9/z, [x8, #2, mul vl]
    fmul   z0.s, p0/m, z0.s, z16.s
    fmul   z1.s, p0/m, z1.s, z16.s
    fmul   z2.s, p0/m, z2.s, z16.s
    fmul   z3.s, p0/m, z3.s, z16.s
    fmul   z0.s, p0/m, z0.s, z20.s
    fmul   z1.s, p0/m, z1.s, z21.s
    fmul   z2.s, p0/m, z2.s, z22.s
    fmul   z3.s, p0/m, z3.s, z23.s
    st1w   {z0.s - z1.s}, pn8, [x9]
    st1w   {z2.s - z3.s}, pn9, [x9, #2, mul vl]
    add    x8,  x8,  x24
    add    x9,  x9,  x25
    add    x10, x10, #4
    subs   x11, x11, #1
    b.ne   .Lnrm_col_v7x2

    addvl  x19, x19, #4
    addvl  x20, x20, #4
    add    x0,  x0, x14
    b      .Lgroup_loop_v7x2

.Lblock_done_v7x2:
    smstop sm

.Lrow_done_v7x2:
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
