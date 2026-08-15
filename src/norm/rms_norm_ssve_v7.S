// rms_norm_ssve_v7.S — RMSNorm V7: V6 with SME2 multi-vector accesses.
//
// V6's group loop already touches four consecutive VL-row blocks per column
// at [x8], [x8,#1,MUL VL], [x8,#2,MUL VL], [x8,#3,MUL VL] — exactly the
// operand shape of SME2's 4-vector contiguous access.  So V7 is V6 with
//   4x LD1W -> ld1w {z0.s-z3.s}, pn8/z, [x8]
//   4x ST1W -> st1w {z0.s-z3.s}, pn8, [x9]
// Same addresses, same traffic, same arithmetic, same summation order: only
// the instruction count changes, which makes it a clean test of whether V6 is
// issue-bound or memory-bound.  The correctness gate is therefore BIT-IDENTITY
// with V6, not a tolerance.  Measured +17% in the DRAM regime, ~0 in cache.
//
// NO TAIL PATH.  V6 needs one because its group path is unpredicated, and that
// tail touches 64 B per column instead of 256 B — the exact low access density
// V6 exists to fix.  SME2's predicate-as-counter removes it: `whilelo pn8.s,
// x0, x22, vlx4` describes min(M - x0, 4*VL) active elements ACROSS the group,
// so one loop body serves full and final partial groups alike.
//
// Inactive-lane safety: loads are pn8/z, so inactive lanes read 0, contribute
// 0 to the sum of squares, and a fully-inactive block ends at 0*(1/N)+eps,
// which FRSQRTE handles finitely.  The store uses the same counter, so nothing
// invalid reaches memory and no SEL is needed.
//
// Registers: z0-z3 data quad (must be a consecutive 4-aligned group), z4-z7
// accumulators (one per row block), z8 eps, z16 gamma, z17 1/N, z18 NR
// scratch, z20-z23 inv_rms, p0 all-true, pn8 the VLx4 row counter.
//
// Requires FEAT_SME2; the C++ bridge dispatches to V6 when it is absent.
//
// C signature: identical to rms_norm_ssve (rms_norm_ssve.S).
//   (a, b, gamma, m, n, ld_a, ld_b) in x0-x6, eps in s0.

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
