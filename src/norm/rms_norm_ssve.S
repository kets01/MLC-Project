// rms_norm_ssve.S — RMSNorm V0: hand-written Streaming SVE baseline.
//
// Layout: column-major, a[r + c*ld_a] — a column is contiguous, a row is
// strided by ld_a.
//
// The loop order is transposed (outer = blocks of VL rows, inner = all N
// columns) because gather loads are unavailable in streaming mode without
// FEAT_SME_FA64, which Apple does not implement.  A row therefore cannot be
// stride-loaded, but a VL-row slice of one column is a single plain LD1W.
//
// Per VL-row block: pass 1 accumulates x^2 per row; inv_rms = 1/sqrt(mean+eps)
// via FSQRT + FDIV; pass 2 re-reads x, applies gamma[col] and inv_rms, stores.
// VLA (decision D): WHILELO builds the row predicate, ADDVL/INCW advance.
//
// C signature:
//   void rms_norm_ssve(const float *a,     // x0
//                      float       *b,     // x1
//                      const float *gamma, // x2
//                      int64_t      m,     // x3
//                      int64_t      n,     // x4
//                      int64_t      ld_a,  // x5  (elements, not bytes)
//                      int64_t      ld_b,  // x6  (elements, not bytes)
//                      float        eps);  // s0

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(rms_norm_ssve)

.text
.align 4

SYMBOL(rms_norm_ssve):

    // Prologue — 128-byte frame:
    //   [sp+0..48] x19-x25   [sp+56..112] d8-d15   [sp+120] eps
    //
    // AAPCS64 makes d8-d15 callee-saved, and smstart/smstop zero all eight
    // (they alias the low bits of z8-z15), so all eight are saved whether or
    // not this kernel names them.  eps goes to memory for the same reason: no
    // register survives the PSTATE.SM transition.
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

    mov   x19, x0          // a_base (byte pointer to a[0,0])
    mov   x20, x1          // b_base (byte pointer to b[0,0])
    mov   x21, x2          // gamma  (fixed, 1-D contiguous N-element array)
    mov   x22, x3          // M      (total rows)
    mov   x23, x4          // N      (total columns / feature dimension)
    lsl   x24, x5, #2      // ld_a_bytes = ld_a * 4
    lsl   x25, x6, #2      // ld_b_bytes = ld_b * 4
    fmov  s8,  s0          // kept for consistency (not relied on across smstart)
    str   s0,  [sp, #120]   // dedicated eps stash, stored BEFORE smstart

    cbz   x22, .Lrow_done

    // p0 = all-true; governs the LD1RW gamma broadcast.
    smstart sm
    ptrue  p0.s, all

    // x0 = row-block start index, used by WHILELO/INCW
    mov   x0, #0

    // Outer loop, one block of VL rows per iteration.  WHILELO activates lane
    // i iff (x0 + i) < M, so full and final partial blocks share one path.
.Lblock_loop:
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done

    // === Pass 1: sum of squares ============================================
    // z2 lane i accumulates row (x0+i); one LD1W per column grabs VL
    // contiguous rows of it.  x8 = column pointer, x9 = column countdown.
    mov    z2.s, #0            // accumulator = 0.0 in all lanes
    mov    x8,   x19           // x8 = &a[x0_start_of_block, col=0]
    mov    x9,   x23           // column countdown = N

.Lred_col_loop:
    ld1w   {z0.s}, p1/z, [x8]             // load VL rows of column (contiguous)
    fmla   z2.s,   p1/m, z0.s, z0.s       // z2[i] += z0[i]²  (merge pred keeps inactive=0)
    add    x8,  x8,  x24                   // x8 += ld_a_bytes  → next column
    subs   x9,  x9,  #1
    b.ne   .Lred_col_loop

    // === inv_rms = 1/sqrt(sumsq/N + eps), per lane =========================
    // Every step is p1/m, so inactive lanes keep the zeros pass 1 left them.
    scvtf  s0, x23                     // s0 = float(N)
    dup    z3.s, z0.s[0]               // z3 = {N, N, …}
    fdiv   z2.s, p1/m, z2.s, z3.s     // z2[i] = sumsq[i] / N

    // eps is reloaded from memory: S8/D8 does not survive smstart.
    ldr    s0, [sp, #120]
    dup    z3.s, z0.s[0]               // z3 = {ε, ε, …}
    fadd   z2.s, p1/m, z2.s, z3.s     // z2[i] = sumsq[i]/N + ε

    fsqrt  z2.s, p1/m, z2.s           // z2[i] = rms[i]
    fmov   z4.s, #1.0                  // z4 = 1.0 in ALL lanes (unpredicated ok;
                                        //   inactive lanes hold 1.0 but are never stored)
    fdiv   z4.s, p1/m, z4.s, z2.s     // z4[i] = 1.0 / rms[i] = inv_rms[i]

    // === Pass 2: y[r,c] = a[r,c] * gamma[c] * inv_rms[r] ===================
    mov    x8,  x19            // &a[row_block, col=0]
    mov    x9,  x20            // &b[row_block, col=0]
    mov    x10, x21            // &gamma[0]
    mov    x11, x23            // countdown = N

.Lnrm_col_loop:
    ld1w   {z0.s}, p1/z, [x8]          // load VL rows of A column c
    ld1rw  {z1.s}, p0/z, [x10]         // broadcast gamma[c] to all VL lanes

    fmul   z0.s, p1/m, z0.s, z1.s      // z0[i] = a[r_i, c] * gamma[c]
    fmul   z0.s, p1/m, z0.s, z4.s      // z0[i] *= inv_rms[r_i]

    st1w   {z0.s}, p1, [x9]            // store VL rows of B column c

    add    x8,  x8,  x24               // A col ptr: next column
    add    x9,  x9,  x25               // B col ptr: next column
    add    x10, x10, #4                // gamma ptr: next element
    subs   x11, x11, #1
    b.ne   .Lnrm_col_loop

    // Next row block.  ADDVL adds SVL bytes, INCW adds VL elements: both VLA.
    addvl  x19, x19, #1        // a_base += SVL bytes  (next VL rows of A)
    addvl  x20, x20, #1        // b_base += SVL bytes
    incw   x0                  // row-block index += VL
    b      .Lblock_loop

.Lblock_done:
    smstop sm

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
