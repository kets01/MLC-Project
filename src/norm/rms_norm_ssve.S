// rms_norm_ssve.S — hand-written Streaming SVE RMSNorm kernel (column-major)
//
// DATA LAYOUT — column-major, explicit leading dimension:
//   element(row r, col c) lives at a[r + c*ld_a]
//   → elements of a COLUMN are contiguous (stride 1 in memory)
//   → elements of a ROW are strided (stride ld_a in memory)
//
// WHY THIS LOOP ORDER:
//   Gather loads (LD1W with vector-offset register) are RESTRICTED in
//   Streaming SVE mode without the SMEFA64 extension, which Apple has not
//   implemented.  We therefore cannot stride-load a whole row in one go.
//
//   Instead we transpose the loop order:
//     Outer loop : blocks of VL rows  (VLA: WHILELO on M, ADDVL/INCW to advance)
//     Inner loop : all N columns      (each column is contiguous → plain LD1W)
//
//   For a block of VL rows:
//     Pass 1 – walk the N columns, load VL-row slice of each column,
//              accumulate x² per row into Z2 (vector of VL sums)
//     Scalar  – VL parallel: z4[r] = 1/sqrt(z2[r]/N + ε)   (FSQRT + FDIV)
//     Pass 2 – walk the N columns again:
//              load VL-row slice of A, broadcast γ[col], multiply by z4, store
//
// KEY INSTRUCTIONS:
//   LD1W  {Zt.S}, Pg/Z, [Xn]      — load VL contiguous FP32s (stride-1)
//   LD1RW {Zt.S}, Pg/Z, [Xn]      — broadcast one FP32 scalar to all active lanes
//   ST1W  {Zt.S}, Pg,   [Xn]      — store VL contiguous FP32s
//   ADDVL Xd, Xn, #1              — Xd = Xn + SVL bytes  (advances VL rows)
//   INCW  Xd                      — Xd += VL elements     (row-block index)
//
// All instructions above are available in SSVE without SMEFA64.
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

    // -----------------------------------------------------------------------
    // Prologue — 80-byte stack frame, 16-byte aligned (extended from 64 to
    // fit a dedicated eps stash slot at [sp,#64]: streaming-mode entry
    // clobbers D8 along with D9-D15, so eps cannot simply live in S8 across
    // smstart — verified empirically; memory is the only thing that survives).
    //
    //  [sp+ 0] x19, x20  — a_base / b_base byte pointers (row-block base)
    //  [sp+16] x21, x22  — gamma ptr / M (total rows)
    //  [sp+32] x23, x24  — N (columns) / ld_a in BYTES (= ld_a_elem * 4)
    //  [sp+48] x25 / d8  — ld_b in BYTES / callee-saved FP register (unused
    //                      for eps now, kept saved/restored regardless)
    //  [sp+64] epsilon   — dedicated eps stash, stored before smstart
    // -----------------------------------------------------------------------
    stp  x19, x20, [sp, #-80]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    str  x25,      [sp, #48]
    str  d8,       [sp, #56]

    // -----------------------------------------------------------------------
    // Initialise callee-saved registers from arguments.
    //
    // x24 = ld_a * 4  (byte stride between consecutive columns in A;
    //                  also the byte distance between a[r,c] and a[r,c+1]
    //                  since a[r,c+1] = a[r + (c+1)*ld_a] = a[r,c] + ld_a elems)
    // -----------------------------------------------------------------------
    mov   x19, x0          // a_base (byte pointer to a[0,0])
    mov   x20, x1          // b_base (byte pointer to b[0,0])
    mov   x21, x2          // gamma  (fixed, 1-D contiguous N-element array)
    mov   x22, x3          // M      (total rows)
    mov   x23, x4          // N      (total columns / feature dimension)
    lsl   x24, x5, #2      // ld_a_bytes = ld_a * 4
    lsl   x25, x6, #2      // ld_b_bytes = ld_b * 4
    fmov  s8,  s0          // kept for consistency (not relied on across smstart)
    str   s0,  [sp, #64]   // dedicated eps stash, stored BEFORE smstart

    cbz   x22, .Lrow_done

    // -----------------------------------------------------------------------
    // Enter streaming SVE mode.
    // P0 = all-true predicate used for LD1RW (gamma broadcast) which always
    //      loads a single scalar and replicates it to all lanes.
    // -----------------------------------------------------------------------
    smstart sm
    ptrue  p0.s, all

    // x0 = row-block start index (element index into M, used by WHILELO/INCW)
    mov   x0, #0

    // -----------------------------------------------------------------------
    // Outer loop — one iteration per block of VL rows.
    // -----------------------------------------------------------------------
.Lblock_loop:
    // P1 = predicate for this row block:
    //   lane i is active iff (x0 + i) < M.
    //   Handles both full blocks and the final partial block automatically.
    whilelo  p1.s, x0, x22
    b.none   .Lblock_done

    // === Pass 1: sum of squares ============================================
    //
    // Z2 = per-row accumulator.  Lane i holds the running sum of squares for
    // row (x0 + i) across all N columns processed so far.
    //
    // We walk each column in turn.  For column c, the M values
    //   a[0 + c*ld_a], a[1 + c*ld_a], ..., a[M-1 + c*ld_a]
    // are contiguous in memory, so a single LD1W loads VL of them at once.
    //
    // x8  = running byte pointer for column c of A (a_base + c * ld_a_bytes)
    // x9  = column countdown (N → 0)
    mov    z2.s, #0            // accumulator = 0.0 in all lanes
    mov    x8,   x19           // x8 = &a[x0_start_of_block, col=0]
    mov    x9,   x23           // column countdown = N

.Lred_col_loop:
    ld1w   {z0.s}, p1/z, [x8]             // load VL rows of column (contiguous)
    fmla   z2.s,   p1/m, z0.s, z0.s       // z2[i] += z0[i]²  (merge pred keeps inactive=0)
    add    x8,  x8,  x24                   // x8 += ld_a_bytes  → next column
    subs   x9,  x9,  #1
    b.ne   .Lred_col_loop

    // === Compute inv_rms for all VL rows in this block =====================
    //
    // After the loop: z2[i] = Σ_c a[r_i + c*ld_a]²  for the M(≤VL) active rows.
    // Inactive lanes contain 0.0 (merge predication kept them there).
    //
    // Steps (all predicated with P1/M so inactive lanes stay as-is):
    //   1. z2 /= N  →  mean square per row
    //   2. z2 += ε  →  mean square + epsilon
    //   3. z2 = √z2 →  rms per row
    //   4. z4 = 1/z2 → inv_rms per row
    //
    // Broadcasting N into a Z register via a scratch scalar:
    //   scvtf s0, x23  writes float(N) to S0 (= Z0.S[0]).
    //   dup z3.s, z0.s[0]  replicates it to all lanes of Z3.
    scvtf  s0, x23                     // s0 = float(N)
    dup    z3.s, z0.s[0]               // z3 = {N, N, …}
    fdiv   z2.s, p1/m, z2.s, z3.s     // z2[i] = sumsq[i] / N

    // Broadcast epsilon, reloaded from the dedicated stack slot each block
    // (S8/D8 does not survive smstart, so it cannot be read from Z8 here):
    ldr    s0, [sp, #64]
    dup    z3.s, z0.s[0]               // z3 = {ε, ε, …}
    fadd   z2.s, p1/m, z2.s, z3.s     // z2[i] = sumsq[i]/N + ε

    // Square root then reciprocal (both predicated; inactive lanes unchanged):
    fsqrt  z2.s, p1/m, z2.s           // z2[i] = rms[i]
    fmov   z4.s, #1.0                  // z4 = 1.0 in ALL lanes (unpredicated ok;
                                        //   inactive lanes hold 1.0 but are never stored)
    fdiv   z4.s, p1/m, z4.s, z2.s     // z4[i] = 1.0 / rms[i] = inv_rms[i]

    // === Pass 2: normalize + scale =========================================
    //
    // For column c:
    //   y[r + c*ld_b] = a[r + c*ld_a] · γ[c] · inv_rms[r]
    //
    // x8  = running pointer for A column (reset to row-block base)
    // x9  = running pointer for B column
    // x10 = running pointer into gamma (advances by 4 bytes per column)
    // x11 = column countdown
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

    // -----------------------------------------------------------------------
    // Advance to next row block.
    //
    // ADDVL Xd, Xn, #1 adds SVL bytes to Xn.
    // SVL = 64 bytes on M4 = VL elements * 4 bytes = 16 * 4 = 64. ✓
    // After this, x19 points to &a[row_block_start + VL].
    //
    // INCW x0 advances the element-index by VL (= SVL/4).
    // -----------------------------------------------------------------------
    addvl  x19, x19, #1        // a_base += SVL bytes  (next VL rows of A)
    addvl  x20, x20, #1        // b_base += SVL bytes
    incw   x0                  // row-block index += VL
    b      .Lblock_loop

.Lblock_done:
    smstop sm

.Lrow_done:
    // -----------------------------------------------------------------------
    // Epilogue — restore in reverse order.
    // -----------------------------------------------------------------------
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
