// layer_norm_ssve.S — Hand-written SSVE LayerNorm kernel (Sprint 2)
//
// Formula: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
// Layout:  column-major — element [row, col] lives at  base + row + col*ld
//
// KEY INSIGHT: in column-major layout, a COLUMN is contiguous in memory.
// Scatter-gather (LD1W with vector offsets) is NOT in the SME1 mandatory
// SSVE subset.  We therefore vectorise ACROSS ROWS, looping OVER COLUMNS:
//
//   Outer loop: row chunks of SVL rows at a time
//   Inner loop: all N columns
//   Each inner load: one contiguous LD1W = SVL consecutive rows of one column
//
// This gives three passes per row chunk (not per row):
//   Pass 1 : for each col, load SVL rows → accumulate into per-row sum z8
//            → divide by N → per-row mean in z8
//   Pass 2 : for each col, load, subtract mean, square → per-row var in z9
//            → divide by N, add eps, sqrt, reciprocal → per-row inv_std in z5
//   Pass 3 : for each col, load, normalise, scale (gamma), shift (beta) → store
//
// VLA: SVL rows are processed per outer iteration.  WHILELO builds the
// row predicate so the code runs correctly on any SVL.

#ifdef __APPLE__
    #define SYM(s) _##s          // macOS prepends an underscore to C symbols
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve)
.text
.align 2

// ─── Calling convention (AArch64) ────────────────────────────────────────────
//  x0  = a       (input  matrix, column-major, leading dim ld_a)
//  x1  = b       (output matrix, column-major, leading dim ld_b)
//  x2  = gamma   (scale vector, N floats, contiguous)
//  x3  = beta    (shift vector, N floats, contiguous)
//  x4  = m       (number of rows)
//  x5  = n       (feature size, i.e. length of the normalized axis)
//  x6  = ld_a    (leading dimension of a)
//  x7  = ld_b    (leading dimension of b)
//  s0  = epsilon (first FP argument register)
//
// ─── Register map ────────────────────────────────────────────────────────────
//  x8         SVL in FP32 elements (constant, from CNTW)
//  x9         r_base — outer loop: first row of the current chunk
//  x10        walking column pointer in a (inner loops)
//  x11        col index (inner loops)
//  x12        walking pointer in b   (pass 3)
//  x13        walking pointer in gamma (pass 3)
//  x14        stride_col_a = ld_a * 4  (bytes to advance to next column in a)
//  x15        stride_col_b = ld_b * 4
//  x16        SVL * 4 (bytes per row chunk — how far r_base advances each iter)
//  x17        walking pointer in beta  (pass 3)
//  x19        a_chunk_base = a + r_base*4  (callee-saved; advances each outer iter)
//  x20        b_chunk_base = b + r_base*4  (callee-saved)
//
//  z0         data register (reused across passes / columns)
//  z1         gamma broadcast for current column (pass 3)
//  z2         beta  broadcast for current column (pass 3)
//  z4         per-row mean   (result of pass 1)
//  z5         per-row inv_std (result of pass 2)
//  z6         [n, n, ..., n]   — divisor for mean and variance
//  z7         [eps, eps, ...]  — epsilon
//  z8         per-row sum accumulator (pass 1); reused as mean after div
//  z9         per-row var accumulator (pass 2); reused as sqrt(var+eps)
//
//  p0         all-true predicate (not used directly — p1 covers full chunks too)
//  p1         row predicate: p1[j] = 1 iff (r_base + j) < m
//             WHILELO makes p1 all-true for full chunks and partial for the tail

SYM(layer_norm_ssve):

// ── Prologue ──────────────────────────────────────────────────────────────────
// Stack frame (48 bytes, 16-byte aligned):
//   [sp+ 0] x29 (frame pointer)
//   [sp+ 8] x30 (link register)
//   [sp+16] x19 (callee-saved: a_chunk_base)
//   [sp+24] x20 (callee-saved: b_chunk_base)
//   [sp+32] epsilon (float, saved here before smstart)
    stp  x29, x30, [sp, #-48]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]

// Save epsilon BEFORE smstart — entering streaming mode can alter FP state.
    str  s0, [sp, #32]

// ── Enter Streaming SVE mode ──────────────────────────────────────────────────
// SMSTART SM activates PSTATE.SM.  From here, SVE instructions operate at
// the streaming vector length (SVL) instead of the normal VL.
    smstart sm

// ── One-time setup ────────────────────────────────────────────────────────────
    cntw x8                     // x8 = SVL/4 = number of 32-bit lanes per vector

// stride_col_a: bytes to step from column c to column c+1 in a.
// Column c starts at a + c*ld_a*4; the next column is ld_a floats = ld_a*4 bytes later.
    lsl  x14, x6, #2            // x14 = ld_a * 4  (stride_col_a)
    lsl  x15, x7, #2            // x15 = ld_b * 4  (stride_col_b)
    lsl  x16, x8, #2            // x16 = SVL * 4   (row-chunk stride in bytes)

// Reload epsilon and broadcast to all lanes (z7 = [eps, eps, ..., eps]).
// We use DUP Zd.S, Zn.S[0] which copies lane 0 of Zn to all lanes of Zd.
// s22 aliases z22 lane 0, so loading s22 then duplicating works.
    ldr  s22, [x29, #32]
    dup  z7.s, z22.s[0]         // z7 = [eps, ..., eps]

// Broadcast n as FP32 for the vector divide (z6 = [n, n, ..., n]).
    scvtf s21, x5               // s21 = (float)n
    dup  z6.s, z21.s[0]         // z6 = [n, ..., n]

// Initialise outer loop: process rows in chunks of SVL.
    mov  x9,  xzr              // x9  = r_base = 0
    mov  x19, x0               // x19 = a_chunk_base (a + 0*4 = a initially)
    mov  x20, x1               // x20 = b_chunk_base (b + 0*4 = b initially)

// ── Outer loop: one iteration per SVL-row chunk ───────────────────────────────
Louter:
    cmp  x9, x4                 // exit when r_base >= m
    b.ge Louter_done

// Build row predicate p1.
// WHILELO p1.s, Xn, Xm sets p1[j] = 1  iff  (Xn + j) < Xm.
// With Xn=r_base, Xm=m:
//   full chunk (r_base + SVL <= m): p1 becomes all-ones
//   tail chunk (r_base + SVL >  m): p1 has 1s only for rows that exist
// Using p1 everywhere means no branch on "full vs tail" is needed.
    whilelo p1.s, x9, x4       // p1 = row predicate for this chunk

// ════════════════════════════════════════════════════════════════════════════
// PASS 1 — accumulate column sums into per-row vector z8
//
// For each column c: load SVL consecutive rows of column c (contiguous in
// memory), add element-wise to z8.  After N columns z8[r] = sum_c a[r_base+r, c].
// Then divide by N to get per-row mean.
// ════════════════════════════════════════════════════════════════════════════

// FMOV Zd.S, #imm broadcasts the FP immediate to ALL lanes (unpredicated).
    fmov z8.s, #0.0             // z8 = per-row sum accumulator = 0

    mov  x10, x19               // x10 = walking col ptr = a_chunk_base (col 0)
    mov  x11, xzr               // x11 = col index = 0

Lp1col:
    cmp  x11, x5
    b.ge Lp1col_done

// LD1W {Zt.S}, Pg/Z, [Xn] — load SVL consecutive floats from [x10].
// /Z means: inactive lanes (p1[j]=0) get 0, so they do not corrupt the sum.
// x10 points at a[r_base, col]; consecutive addresses hold a[r_base+1, col], etc.
// That is exactly the contiguous column slice we want.
    ld1w {z0.s}, p1/z, [x10]
    fadd z8.s, p1/m, z8.s, z0.s // z8[r] += a[r_base+r, col]

    add  x10, x10, x14          // advance to next column: x10 += stride_col_a
    add  x11, x11, #1
    b    Lp1col

Lp1col_done:
// z8 = per-row sums.  Divide by n to get per-row mean.
// FDIV Zdn.S, Pg/M, Zdn.S, Zm.S — destination = first source (in-place).
// Inactive lanes (p1[j]=0) retain their current value (0 from init + /z loads).
    fdiv z8.s, p1/m, z8.s, z6.s // z8 = sum / n = per-row mean

// ════════════════════════════════════════════════════════════════════════════
// PASS 2 — compute per-row variance and derive inv_std
//
// z9[r] = sum_c (a[r_base+r, c] - mean[r])^2  /  N
// inv_std[r] = 1 / sqrt(z9[r] + eps)
// ════════════════════════════════════════════════════════════════════════════

    fmov z9.s, #0.0             // z9 = per-row variance accumulator = 0

    mov  x10, x19               // reset walking col ptr
    mov  x11, xzr

Lp2col:
    cmp  x11, x5
    b.ge Lp2col_done

    ld1w {z0.s}, p1/z, [x10]

// Subtract per-row mean (z8) from loaded values.
// p1/m: only active rows are updated; inactive lanes stay as loaded (0 from /z).
    fsub z0.s, p1/m, z0.s, z8.s // z0 = x - mean

// Square in place: (x - mean)^2
    fmul z0.s, p1/m, z0.s, z0.s

// Accumulate into variance sum.
    fadd z9.s, p1/m, z9.s, z0.s

    add  x10, x10, x14
    add  x11, x11, #1
    b    Lp2col

Lp2col_done:
// z9 = sum of squared deviations.  Divide by n → variance.
    fdiv z9.s, p1/m, z9.s, z6.s // z9 = variance

// Add epsilon (numerical stability for constant rows).
    fadd z9.s, p1/m, z9.s, z7.s // z9 = var + eps

// FSQRT Zd.S, Pg/M, Zn.S — vector square root.
    fsqrt z9.s, p1/m, z9.s      // z9 = sqrt(var + eps)

// Compute per-row inv_std = 1 / z9.
// Load 1.0 into z5 (all lanes); then divide by z9.
    fmov z5.s, #1.0             // z5 = [1.0, ..., 1.0]  (unpredicated)
    fdiv z5.s, p1/m, z5.s, z9.s // z5 = 1 / sqrt(var + eps) = inv_std

// ════════════════════════════════════════════════════════════════════════════
// PASS 3 — normalise, scale (gamma), shift (beta), and write output
//
// For each column:
//   z0 = a[r_chunk, col]
//   z0 = (z0 - mean) * inv_std * gamma[col] + beta[col]
//   store z0 → b[r_chunk, col]
//
// gamma and beta are 1-D contiguous arrays: one scalar per column.
// We broadcast each scalar to the whole vector with LD1RW.
// ════════════════════════════════════════════════════════════════════════════

    mov  x10, x19               // x10 = walking col ptr in a
    mov  x12, x20               // x12 = walking col ptr in b
    mov  x13, x2                // x13 = gamma walking ptr
    mov  x17, x3                // x17 = beta  walking ptr
    mov  x11, xzr

Lp3col:
    cmp  x11, x5
    b.ge Lp3col_done

    ld1w {z0.s}, p1/z, [x10]   // z0 = a[r_chunk, col]

// Normalise: (x - mean) * inv_std
    fsub z0.s, p1/m, z0.s, z8.s // z0 = x - mean
    fmul z0.s, p1/m, z0.s, z5.s // z0 = (x - mean) * inv_std = x_hat

// Scale by gamma[col].
// LD1RW {Zt.S}, Pg/Z, [Xn] loads one float from [Xn] and copies it to
// all active lanes — a single-instruction broadcast from memory.
    ld1rw {z1.s}, p1/z, [x13]  // z1 = [gamma[col], ..., gamma[col]]
    fmul z0.s, p1/m, z0.s, z1.s // z0 = x_hat * gamma

// Shift by beta[col].
    ld1rw {z2.s}, p1/z, [x17]  // z2 = [beta[col], ..., beta[col]]
    fadd z0.s, p1/m, z0.s, z2.s // z0 = x_hat * gamma + beta

// Store to b[r_chunk, col] — contiguous, predicated so only active rows are written.
    st1w {z0.s}, p1, [x12]

// Advance all walking pointers to the next column.
    add  x10, x10, x14          // a: += stride_col_a
    add  x12, x12, x15          // b: += stride_col_b
    add  x13, x13, #4           // gamma: += 4 bytes (one float)
    add  x17, x17, #4           // beta:  += 4 bytes
    add  x11, x11, #1
    b    Lp3col

Lp3col_done:

// ── Advance outer loop ────────────────────────────────────────────────────────
// Move both chunk-base pointers forward by SVL rows = SVL*4 bytes.
    add  x19, x19, x16          // a_chunk_base += SVL*4
    add  x20, x20, x16          // b_chunk_base += SVL*4
    add  x9,  x9,  x8           // r_base += SVL
    b    Louter

Louter_done:

// ── Epilogue ──────────────────────────────────────────────────────────────────
    smstop sm

    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #48
    ret
