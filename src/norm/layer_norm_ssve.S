// layer_norm_ssve.S — LayerNorm V0: hand-written Streaming SVE baseline.
//
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Layout: column-major, base + row + col*ld — a column is contiguous.
// Scatter-gather is not in the SME1 mandatory streaming subset (it needs
// FEAT_SME_FA64, which Apple does not implement) and SIGILLs on the M4, so
// the kernel vectorises ACROSS ROWS and loops OVER COLUMNS: each load is one
// contiguous LD1W covering SVL consecutive rows of a single column.
//
// Three passes per row chunk, which is three READS of x — the literature's
// "two-pass" counts reduction stages, not traversals:
//   1. per-row sum -> mean
//   2. per-row (x-mean)^2 -> variance -> inv_std
//   3. normalise, scale by gamma, shift by beta, store
// gamma/beta are N-element vectors and stay L1-resident.
//
// VLA (decision D): CNTW queries SVL, WHILELO builds the tail predicate, so
// the kernel is correct at any SVL rather than the M4's 16 lanes.
//
// eps arrives in s0 and is spilled before SMSTART, which alters FP state.
//
// C signature:
//   void layer_norm_ssve(const float *a,     // x0
//                        float       *b,     // x1
//                        const float *gamma, // x2
//                        const float *beta,  // x3
//                        int64_t      m,     // x4
//                        int64_t      n,     // x5
//                        int64_t      ld_a,  // x6
//                        int64_t      ld_b,  // x7
//                        float        eps);  // s0

#ifdef __APPLE__
    #define SYM(s) _##s          // macOS prepends an underscore to C symbols
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve)
.text
.align 2

// Register map:
//   x8  SVL in FP32 lanes (CNTW)      x9  r_base, first row of the chunk
//   x10 walking column ptr in a       x11 col index
//   x12 walking ptr in b              x13 walking ptr in gamma
//   x14 stride_col_a = ld_a*4         x15 stride_col_b = ld_b*4
//   x16 SVL*4, the row-chunk stride   x17 walking ptr in beta
//   x19 a_chunk_base (callee-saved)   x20 b_chunk_base (callee-saved)
//
//   z0 data          z1 gamma bcast   z2 beta bcast
//   z4 mean          z5 inv_std       z6 {n,...}        z7 {eps,...}
//   z8 sum acc -> mean               z9 var acc -> sqrt(var+eps)
//   p1 row predicate: lane j active iff (r_base + j) < m

SYM(layer_norm_ssve):

// Prologue — 112-byte frame.  AAPCS64 makes d8-d15 callee-saved and
// smstart/smstop zero all eight (they alias the low bits of z8-z15), which
// this kernel additionally uses as its mean/variance accumulators.
    stp  x29, x30, [sp, #-112]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    str  d8,  [sp, #48]
    str  d9,  [sp, #56]
    str  d10, [sp, #64]
    str  d11, [sp, #72]
    str  d12, [sp, #80]
    str  d13, [sp, #88]
    str  d14, [sp, #96]
    str  d15, [sp, #104]

// Save epsilon BEFORE smstart — entering streaming mode can alter FP state.
    str  s0, [sp, #32]

// SMSTART SM: SVE now operates at the streaming vector length.
    smstart sm

// ── One-time setup ────────────────────────────────────────────────────────────
    cntw x8                     // x8 = SVL/4 = number of 32-bit lanes per vector

    lsl  x14, x6, #2            // x14 = ld_a * 4  (stride_col_a)
    lsl  x15, x7, #2            // x15 = ld_b * 4  (stride_col_b)
    lsl  x16, x8, #2            // x16 = SVL * 4   (row-chunk stride in bytes)

    // eps reloaded from the stack: no register survives smstart.
    ldr  s22, [x29, #32]
    dup  z7.s, z22.s[0]         // z7 = [eps, ..., eps]

    scvtf s21, x5               // s21 = (float)n
    dup  z6.s, z21.s[0]         // z6 = [n, ..., n]

    mov  x9,  xzr              // x9  = r_base = 0
    mov  x19, x0               // x19 = a_chunk_base (a + 0*4 = a initially)
    mov  x20, x1               // x20 = b_chunk_base (b + 0*4 = b initially)

// Outer loop: one SVL-row chunk per iteration.
Louter:
    cmp  x9, x4                 // exit when r_base >= m
    b.ge Louter_done

    // p1[j] = 1 iff (r_base + j) < m — all-true for a full chunk, partial
    // for the tail, so no separate branch is needed.
    whilelo p1.s, x9, x4       // p1 = row predicate for this chunk

// ════════════════════════════════════════════════════════════════════════════
// PASS 1 — per-row column sums into z8, then /N for the mean.  Each LD1W
// takes SVL consecutive rows of one column, which is contiguous in memory.
// ════════════════════════════════════════════════════════════════════════════

    fmov z8.s, #0.0             // z8 = per-row sum accumulator = 0

    mov  x10, x19               // x10 = walking col ptr = a_chunk_base (col 0)
    mov  x11, xzr               // x11 = col index = 0

Lp1col:
    cmp  x11, x5
    b.ge Lp1col_done

    // /z: inactive lanes read 0, so they cannot corrupt the sum.
    ld1w {z0.s}, p1/z, [x10]
    fadd z8.s, p1/m, z8.s, z0.s // z8[r] += a[r_base+r, col]

    add  x10, x10, x14          // advance to next column: x10 += stride_col_a
    add  x11, x11, #1
    b    Lp1col

Lp1col_done:
    fdiv z8.s, p1/m, z8.s, z6.s // z8 = sum / n = per-row mean

// ════════════════════════════════════════════════════════════════════════════
// PASS 2 — per-row variance from CENTRED values (the stable form), then
// inv_std = 1/sqrt(var + eps).
// ════════════════════════════════════════════════════════════════════════════

    fmov z9.s, #0.0             // z9 = per-row variance accumulator = 0

    mov  x10, x19               // reset walking col ptr
    mov  x11, xzr

Lp2col:
    cmp  x11, x5
    b.ge Lp2col_done

    ld1w {z0.s}, p1/z, [x10]

    fsub z0.s, p1/m, z0.s, z8.s // z0 = x - mean

    fmul z0.s, p1/m, z0.s, z0.s

    fadd z9.s, p1/m, z9.s, z0.s

    add  x10, x10, x14
    add  x11, x11, #1
    b    Lp2col

Lp2col_done:
    fdiv z9.s, p1/m, z9.s, z6.s // z9 = variance

    fadd z9.s, p1/m, z9.s, z7.s // z9 = var + eps

    fsqrt z9.s, p1/m, z9.s      // z9 = sqrt(var + eps)

    fmov z5.s, #1.0             // z5 = [1.0, ..., 1.0]  (unpredicated)
    fdiv z5.s, p1/m, z5.s, z9.s // z5 = 1 / sqrt(var + eps) = inv_std

// ════════════════════════════════════════════════════════════════════════════
// PASS 3 — y = (x - mean)*inv_std*gamma[col] + beta[col].  gamma and beta
// are one scalar per column, broadcast with LD1RW.
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

    fsub z0.s, p1/m, z0.s, z8.s // z0 = x - mean
    fmul z0.s, p1/m, z0.s, z5.s // z0 = (x - mean) * inv_std = x_hat

    ld1rw {z1.s}, p1/z, [x13]  // z1 = [gamma[col], ..., gamma[col]]
    fmul z0.s, p1/m, z0.s, z1.s // z0 = x_hat * gamma

    ld1rw {z2.s}, p1/z, [x17]  // z2 = [beta[col], ..., beta[col]]
    fadd z0.s, p1/m, z0.s, z2.s // z0 = x_hat * gamma + beta

    st1w {z0.s}, p1, [x12]

    add  x10, x10, x14          // a: += stride_col_a
    add  x12, x12, x15          // b: += stride_col_b
    add  x13, x13, #4           // gamma: += 4 bytes (one float)
    add  x17, x17, #4           // beta:  += 4 bytes
    add  x11, x11, #1
    b    Lp3col

Lp3col_done:

// Advance both chunk bases by SVL rows.
    add  x19, x19, x16          // a_chunk_base += SVL*4
    add  x20, x20, x16          // b_chunk_base += SVL*4
    add  x9,  x9,  x8           // r_base += SVL
    b    Louter

Louter_done:

    smstop sm

    ldr  d8,  [sp, #48]
    ldr  d9,  [sp, #56]
    ldr  d10, [sp, #64]
    ldr  d11, [sp, #72]
    ldr  d12, [sp, #80]
    ldr  d13, [sp, #88]
    ldr  d14, [sp, #96]
    ldr  d15, [sp, #104]
    ldp  x19, x20, [sp, #16]
    ldp  x29, x30, [sp], #112
    ret
