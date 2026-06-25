// layer_norm_ssve.S — Hand-written SSVE LayerNorm kernel (Sprint 2)
//
// Formula: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
// Layout:  column-major — element [row, col] lives at  base + row + col*ld
//
// Three passes over each row (all using gather-load because the row is strided):
//   Pass 1 : sum every element → divide by N → mean
//   Pass 2 : sum (x - mean)^2  → divide by N → variance
//   Pass 3 : (x - mean) * inv_std * gamma + beta → scatter-store to output
//
// VLA: SVL is queried at runtime with CNTW; WHILELO builds the tail predicate
// so the code runs correctly on any SVL (not just 512-bit / 16 floats on M4).

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
//  x8        SVL in FP32 elements (constant, from CNTW)
//  x9        walking base pointer through a for the current pass
//  x10       walking base pointer through b (pass 3)
//  x11       column index i (steps by SVL each iteration)
//  x12       walking pointer into gamma (pass 3)
//  x13       walking pointer into beta  (pass 3)
//  x14       stride_a = SVL * ld_a * 4  (bytes per vector chunk in a)
//  x15       stride_b = SVL * ld_b * 4
//  x16       SVL * 4  (bytes per contiguous chunk in gamma / beta)
//  x17       loop temp  (i + SVL, reused as stride temp in pass 3)
//  x19       b_row_base (callee-saved; row r in b starts at b_ptr + r*4)
//  x20       a_row_base (callee-saved; row r in a starts at a_ptr + r*4)
//
//  z4        gather offsets for a: [0, ld_a, 2*ld_a, ..., (SVL-1)*ld_a]
//  z5        scatter offsets for b: same pattern with ld_b
//  z6        [mean, mean, ..., mean]  — broadcast, built in pass 1
//  z7        [inv_std, inv_std, ...]  — broadcast, built in pass 2
//  z0-z3     data registers (reused across passes)
//  p0        all-true predicate  (full vectors)
//  p1        tail predicate      (last partial vector)
//
//  s16       running sum    (pass 1)
//  s17       mean           (pass 1 result)
//  s18       running var    (pass 2)
//  s20       inv_std        (pass 2 result), also horizontal-add scratch
//  s21       (float)n       — divisor for mean and variance
//  s22       epsilon        — loaded from stack after smstart

SYM(layer_norm_ssve):

// ── Prologue ──────────────────────────────────────────────────────────────────
// Stack frame (48 bytes, 16-byte aligned):
//   [sp+ 0] x29 (frame pointer)
//   [sp+ 8] x30 (link register)
//   [sp+16] x19 (callee-saved: b_row_base)
//   [sp+24] x20 (callee-saved: a_row_base)
//   [sp+32] epsilon (float, saved here before smstart)
    stp  x29, x30, [sp, #-48]!  // push frame + return address; stack grows down
    mov  x29, sp                 // frame pointer = current stack top
    stp  x19, x20, [sp, #16]    // save the two callee-saved regs we will use

// Save epsilon BEFORE smstart.
// s0 is the first FP argument register; entering streaming mode can alter
// the FP/SIMD state, so we spill to memory now and reload after smstart.
    str  s0, [sp, #32]

// ── Enter Streaming SVE mode ──────────────────────────────────────────────────
// SMSTART SM turns on PSTATE.SM.  From this point we can use SVE instructions
// with the streaming vector length (SVL).  SMSTOP SM will undo this at the end.
    smstart sm

// ── One-time setup (computed once, valid for all rows) ────────────────────────

// CNTW: "count words" — returns SVL / 4 = number of 32-bit lanes per vector.
// On M4: SVL = 512 bits → CNTW = 16.  On a future chip with SVL=1024: 32.
    cntw x8                      // x8 = SVL in FP32 elements

// PTRUE P0.S: set ALL lanes of p0 to TRUE.
// We use p0 whenever we are processing a full vector (no tail).
    ptrue p0.s                   // p0 = 111...1 (SVL bits)

// Build z4 = [0, ld_a, 2*ld_a, ..., (SVL-1)*ld_a] — gather offsets for a.
// ld1w with 'uxtw #2' multiplies these 32-bit offsets by 4 (sizeof float),
// turning them into byte offsets.  So z4[i] * 4 = byte distance from column 0
// to column i along row r in a column-major matrix with leading dimension ld_a.
    index z4.s, #0, #1           // z4 = [0, 1, 2, ..., SVL-1]
    dup   z16.s, w6              // z16 = [ld_a, ld_a, ..., ld_a]  (32-bit broadcast)
    mul   z4.s, p0/m, z4.s, z16.s  // z4 = [0, ld_a, 2*ld_a, ...]

// Build z5 = [0, ld_b, 2*ld_b, ...] — same idea for the output matrix b.
    index z5.s, #0, #1
    dup   z17.s, w7
    mul   z5.s, p0/m, z5.s, z17.s  // z5 = [0, ld_b, 2*ld_b, ...]

// stride_a: how many bytes to add to the base pointer to reach the NEXT
// vector chunk of the same row.  Moving SVL columns to the right = SVL*ld_a
// floats = SVL*ld_a*4 bytes.
    lsl  x14, x6, #2             // x14 = ld_a * 4
    mul  x14, x8, x14            // x14 = SVL * ld_a * 4  (stride_a)

// stride_b: same for the output.
    lsl  x15, x7, #2
    mul  x15, x8, x15            // x15 = SVL * ld_b * 4  (stride_b)

// stride for gamma / beta: these are plain 1-D contiguous arrays.
// Moving SVL elements right = SVL * 4 bytes.
    lsl  x16, x8, #2             // x16 = SVL * 4  (gamma/beta stride)

// Convert n to FP32 so we can divide by it (FDIV on scalar registers).
    scvtf s21, x5                // s21 = (float)n

// Reload epsilon now that we are safely in streaming mode.
    ldr  s22, [x29, #32]        // s22 = epsilon

// Set up row-base pointers.  In column-major layout, row r starts at:
//   matrix_base + r * sizeof(float)   (i.e. offset r*4 bytes from the base).
    mov  x20, x0                 // x20 = a_row_base  (row 0 → a_ptr + 0)
    mov  x19, x1                 // x19 = b_row_base  (row 0 → b_ptr + 0)

// ── Outer row loop ────────────────────────────────────────────────────────────
// x4 starts at m and counts down to 0.  Each iteration handles one full row.
Lrow_loop:
    cbz  x4, Lrow_done           // if m == 0, nothing to do → exit

// ════════════════════════════════════════════════════════════════════════════
// PASS 1 — compute mean(x) for this row
// Strategy: gather-load SVL floats at a time, reduce with FADDV (horizontal
// add), accumulate partial sums in s16.  Then divide by n.
// ════════════════════════════════════════════════════════════════════════════

    fmov  s16, wzr               // s16 = 0.0  (sum accumulator; wzr = integer 0)
    mov   x9,  x20              // x9  = walking ptr into a, starts at row base
    mov   x11, xzr              // x11 = column index i = 0

Lp1_loop:
// Check: are there still SVL or more columns left?  If i + SVL > n, go to tail.
    add   x10, x11, x8          // x10 = i + SVL
    cmp   x10, x5               // compare with n
    b.hi  Lp1_tail              // if i+SVL > n: fewer than SVL elements remain

// Gather-load: read a[row, i], a[row, i+1], ..., a[row, i+SVL-1]
// x9 is already pointing at a_ptr + row*4 + i*ld_a*4  (the first element).
// z4 holds [0, ld_a, 2*ld_a, ...]; 'uxtw #2' multiplies them by 4 → byte offsets.
// So element j loads from: x9 + z4[j]*4 = a_ptr + row*4 + (i+j)*ld_a*4  ✓
    ld1w  {z0.s}, p0/z, [x9, z4.s, uxtw #2]

// FADDV: reduce all SVL active lanes of z0 to a single scalar in s20.
    faddv s20, p0, z0.s         // s20 = z0[0] + z0[1] + ... + z0[SVL-1]
    fadd  s16, s16, s20         // sum += s20

// Advance: move base pointer right by one vector chunk (SVL columns).
    add   x9,  x9,  x14        // x9 += stride_a
    add   x11, x11, x8         // i  += SVL
    b     Lp1_loop

Lp1_tail:
// Fewer than SVL elements remain (columns x11..n-1).
// WHILELO creates a predicate where lane j is TRUE if (i + j) < n.
// That's exactly the lanes we still need to process.
    cmp   x11, x5               // are there any tail elements at all?
    b.ge  Lp1_done              // if i >= n: no tail → skip

    whilelo p1.s, x11, x5      // p1[j] = 1  iff  (i + j) < n
    ld1w  {z0.s}, p1/z, [x9, z4.s, uxtw #2]  // /z: inactive lanes = 0
    faddv s20, p1, z0.s        // sum only the active lanes
    fadd  s16, s16, s20

Lp1_done:
// mean = sum / n
    fdiv  s17, s16, s21         // s17 = mean

// Broadcast mean to all vector lanes.
// s17 is the low 32 bits of z17 (lane 0).  DUP copies lane 0 to every lane.
    dup   z6.s, z17.s[0]       // z6 = [mean, mean, ..., mean]


// ════════════════════════════════════════════════════════════════════════════
// PASS 2 — compute variance: mean((x - mean)^2)
// Same loop structure as pass 1, but we subtract mean first, then square.
// ════════════════════════════════════════════════════════════════════════════

    fmov  s18, wzr               // s18 = 0.0  (var_sum accumulator)
    mov   x9,  x20              // reset walking ptr to row base
    mov   x11, xzr              // i = 0

Lp2_loop:
    add   x10, x11, x8
    cmp   x10, x5
    b.hi  Lp2_tail

    ld1w  {z0.s}, p0/z, [x9, z4.s, uxtw #2]

// z0 = x - mean   (/m = merging; all lanes active with p0 so same as /z here)
    fsub  z0.s, p0/m, z0.s, z6.s

// z0 = (x - mean)^2   (in-place square)
    fmul  z0.s, p0/m, z0.s, z0.s

    faddv s20, p0, z0.s
    fadd  s18, s18, s20

    add   x9,  x9,  x14
    add   x11, x11, x8
    b     Lp2_loop

Lp2_tail:
    cmp   x11, x5
    b.ge  Lp2_done

    whilelo p1.s, x11, x5
    ld1w  {z0.s}, p1/z, [x9, z4.s, uxtw #2]
// /m keeps inactive lanes at their current value (0 from the /z load above).
// That's fine: inactive lanes do not contribute to faddv.
    fsub  z0.s, p1/m, z0.s, z6.s
    fmul  z0.s, p1/m, z0.s, z0.s
    faddv s20, p1, z0.s
    fadd  s18, s18, s20

Lp2_done:
// var = var_sum / n
    fdiv  s18, s18, s21         // s18 = variance

// var + eps   (eps prevents division by zero when all inputs are identical)
    fadd  s18, s18, s22         // s18 = var + epsilon

// inv_std = 1 / sqrt(var + eps)
    fsqrt s18, s18              // s18 = sqrt(var + epsilon)
    fmov  s20, #1.0             // s20 = 1.0  (FP immediate)
    fdiv  s20, s20, s18         // s20 = inv_std

// Broadcast inv_std to all lanes.
    dup   z7.s, z20.s[0]       // z7 = [inv_std, inv_std, ..., inv_std]


// ════════════════════════════════════════════════════════════════════════════
// PASS 3 — normalize, scale, shift, and write output
//
// y[row, col] = (x[row, col] - mean) * inv_std * gamma[col] + beta[col]
//
// gamma and beta are contiguous 1-D arrays → plain LD1W (no gather).
// Output b is column-major → scatter-store ST1W with z5 offsets.
// ════════════════════════════════════════════════════════════════════════════

    mov   x9,  x20              // x9  = a walking ptr  (starts at a row base)
    mov   x10, x19              // x10 = b walking ptr  (starts at b row base)
    mov   x12, x2               // x12 = gamma walking ptr  (gamma[0])
    mov   x13, x3               // x13 = beta  walking ptr  (beta[0])
    mov   x11, xzr              // i = 0
// x16 still holds SVL*4 (gamma/beta advance per chunk) from the one-time setup.

Lp3_loop:
    add   x17, x11, x8          // x17 = i + SVL  (temp)
    cmp   x17, x5
    b.hi  Lp3_tail

// Load input chunk from a (column-major gather)
    ld1w  {z0.s}, p0/z, [x9, z4.s, uxtw #2]

// Normalize: (x - mean) * inv_std
    fsub  z0.s, p0/m, z0.s, z6.s    // z0 = x - mean
    fmul  z0.s, p0/m, z0.s, z7.s    // z0 = (x - mean) * inv_std  =  x_hat

// Scale: x_hat * gamma  (gamma is contiguous → simple sequential LD1W)
    ld1w  {z1.s}, p0/z, [x12]       // z1 = gamma[i..i+SVL-1]
    fmul  z0.s, p0/m, z0.s, z1.s    // z0 = x_hat * gamma

// Shift: + beta
    ld1w  {z2.s}, p0/z, [x13]       // z2 = beta[i..i+SVL-1]
    fadd  z0.s, p0/m, z0.s, z2.s    // z0 = x_hat * gamma + beta

// Scatter-store result to b (column-major)
    st1w  {z0.s}, p0, [x10, z5.s, uxtw #2]

// Advance all walking pointers to the next chunk.
    add   x9,  x9,  x14        // a:     += stride_a      (SVL*ld_a*4 bytes)
    add   x10, x10, x15        // b:     += stride_b
    add   x12, x12, x16        // gamma: += SVL*4 bytes   (contiguous)
    add   x13, x13, x16        // beta:  += SVL*4 bytes
    add   x11, x11, x8         // i     += SVL
    b     Lp3_loop

Lp3_tail:
    cmp   x11, x5
    b.ge  Lp3_done

    whilelo p1.s, x11, x5

    ld1w  {z0.s}, p1/z, [x9, z4.s, uxtw #2]
    fsub  z0.s, p1/m, z0.s, z6.s
    fmul  z0.s, p1/m, z0.s, z7.s
    ld1w  {z1.s}, p1/z, [x12]
    fmul  z0.s, p1/m, z0.s, z1.s
    ld1w  {z2.s}, p1/z, [x13]
    fadd  z0.s, p1/m, z0.s, z2.s
// ST1W with p1: only active lanes are stored → writes exactly n-i elements.
    st1w  {z0.s}, p1, [x10, z5.s, uxtw #2]

Lp3_done:

// ── Advance row pointers for next iteration ───────────────────────────────────
// Column-major: row r+1 is adjacent to row r in memory (offset +4 bytes).
    add  x20, x20, #4           // a_row_base moves to the next row
    add  x19, x19, #4           // b_row_base moves to the next row
    sub  x4, x4, #1             // row counter--
    b    Lrow_loop

Lrow_done:

// ── Epilogue ──────────────────────────────────────────────────────────────────
    smstop sm                    // exit streaming SVE mode

    ldp  x19, x20, [sp, #16]   // restore callee-saved registers
    ldp  x29, x30, [sp], #48   // restore frame pointer + link register; unwind stack
    ret                          // return to caller
