// rms_norm_ssve.S — hand-written Streaming SVE RMSNorm kernel
//
// Algorithm per row:
//   sumsq   = Σ x[j]²  (pass 1: FMLA reduction + FADDV)
//   inv_rms = 1 / √(sumsq/N + ε)  (scalar FDIV/FSQRT)
//   y[j]    = x[j] · γ[j] · inv_rms  (pass 2: FMUL)
//
// VLA: INCW advances the element index by SVL/4 (vector length in FP32
// elements) — no hard-coded tile size.  WHILELO sets predicate for the
// partial tail so the same code handles N not divisible by the vector width.
//
// One SMSTART SM wraps all M rows; SMSTOP SM at the very end.
//
// C signature:
//   void rms_norm_ssve(const float *a,     // x0
//                      float       *b,     // x1
//                      const float *gamma, // x2
//                      int64_t      m,     // x3
//                      int64_t      n,     // x4
//                      int64_t      ld_a,  // x5  (in elements)
//                      int64_t      ld_b,  // x6  (in elements)
//                      float        eps);  // s0  (first FP arg register)

// macOS prepends an underscore to C symbols; Linux does not.
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
    // Prologue — save callee-saved registers we will clobber.
    //
    // AArch64 ABI: X19-X28 and D8-D15 are callee-saved.
    // SP must remain 16-byte aligned at all times.
    // We allocate 64 bytes on the stack: 5 GPR pairs (x19-x25) + d8.
    // -----------------------------------------------------------------------
    stp  x19, x20, [sp, #-64]!   // allocate 64 bytes; save x19, x20 at [sp+0]
    stp  x21, x22, [sp, #16]     // save x21, x22 at [sp+16]
    stp  x23, x24, [sp, #32]     // save x23, x24 at [sp+32]
    str  x25,      [sp, #48]     // save x25 at [sp+48]
    str  d8,       [sp, #56]     // save d8 (= S8 is its 32-bit view) at [sp+56]

    // -----------------------------------------------------------------------
    // Move arguments into callee-saved registers so the inner loops can use
    // x0-x17 freely as scratch without losing the base values.
    //
    // Epsilon arrives in S0 (the 32-bit alias of V0).  We copy it to S8
    // (callee-saved FP register) so the reduction/normalize passes can
    // overwrite Z0/Z1/Z2/Z3 freely.
    // -----------------------------------------------------------------------
    mov   x19, x0            // a  — row pointer, advanced each row
    mov   x20, x1            // b  — row pointer, advanced each row
    mov   x21, x2            // gamma  — fixed throughout
    mov   x22, x3            // m  — rows remaining
    mov   x23, x4            // n  — feature count (fixed)
    lsl   x24, x5, #2        // ld_a_bytes = ld_a * 4 (byte stride between rows of A)
    lsl   x25, x6, #2        // ld_b_bytes = ld_b * 4
    fmov  s8,  s0            // save epsilon into callee-saved S8

    // Early exit if m == 0 (nothing to process).
    cbz   x22, .Lrow_done

    // -----------------------------------------------------------------------
    // Enter Streaming SVE mode.
    //
    // SMSTART SM switches the SVE vector length from the non-streaming VL to
    // the streaming VL (SVL).  On the M4 SVL = 512 bits = 16 FP32 lanes.
    // All Z and P registers may be in an undefined state after SMSTART;
    // we initialise every register we read before using it.
    // -----------------------------------------------------------------------
    smstart sm

    // P0 = all-true predicate (every FP32 lane active).
    // Used by FADDV to sum *all* lanes of the accumulator at the end of
    // the reduction, including lanes that were zeroed during the tail.
    ptrue  p0.s, all

    // -----------------------------------------------------------------------
    // Outer loop — one iteration per row.
    // -----------------------------------------------------------------------
.Lrow_loop:

    // === Pass 1: sum of squares ============================================
    //
    // Initialise the accumulator Z2 to 0.0 in all lanes.
    // MOV Z2.S, #0 is an alias for DUP Z2.S, #0 (integer immediate 0 =
    // bit-identical to FP32 +0.0).
    mov   z2.s, #0

    mov   x0,  #0             // element index j = 0

.Lred_loop:
    // WHILELO P1.S, Xj, Xn:
    //   Lane i is TRUE iff (j + i) < n  (unsigned).
    //   This handles both full vectors and the partial tail in one idiom.
    //   Condition flags after WHILELO: Z=1 if no lanes are active (j >= n).
    whilelo p1.s, x0, x23
    b.none  .Lred_done        // exit when j >= n (no active lanes)

    // Load a chunk of x from the current row.  [x19, x0, lsl #2] means
    // base = x19 (= a_row), offset = j * 4 bytes.  P1/Z = zero-predication:
    // inactive lanes get 0.0, so they don't pollute the accumulator.
    ld1w   {z0.s}, p1/z, [x19, x0, lsl #2]

    // FMLA Z2.S, P1/M, Z0.S, Z0.S  →  Z2[i] += Z0[i] * Z0[i]  (x² per lane)
    // P1/M = merge predication: inactive lanes keep their previous Z2 value
    // (which is 0.0 from initialisation or a correct partial sum).
    fmla   z2.s, p1/m, z0.s, z0.s

    // INCW Xj increments by SVL/4 — the number of FP32 elements per vector.
    // This is the VLA-correct idiom: no hard-coded constant.
    incw   x0
    b      .Lred_loop

.Lred_done:

    // === Scalar: compute inv_rms ===========================================
    //
    // FADDV S0, P0, Z2.S — horizontal sum of all lanes.
    // We use P0 (all-true) because the inactive lanes were initialised to 0
    // and FMLA kept them at 0, so summing them is safe.
    // After this, S0 = sumsq = Σ x[j]².
    //
    // Note: S0 is the 32-bit alias of V0, which is element [0] of Z0 in
    // streaming mode.  All scalar FP instructions remain valid in SSVE.
    faddv  s0, p0, z2.s       // s0 = Σ z2 lanes  = sumsq

    // Convert N (integer in X23) to FP32 in S1 for the division.
    scvtf  s1, x23            // s1 = float(n)
    fdiv   s0, s0, s1         // s0 = sumsq / n   = mean(x²)
    fadd   s0, s0, s8         // s0 += epsilon     = mean(x²) + ε
    fsqrt  s0, s0             // s0 = √(mean(x²) + ε) = rms
    fmov   s1, #1.0           // s1 = 1.0
    fdiv   s0, s1, s0         // s0 = 1.0 / rms   = inv_rms

    // Broadcast inv_rms from S0 (= Z0.S[0]) to all lanes of Z3.
    // The normalize pass multiplies every chunk by Z3.
    dup    z3.s, z0.s[0]      // z3 = {inv_rms, inv_rms, ...}

    // === Pass 2: normalize + scale =========================================
    mov   x0, #0              // reset element index j = 0

.Lnrm_loop:
    whilelo p1.s, x0, x23
    b.none  .Lnrm_done

    ld1w   {z0.s}, p1/z, [x19, x0, lsl #2]   // z0 = x[j:j+vl]
    ld1w   {z1.s}, p1/z, [x21, x0, lsl #2]   // z1 = gamma[j:j+vl]

    // y = x * gamma * inv_rms  (two in-place multiplies; order doesn't matter
    // for FP32 at this accuracy level, and we have no beta in RMSNorm).
    fmul   z0.s, p1/m, z0.s, z1.s            // z0 = x * gamma
    fmul   z0.s, p1/m, z0.s, z3.s            // z0 *= inv_rms

    st1w   {z0.s}, p1, [x20, x0, lsl #2]     // store y[j:j+vl] (inactive lanes not written)

    incw   x0
    b      .Lnrm_loop

.Lnrm_done:

    // === Advance row pointers and loop =====================================
    add    x19, x19, x24      // a_row += ld_a_bytes
    add    x20, x20, x25      // b_row += ld_b_bytes
    subs   x22, x22, #1       // --rows_remaining
    b.ne   .Lrow_loop

    smstop sm                  // exit streaming mode

.Lrow_done:
    // -----------------------------------------------------------------------
    // Epilogue — restore callee-saved registers in reverse order.
    // -----------------------------------------------------------------------
    ldr  d8,       [sp, #56]
    ldr  x25,      [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #64  // restore x19/x20 and deallocate stack frame
    ret
