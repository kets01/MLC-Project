// layer_norm_ssve_welford.S — LayerNorm SSVE Sprint 2c ablation: Welford
//
// Replaces the two-pass (mean then variance) reduction with Welford's online
// algorithm, computing per-lane mean and M2 in a single column sweep.  This
// eliminates one full read of the input, giving 2R+1W traffic (same as
// RMSNorm) instead of LayerNorm's usual 3R+1W.
//
// The bottleneck intentionally exposed: Welford's update rule requires
//   mean_new = mean + delta / count    (count = j+1, changes every column)
// so a scalar reciprocal 1/(j+1) must be computed and broadcast before each
// FMLA.  Here we use a scalar FDIV per column iteration (~12-cycle latency),
// which serialises the column loop and is expected to make this variant
// SLOWER than the two-pass V2 for all N > ~64.
//
// Two passes overall:
//   Pass 1 (Welford): per-lane (mean, M2) updated across all N columns.
//            After the loop: variance = M2 * 1/N; inv_std via FRSQRTE+NR.
//   Pass 2 (normalize): y = (x - mean) * inv_std * gamma + beta.
//
// C signature: same as layer_norm_ssve_v2.

#ifdef __APPLE__
    #define SYM(s) _##s
#else
    #define SYM(s) s
#endif

.global SYM(layer_norm_ssve_welford)
.text
.align 4

// Register map (constant):
//   x19 = a chunk base   x20 = b chunk base
//   x21 = gamma          x22 = beta
//   x23 = m              x24 = n
//   x25 = stride_a(B)    x26 = stride_b(B)
//   d8  = eps            x0  = row counter
//   s20 = 1.0 (constant for scalar FDIV reciprocal)
//   z4  = [1/N,...]      z7  = [eps,...]
//
// Per-block Welford pass:
//   z8 = per-lane running mean   z9  = per-lane running M2
//   z6 = x data (each column)   z0  = delta (x - old mean)
//   z1 = delta2 (x - new mean)  z2  = [1/count,...]
//   x8 = column ptr   x9 = count (1..N)   x10 = loop counter (N..1)
//
// Per-block normalize pass:
//   z5  = inv_std   z3 = NR scratch
//   z0/z1/z2 = data/gamma/beta

SYM(layer_norm_ssve_welford):

    stp  x19, x20, [sp, #-80]!
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    str  d8,       [sp, #64]

    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, x5
    lsl   x25, x6, #2
    lsl   x26, x7, #2
    fmov  s8, s0
    str   s0, [sp, #72]            // dedicated eps stash, stored BEFORE smstart
                                    // (streaming-mode entry clobbers D8 too, not
                                    // just D9-D15 — memory survives smstart, no
                                    // register does; verified empirically)

    cbz   x23, .Ldone_wf

    smstart sm
    ptrue p0.s, all

    ldr   s0, [sp, #72]            // eps reloaded from the dedicated stack slot
    dup   z7.s, z0.s[0]            // z7 = [eps,...]

    fmov  s0, #1.0
    scvtf s1, x24
    fdiv  s0, s0, s1
    dup   z4.s, z0.s[0]            // z4 = [1/N,...]

    fmov  s20, #1.0                // s20 = 1.0 (constant for per-col FDIV)
    mov   x0, #0

.Lblock_wf:
    whilelo p1.s, x0, x23
    b.none  .Lblkdone_wf

    // ── Pass 1: Welford online mean + M2 ──────────────────────────
    // Initialise per-lane state
    mov   z8.s, #0                 // mean = 0
    mov   z9.s, #0                 // M2   = 0

    mov   x8, x19                  // column pointer
    mov   x9, #1                   // count = 1 (for col 0)
    mov   x10, x24                 // loop counter

.Lwelford_col:
    // 1/(j+1) — the per-column scalar serialisation bottleneck
    scvtf s0, x9                   // float(count)
    fdiv  s0, s20, s0              // s0 = 1/count
    dup   z2.s, z0.s[0]            // z2 = [1/count,...]

    ld1w  {z6.s}, p1/z, [x8]      // z6 = x[col]

    // delta = x - mean_old  (unpredicated: inactive lanes compute 0-0=0, never stored)
    fsub  z0.s, z6.s, z8.s

    // mean_new = mean + delta / count  (predicated: only active lanes updated)
    fmla  z8.s, p1/m, z0.s, z2.s

    // delta2 = x - mean_new  (unpredicated: inactive lanes safe, never used)
    fsub  z1.s, z6.s, z8.s

    // M2 += delta * delta2
    fmla  z9.s, p1/m, z0.s, z1.s

    add   x8, x8, x25             // advance column pointer
    add   x9, x9, #1              // count++
    subs  x10, x10, #1
    b.ne  .Lwelford_col

    // variance = M2 / N; inv_std via FRSQRTE+NR
    fmul  z9.s, p1/m, z9.s, z4.s  // z9 = variance
    fadd  z9.s, p1/m, z9.s, z7.s  // var + eps

    sel     z9.s, p1, z9.s, z7.s
    frsqrte z5.s, z9.s
    fmul    z3.s, z5.s, z9.s
    frsqrts z3.s, z5.s, z3.s
    fmul    z5.s, p1/m, z5.s, z3.s // z5 = inv_std

    // ── Pass 2: normalize + scale + shift ─────────────────────────
    mov   x8,  x19
    mov   x9,  x20
    mov   x10, x21
    mov   x11, x22
    mov   x12, x24

.Lnorm_wf:
    ld1w  {z0.s}, p1/z, [x8]
    ld1rw {z1.s}, p1/z, [x10]
    ld1rw {z2.s}, p1/z, [x11]
    fsub  z0.s, p1/m, z0.s, z8.s   // x - mean
    fmul  z0.s, p1/m, z0.s, z5.s   // * inv_std
    fmul  z0.s, p1/m, z0.s, z1.s   // * gamma
    fadd  z0.s, p1/m, z0.s, z2.s   // + beta
    st1w  {z0.s}, p1, [x9]
    add   x8,  x8,  x25
    add   x9,  x9,  x26
    add   x10, x10, #4
    add   x11, x11, #4
    subs  x12, x12, #1
    b.ne  .Lnorm_wf

    addvl x19, x19, #1
    addvl x20, x20, #1
    incw  x0
    b     .Lblock_wf

.Lblkdone_wf:
    smstop sm

.Ldone_wf:
    ldr  d8,       [sp, #64]
    ldp  x25, x26, [sp, #48]
    ldp  x23, x24, [sp, #32]
    ldp  x21, x22, [sp, #16]
    ldp  x19, x20, [sp], #80
    ret
