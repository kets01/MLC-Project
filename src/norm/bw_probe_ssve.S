// bw_probe_ssve.S — the single-core roofline probe.  NOT a norm kernel.
//
// A STREAM-style scale-add d[i] = s[i] + 1.0f, executed the way the norm
// kernels access memory: one streaming region, contiguous LD1W/ST1W,
// predicated VLA tail.  The compiler-vectorized C++ probe runs in NEON, a
// different execution mode and therefore a different ceiling — measuring the
// wrong one is the Sprint-2a "peak of what" error.
//
// Scale-add rather than a copy so d != s after each pass, which stops macOS
// copy-on-write from short-circuiting it.
//
// Four independent LD1W/FADD/ST1W chains per iteration keep enough loads in
// flight to saturate one core; WHILELO handles any n (decision D).
//
// The prologue saves d8-d15 even though the probe never names them: AAPCS64
// makes them callee-saved and smstart/smstop zero all eight.  The omission
// went unnoticed for exactly that reason, and this probe is the denominator of
// every "% of peak" in the report — measured before the fix, it destroyed 6 of
// the caller's 7 d9-d15 registers.
//
// C signature:
//   void bw_probe_ssve(float *d,          // x0
//                      const float *s,    // x1
//                      int64_t n);        // x2  (element count)

#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.arch armv9-a+sme

.global SYMBOL(bw_probe_ssve)

.text
.align 4

SYMBOL(bw_probe_ssve):

    // No callee-saved GPRs are used, but d8-d15 still must be saved: see the
    // note above.
    stp  d8,  d9,  [sp, #-64]!
    stp  d10, d11, [sp, #16]
    stp  d12, d13, [sp, #32]
    stp  d14, d15, [sp, #48]

    cbz   x2, .Lret

    smstart sm

    ptrue p0.s, all
    fmov  z4.s, #1.0           // the scale-add constant (set AFTER smstart —
                               // smstart zeroes every Z register)

    mov   x3, #0               // x3 = elements processed
    cntw  x4                   // x4 = VL in 32-bit words (queried, decision D)
    lsl   x5, x4, #2           // x5 = 4*VL words per main-loop iteration

    // Main loop: 4 full vectors per iteration, unpredicated.
.Lmain_loop:
    add   x6, x3, x5
    cmp   x6, x2
    b.hi  .Ltail               // fewer than 4*VL elements left → tail

    ld1w  {z0.s}, p0/z, [x1]
    ld1w  {z1.s}, p0/z, [x1, #1, mul vl]
    ld1w  {z2.s}, p0/z, [x1, #2, mul vl]
    ld1w  {z3.s}, p0/z, [x1, #3, mul vl]

    fadd  z0.s, z0.s, z4.s
    fadd  z1.s, z1.s, z4.s
    fadd  z2.s, z2.s, z4.s
    fadd  z3.s, z3.s, z4.s

    st1w  {z0.s}, p0, [x0]
    st1w  {z1.s}, p0, [x0, #1, mul vl]
    st1w  {z2.s}, p0, [x0, #2, mul vl]
    st1w  {z3.s}, p0, [x0, #3, mul vl]

    addvl x1, x1, #4
    addvl x0, x0, #4
    mov   x3, x6
    b     .Lmain_loop

    // Tail: one predicated vector at a time (handles any n, VLA).
.Ltail:
    whilelo p1.s, x3, x2
    b.none  .Ldone

    ld1w  {z0.s}, p1/z, [x1]
    fadd  z0.s, p1/m, z0.s, z4.s
    st1w  {z0.s}, p1, [x0]

    addvl x1, x1, #1
    addvl x0, x0, #1
    incw  x3
    b     .Ltail

.Ldone:
    smstop sm

.Lret:
    ldp  d10, d11, [sp, #16]
    ldp  d12, d13, [sp, #32]
    ldp  d14, d15, [sp, #48]
    ldp  d8,  d9,  [sp], #64
    ret
