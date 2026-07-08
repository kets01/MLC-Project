// bw_probe_ssve.S — Sprint 2a roofline probe (NOT a norm kernel)
//
// STREAM-style scale-add executed the way the norm kernels actually access
// memory: inside ONE streaming region, with contiguous LD1W/ST1W and
// predicated tails.  This measures the single-core ceiling the SSVE kernels
// can physically reach — the NEON figure from the C++ probe is a different
// execution mode and therefore a different (possibly higher) ceiling.
//
//   d[i] = s[i] + 1.0f          (counts 1 read + 1 write per element)
//
// Same rationale as the C++ probe for scale-add rather than plain copy:
// d != s after each pass, so macOS copy-on-write cannot short-circuit.
//
// Structure: main loop processes 4 vectors per iteration (4 independent
// LD1W/FADD/ST1W chains — enough outstanding loads to saturate one core),
// then a predicated per-VL tail via WHILELO handles any remainder (VLA,
// decision D: no hard-coded SVL anywhere).
//
// C signature:
//   void bw_probe_ssve(float       *d,   // x0
//                      const float *s,   // x1
//                      int64_t      n);  // x2  (element count)

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

    // No callee-saved registers used (x0–x6, z0–z4, p0–p1 are all volatile),
    // so no prologue is needed.  Skip streaming entry entirely for n == 0.
    cbz   x2, .Lret

    smstart sm

    ptrue p0.s, all
    fmov  z4.s, #1.0           // the scale-add constant (set AFTER smstart —
                               // smstart zeroes every Z register)

    mov   x3, #0               // x3 = elements processed
    cntw  x4                   // x4 = VL in 32-bit words (queried, decision D)
    lsl   x5, x4, #2           // x5 = 4*VL words per main-loop iteration

    // -----------------------------------------------------------------------
    // Main loop: 4 full vectors per iteration, unpredicated.
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Tail: one predicated vector at a time (handles any n, VLA).
    // -----------------------------------------------------------------------
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
    ret
