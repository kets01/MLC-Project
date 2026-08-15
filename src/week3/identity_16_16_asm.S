#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(identity_16_16_asm)

.text
.align 2
SYMBOL(identity_16_16_asm):
// x0: a (src), x1: b (dst), x2: ld_a, x3: ld_b, w4: trans_b
//
// Bare SMSTART/SMSTOP (not `smstart sm`): the trans_b == 1 path stages the
// transpose through a ZA tile, so PSTATE.ZA must be enabled too.  SMSTOP
// disables both before returning, so no ZA state is live across the call.
    smstart                        // Enter SSVE mode + enable ZA
    ptrue   p0.s, vl16             // Governing Predicate (16 fp32 lanes)

    cbz     w4, 2f          // If trans_b == 0, jump to col_major
    // -----------------------------------------------------------------
    // trans_b == 1 — A column-major in, B row-major out:
    //   b[i*ld_b + j] = a[i + j*ld_a]
    //
    // A real memory transpose: the 16 values of one A column are contiguous
    // but belong to 16 different rows of B.  A scatter store would express
    // that directly, and this kernel used to do exactly that — but scatter
    // with vector offsets is unavailable in streaming mode without
    // FEAT_SME_FA64, which Apple has not implemented, so it never completed
    // on the M4.  (CI never caught it: no SME, so the case is skipped.)
    //
    // The fix uses ZA for what it is good at — 2-D movement.  Load A's
    // columns into HORIZONTAL slices, so za0[j][i] = a[i + j*ld_a]; vertical
    // slice i is then row i of A, exactly the 16 contiguous floats row i of B
    // needs.  Every slice is written before it is read, so no `zero {za}`.
    //
    // The fp32 MOVA slice-offset immediate is 2 bits (0-3); the full 0-15
    // index is W12 (which must be W12-W15) plus that offset, hence 4 groups
    // of 4 with W12 stepping by 4.
    // -----------------------------------------------------------------
    mov     x6, x0          // running A pointer (walks columns)
    mov     w12, #0         // ZA slice base
    mov     x5, #4          // 4 groups x 4 slices = 16 columns
1:
    ld1w    {z0.s}, p0/z, [x6]              // A column j (contiguous)
    mova    za0h.s[w12, 0], p0/m, z0.s      // -> horizontal slice j
    add     x6, x6, x2, lsl #2              // A += ld_a * 4 (next column)
    ld1w    {z0.s}, p0/z, [x6]
    mova    za0h.s[w12, 1], p0/m, z0.s
    add     x6, x6, x2, lsl #2
    ld1w    {z0.s}, p0/z, [x6]
    mova    za0h.s[w12, 2], p0/m, z0.s
    add     x6, x6, x2, lsl #2
    ld1w    {z0.s}, p0/z, [x6]
    mova    za0h.s[w12, 3], p0/m, z0.s
    add     x6, x6, x2, lsl #2
    add     w12, w12, #4
    subs    x5, x5, #1
    b.ne    1b

    mov     x7, x1          // running B pointer (walks rows)
    mov     w12, #0
    mov     x5, #4
5:
    mova    z0.s, p0/m, za0v.s[w12, 0]      // vertical slice i = row i of A
    st1w    {z0.s}, p0, [x7]                // -> row i of B (contiguous)
    add     x7, x7, x3, lsl #2              // B += ld_b * 4 (next row)
    mova    z0.s, p0/m, za0v.s[w12, 1]
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    mova    z0.s, p0/m, za0v.s[w12, 2]
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    mova    z0.s, p0/m, za0v.s[w12, 3]
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    add     w12, w12, #4
    subs    x5, x5, #1
    b.ne    5b
    b       3f

    // CASE: trans_b == 0 (Col-Major store)
2:
    mov     x5, #4          // Process 4 columns per loop (4 iterations * 4 = 16)
4:
    // Load 4 contiguous columns
    ld1w    {z0.s}, p0/z, [x0]
    add     x6, x0, x2, lsl #2
    ld1w    {z1.s}, p0/z, [x6]
    add     x7, x6, x2, lsl #2
    ld1w    {z2.s}, p0/z, [x7]
    add     x8, x7, x2, lsl #2
    ld1w    {z3.s}, p0/z, [x8]

    // Store 4 contiguous columns
    st1w    {z0.s}, p0, [x1]
    add     x9, x1, x3, lsl #2
    st1w    {z1.s}, p0, [x9]
    add     x10, x9, x3, lsl #2
    st1w    {z2.s}, p0, [x10]
    add     x11, x10, x3, lsl #2
    st1w    {z3.s}, p0, [x11]

    add     x0, x8, x2, lsl #2             
    add     x1, x11, x3, lsl #2          
    subs    x5, x5, #1
    b.ne    4b

3:
    smstop               // Exit SSVE and disable ZA
    ret

