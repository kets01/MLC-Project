#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(relu_16_16_asm)

.text
.align 2
SYMBOL(relu_16_16_asm):
// x0: a (src), x1: b (dst), x2: ld_a, x3: ld_b, w4: trans_b
//
// Bare SMSTART/SMSTOP (not `smstart sm`): the trans_b == 1 path stages the
// transpose through a ZA tile, so PSTATE.ZA must be enabled too.
    smstart                  // Enter SSVE mode + enable ZA
    ptrue   p0.s, vl16
    fmov     z31.s, #0.0      // Initialize comparison register with 0.0
                              // (set AFTER smstart — streaming entry zeroes Z)
    cbz     w4, 1f           // If trans_b == 0, jump to col-major logic

    // -----------------------------------------------------------------
    // trans_b == 1 — ReLU with a transposing store:
    //   b[i*ld_b + j] = max(a[i + j*ld_a], 0)
    //
    // Same constraint and same fix as identity_16_16_asm.S: streaming mode has
    // no scatter without FEAT_SME_FA64, so the transpose is staged through ZA
    // (A's columns into horizontal slices, vertical slices back out as rows).
    // This path had never been exercised — the ReLU test only passed
    // trans_b = 0 — which is how it kept the broken scatter store long after
    // the identical defect was fixed next door.  ReLU is elementwise, so it
    // may be applied on either side of the staging; clamping on the way OUT
    // matches week6's Unary::emit_transposed.
    // -----------------------------------------------------------------
    mov     x6, x0          // running A pointer (walks columns)
    mov     w12, #0         // ZA slice base
    mov     x5, #4          // 4 groups x 4 slices = 16 columns
2:
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
    b.ne    2b

    mov     x7, x1          // running B pointer (walks rows)
    mov     w12, #0
    mov     x5, #4
5:
    mova    z0.s, p0/m, za0v.s[w12, 0]      // vertical slice i = row i of A
    fmax    z0.s, p0/m, z0.s, z31.s         // Apply ReLU: max(x, 0)
    st1w    {z0.s}, p0, [x7]                // -> row i of B (contiguous)
    add     x7, x7, x3, lsl #2              // B += ld_b * 4 (next row)
    mova    z0.s, p0/m, za0v.s[w12, 1]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    mova    z0.s, p0/m, za0v.s[w12, 2]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    mova    z0.s, p0/m, za0v.s[w12, 3]
    fmax    z0.s, p0/m, z0.s, z31.s
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x3, lsl #2
    add     w12, w12, #4
    subs    x5, x5, #1
    b.ne    5b
    b       3f               // Exit
1:
    mov     x5, #4           // Unroll 4x4
4:
    // Load 4 columns
    ld1w    {z0.s}, p0/z, [x0]
    add     x6, x0, x2, lsl #2
    ld1w    {z1.s}, p0/z, [x6]
    add     x7, x6, x2, lsl #2
    ld1w    {z2.s}, p0/z, [x7]
    add     x8, x7, x2, lsl #2
    ld1w    {z3.s}, p0/z, [x8]

    // Apply ReLU (Max against 0.0) in parallel
    fmax    z0.s, p0/m, z0.s, z31.s
    fmax    z1.s, p0/m, z1.s, z31.s
    fmax    z2.s, p0/m, z2.s, z31.s
    fmax    z3.s, p0/m, z3.s, z31.s           
    
    // Store 4 columns
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
