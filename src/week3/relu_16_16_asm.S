#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(relu_16_16_asm)

.text
.align 2
SYMBOL(relu_16_16_asm):
    smstart sm
    ptrue   p0.s, vl16
    fmov     z31.s, #0.0      // Initialize comparison register with 0.0
    cbz     w4, 1f           // If trans_b == 0, jump to col-major logic

    // --- CASE: trans_b == 1 (Transpose + ReLU) ---
    index   z29.s, #0, #1     // [0, 1, 2, ... 15]
    dup     z30.s, w3         // Broadcast ldb
    mul     z29.s, p0/m, z29.s, z30.s // [0, ldb, 2*ldb, ... 15*ldb]
    mov     x5, #16
2:
    ld1w    {z0.s}, p0/z, [x0]
    fmax    z0.s, p0/m, z0.s, z31.s          // Apply ReLU: max(x, 0)
    st1w    {z0.s}, p0, [x1, z29.s, uxtw #2] // Scatter Store (Transposed)
    
    add     x0, x0, x2, lsl #2               // a += lda
    add     x1, x1, #4                       // b += 1 float (next row)
    subs    x5, x5, #1
    b.ne    2b
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
    smstop  sm
    ret
