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
    ptrue   p0.s
    dup     z31.s, #0.0      // Initialize comparison register with 0.0
    mov     x5, #4           // Unroll 4x4
.loop:
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
    b.ne    .loop
    smstop  sm
    ret
