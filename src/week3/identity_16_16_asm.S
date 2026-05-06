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
identity_16_16:
    smstart  sm            // Enter SSVE Mode
    ptrue   p0.s            // Governing Predicate 
    
    cbz     w4, .col_major  // If trans_b == 0

    // CASE: trans_b == 1 (Row-Major store)
    index   z31.s, #0 , #1    // Create offsets:  // z1 = [0, 1, 2, 3, ... 15] 
    mov     z30.s, w3                // Broadcast ld_b (w3) into every lane of z2
    mul     z31.s, p0/m, z31.s, z30.s  // z1 = z1 * z2 = [0, ld_b, 2*ld_b, ...]
    mov     x5, #16         // 16 columns
.row_loop:
    ld1w    {z0.s}, p0/z, [x0]               // Load contiguous col from A 
    st1w    {z0.s}, p0, [x1, z31.s, uxtw #2] // Scatter Store into rows of B 
    add     x0, x0, x2, lsl #2             // A += ld_a * 4
    add     x1, x1, #4                     // B += 1 float (move to next col in row)
    subs    x5, x5, #1
    b.ne    .row_loop
    b       .exit

    // CASE: trans_b == 0 (Col-Major store)
.col_major:
    mov     x5, #4          // Process 4 columns per loop (4 iterations * 4 = 16)
.col_loop:
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
    b.ne    .col_loop

.exit:
    smstop  sm           // Exit SSVE
    ret
