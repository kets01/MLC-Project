#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(gemm_32_32_1)
.p2align 2

// void gemm_32_32_1(x0: a, x1: b, x2: c, x3: lda, x4: ldb, x5: ldc)

SYMBOL(gemm_32_32_1):
    smstart      // Start SME hardware
    ptrue p0.s      // Predicate for 16 floats (512-bit)

    // --- PART 1: LOAD 32x32 BLOCK OF C INTO ZA ---
    // We treat the 32x32 block as 32 vertical columns.
    // Each vertical column of 32 elements is loaded as 2 vectors.
    mov w12, #0     // Slice index (0-15)
    mov x10, x2     // Current column pointer
load_c_loop:
    // Load top half (rows 0-15) and bottom half (rows 16-31) of column j
    ld1w {z0.s}, p0/z, [x10]
    ld1w {z1.s}, p0/z, [x10, #1, mul vl]   // FIX: Use scaled immediate
    
    // Move into Vertical slices of the interleaved tiles
    mova za0v.s[w12, 0], p0/m, z0.s
    mova za2v.s[w12, 0], p0/m, z1.s
    
    // Move to next column (j+1)
    add x11, x10, x5, lsl #2              // pointer + (ldc * 4 bytes)
    ld1w {z0.s}, p0/z, [x11]
    ld1w {z1.s}, p0/z, [x11, #1, mul vl]   // FIX: Use scaled immediate
    
    mova za1v.s[w12, 0], p0/m, z0.s
    mova za3v.s[w12, 0], p0/m, z1.s

    add x10, x11, x5, lsl #2              // Move to column j+2
    add w12, w12, #1                      // Process 2 columns per loop
    cmp w12, #16
    b.ne load_c_loop

    // --- PART 2: THE MATH (K=1) ---
    // Load Column A (32 elements)
    ld1w {z0.s}, p0/z, [x0]
    ld1w {z1.s}, p0/z, [x0, #1, mul vl]    // FIX: Use scaled immediate

    // Load Row B (32 elements)
    ld1w {z2.s}, p0/z, [x1]
    ld1w {z3.s}, p0/z, [x1, #1, mul vl]    // FIX: Use scaled immediate

    // Outer Product Update
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s

    // --- PART 3: STORE 32x32 BLOCK BACK TO RAM ---
    mov w12, #0
    mov x10, x2
store_c_loop:
    mova z0.s, p0/m, za0v.s[w12, 0]
    mova z1.s, p0/m, za2v.s[w12, 0]
    st1w {z0.s}, p0, [x10]
    st1w {z1.s}, p0, [x10, #1, mul vl]     // FIX: Use scaled immediate
    
    add x11, x10, x5, lsl #2
    mova z0.s, p0/m, za1v.s[w12, 0]
    mova z1.s, p0/m, za3v.s[w12, 0]
    st1w {z0.s}, p0, [x11]
    st1w {z1.s}, p0, [x11, #1, mul vl]     // FIX: Use scaled immediate
    
    add x10, x11, x5, lsl #2
    add w12, w12, #1
    cmp w12, #16
    b.ne store_c_loop
    
    smstop 
    ret
