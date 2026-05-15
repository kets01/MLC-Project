#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif
 
.global SYMBOL(gemm_32_32_1)
.p2align 2
SYMBOL(gemm_32_32_1):
    smstart
    ptrue p0.s
 
    // --- LOAD C (32x32) ---
    // Loop 1: columns 0..15 -> ZA0 (rows 0..15) and ZA2 (rows 16..31)
    mov x13, x2         // x13 = &C[0, 0]
    mov w14, #0
1:  ld1w {z0.s}, p0/z, [x13]               // C[0..15,  j=w14]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]   // C[16..31, j=w14]
    mova za0v.s[w14, 0], p0/m, z0.s
    mova za2v.s[w14, 0], p0/m, z1.s
    add x13, x13, x5, lsl #2               // advance one column (ldc floats)
    add w14, w14, #1
    cmp w14, #16
    b.ne 1b
    // x13 now points to &C[0, 16]
    // Loop 2: columns 16..31 -> ZA1 (rows 0..15) and ZA3 (rows 16..31)
    mov w14, #0
2:  ld1w {z0.s}, p0/z, [x13]               // C[0..15,  j=16+w14]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]   // C[16..31, j=16+w14]
    mova za1v.s[w14, 0], p0/m, z0.s
    mova za3v.s[w14, 0], p0/m, z1.s
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 2b
 
    // --- COMPUTE: single outer product (K=1) ---
    // A column 0: x0[0..15] in z0, x0[16..31] in z1
    // B row 0:    x1[0..15] in z2, x1[16..31] in z3
    ld1w {z0.s}, p0/z, [x0]
    ld1w {z1.s}, p0/z, [x0, #1, mul vl]
    ld1w {z2.s}, p0/z, [x1]
    ld1w {z3.s}, p0/z, [x1, #1, mul vl]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s   // C[0..15,  0..15]  += A[0..15]  * B[0..15]
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s   // C[0..15,  16..31] += A[0..15]  * B[16..31]
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s   // C[16..31, 0..15]  += A[16..31] * B[0..15]
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s   // C[16..31, 16..31] += A[16..31] * B[16..31]
 
    // --- STORE C (32x32) ---
    // Loop 1: ZA0/ZA2 -> columns 0..15
    mov x13, x2
    mov w14, #0
3:  mova z0.s, p0/m, za0v.s[w14, 0]
    mova z1.s, p0/m, za2v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 3b
    // Loop 2: ZA1/ZA3 -> columns 16..31
    mov w14, #0
4:  mova z0.s, p0/m, za1v.s[w14, 0]
    mova z1.s, p0/m, za3v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 4b
 
    smstop
    ret
