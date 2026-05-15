#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif
 
.global SYMBOL(gemm_32_32_512)
.p2align 2
SYMBOL(gemm_32_32_512):
    smstart
    ptrue p0.s
 
    // --- LOAD C (32x32) ---
    // Loop 1: columns 0..15 -> ZA0/ZA2
    mov x13, x2
    mov w14, #0
1:  ld1w {z0.s}, p0/z, [x13]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    mova za0v.s[w14, 0], p0/m, z0.s
    mova za2v.s[w14, 0], p0/m, z1.s
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 1b
    // Loop 2: columns 16..31 -> ZA1/ZA3
    mov w14, #0
2:  ld1w {z0.s}, p0/z, [x13]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    mova za1v.s[w14, 0], p0/m, z0.s
    mova za3v.s[w14, 0], p0/m, z1.s
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 2b
 
    // --- COMPUTE: K=512 loop ---
    mov x9, #512
3:  ld1w {z0.s}, p0/z, [x0]               // A[0..15,  k]
    ld1w {z1.s}, p0/z, [x0, #1, mul vl]   // A[16..31, k]
    ld1w {z2.s}, p0/z, [x1]               // B[k, 0..15]
    ld1w {z3.s}, p0/z, [x1, #1, mul vl]   // B[k, 16..31]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x0, x0, x3, lsl #2               // next column of A (stride lda)
    add x1, x1, x4, lsl #2               // next row of B (stride ldb)
    subs x9, x9, #1
    b.ne 3b
 
    // --- STORE C (32x32) ---
    // Loop 1: ZA0/ZA2 -> columns 0..15
    mov x13, x2
    mov w14, #0
4:  mova z0.s, p0/m, za0v.s[w14, 0]
    mova z1.s, p0/m, za2v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 4b
    // Loop 2: ZA1/ZA3 -> columns 16..31
    mov w14, #0
5:  mova z0.s, p0/m, za1v.s[w14, 0]
    mova z1.s, p0/m, za3v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 5b
 
    smstop
    ret
