#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif
 
.global SYMBOL(gemm_512_32_512)
.p2align 2
SYMBOL(gemm_512_32_512):
    smstart
    ptrue p0.s
    mov x15, #0                         // m_offset = 0
 
m_loop:
    // x10 = &A[m_offset, k=0] = a + m_offset  (col-major: row offset in bytes)
    add x10, x0, x15, lsl #2
    mov x11, x1                         // x11 = &B[k=0, 0]
    // x12 = &C[m_offset, j=0] = c + m_offset
    add x12, x2, x15, lsl #2
 
    // --- LOAD C (32x32 block at row m_offset) ---
    // Loop 1: columns 0..15 -> ZA0 (rows 0..15 of block) / ZA2 (rows 16..31)
    mov x13, x12
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
3:  ld1w {z0.s}, p0/z, [x10]               // A[m..m+15,  k]
    ld1w {z1.s}, p0/z, [x10, #1, mul vl]   // A[m+16..m+31, k]
    ld1w {z2.s}, p0/z, [x11]               // B[k, 0..15]
    ld1w {z3.s}, p0/z, [x11, #1, mul vl]   // B[k, 16..31]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x3, lsl #2               // next column of A
    add x11, x11, x4, lsl #2               // next row of B
    subs x9, x9, #1
    b.ne 3b
 
    // --- STORE C (32x32 block) ---
    // Loop 1: ZA0/ZA2 -> columns 0..15
    mov x13, x12
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
 
    add x15, x15, #32
    cmp x15, #512
    b.ne m_loop
 
    smstop
    ret
