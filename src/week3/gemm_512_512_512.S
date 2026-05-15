#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif
 
.global SYMBOL(gemm_512_512_512)
.p2align 2
SYMBOL(gemm_512_512_512):
    smstart
    ptrue p0.s
    mov x17, #0                         // n_offset = 0
 
n_loop:
    mov x15, #0                         // m_offset = 0
 
m_loop:
    // x12 = &C[m_offset, n_offset] = c + n_offset*ldc + m_offset
    mul x9, x17, x5
    add x12, x2, x9, lsl #2
    add x12, x12, x15, lsl #2
 
    // --- LOAD C (32x32 tile) ---
    // Loop 1: columns n_offset+0 .. n_offset+15 -> ZA0 (rows 0..15) / ZA2 (rows 16..31)
    mov x13, x12
    mov w14, #0
1:  ld1w {z0.s}, p0/z, [x13]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    mova za0v.s[w14, 0], p0/m, z0.s
    mova za2v.s[w14, 0], p0/m, z1.s
    add x13, x13, x5, lsl #2               // next column
    add w14, w14, #1
    cmp w14, #16
    b.ne 1b
    // Loop 2: columns n_offset+16 .. n_offset+31 -> ZA1/ZA3
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
    // x10 = &A[m_offset, k=0]  (col-major: a + m_offset)
    // x11 = &B[k=0, n_offset]  (row-major: b + n_offset)
    add x10, x0, x15, lsl #2
    add x11, x1, x17, lsl #2
    mov x9, #512
k_loop:
    ld1w {z0.s}, p0/z, [x10]               // A[m..m+15,  k]
    ld1w {z1.s}, p0/z, [x10, #1, mul vl]   // A[m+16..m+31, k]
    ld1w {z2.s}, p0/z, [x11]               // B[k, n..n+15]
    ld1w {z3.s}, p0/z, [x11, #1, mul vl]   // B[k, n+16..n+31]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x3, lsl #2               // next column of A
    add x11, x11, x4, lsl #2               // next row of B
    subs x9, x9, #1
    b.ne k_loop
 
    // --- STORE C (32x32 tile) ---
    // Loop 1: ZA0/ZA2 -> columns n_offset+0 .. n_offset+15
    mov x13, x12
    mov w14, #0
3:  mova z0.s, p0/m, za0v.s[w14, 0]
    mova z1.s, p0/m, za2v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 3b
    // Loop 2: ZA1/ZA3 -> columns n_offset+16 .. n_offset+31
    mov w14, #0
4:  mova z0.s, p0/m, za1v.s[w14, 0]
    mova z1.s, p0/m, za3v.s[w14, 0]
    st1w {z0.s}, p0, [x13]
    st1w {z1.s}, p0, [x13, #1, mul vl]
    add x13, x13, x5, lsl #2
    add w14, w14, #1
    cmp w14, #16
    b.ne 4b
 
    add x15, x15, #32
    cmp x15, #512
    b.ne m_loop
    add x17, x17, #32
    cmp x17, #512
    b.ne n_loop
 
    smstop
    ret
