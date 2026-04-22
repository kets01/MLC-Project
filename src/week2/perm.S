#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(perm_neon_abc_cba)
.text
.align 2

// x0: size_c
//x1: *abc (Input)
//x2: *cba (Output)
SYMBOL(perm_neon_abc_cba):
    cbz x0, .L_exit             // Exit if size_c == 0
    
    mov x3, #0                  // k = 0
    sub x4, x0, #4              // x4 = size_c - 4

    // Precompute the stride to go from i to i+1
    // distance = 4 (size of b) * size_c * 4 (bytes)
    lsl x8, x0, #4              // x8 = size_c * 16

.L_k_loop:
    cmp x3, x4
    b.gt .L_scalar_start

    mov x5, #0                  // j = 0
.L_j_loop:
    mov x6, #0                  // i_low = 0 (we do i=0..3 then i=4..7)
.L_i_loop:
    // Calculate input address: ((i_low * 4 + j) * size_c + k) * 4
    lsl x7, x6, #2              // i_low * 4
    add x7, x7, x5              // i_low * 4 + j
    mul x7, x7, x0              // (...) * size_c
    add x7, x7, x3              // (...) + k
    lsl x7, x7, #2              // Convert to bytes
    add x7, x7, x1              // Base + offset

    // LOAD 4x4 block across the 'i' dimension
    ldr q0, [x7]                // abc[i+0][j][k...k+3]
    add x7, x7, x8              // Stride to next 'i'
    ldr q1, [x7]                // abc[i+1][j][k...k+3]
    add x7, x7, x8
    ldr q2, [x7]                // abc[i+2][j][k...k+3]
    add x7, x7, x8
    ldr q3, [x7]                // abc[i+3][j][k...k+3]

    // TRANSPOSE: This turns 4 values of 'k' into contiguous 'i' values
    trn1 v4.4s, v0.4s, v1.4s    // [i0k0, i1k0, i0k2, i1k2]
    trn2 v5.4s, v0.4s, v1.4s    // [i0k1, i1k1, i0k3, i1k3]
    trn1 v6.4s, v2.4s, v3.4s    // [i2k0, i3k0, i2k2, i3k2]
    trn2 v7.4s, v2.4s, v3.4s    // [i2k1, i3k1, i2k3, i3k3]

    zip1 v0.2d, v4.2d, v6.2d    // v0 = [i0, i1, i2, i3] for k+0
    zip1 v1.2d, v5.2d, v7.2d    // v1 = [i0, i1, i2, i3] for k+1
    zip2 v2.2d, v4.2d, v6.2d    // v2 = [i0, i1, i2, i3] for k+2
    zip2 v3.2d, v5.2d, v7.2d    // v3 = [i0, i1, i2, i3] for k+3

    // STORE: output cba is k*32 + j*8 + i
    mov x9, #32
    mul x9, x9, x3              // k * 32
    lsl x10, x5, #3             // j * 8
    add x9, x9, x10             // k*32 + j*8
    add x9, x9, x6              // k*32 + j*8 + i_low
    lsl x9, x9, #2              // bytes
    add x9, x9, x2              // cba + offset

    str q0, [x9]                // Store k+0
    str q1, [x9, #128]          // Store k+1 (offset 32 floats = 128 bytes)
    str q2, [x9, #256]          // Store k+2
    str q3, [x9, #384]          // Store k+3

    add x6, x6, #4              // i_low += 4
    cmp x6, #8
    b.lt .L_i_loop

    add x5, x5, #1              // j++
    cmp x5, #4
    b.lt .L_j_loop

    add x3, x3, #4              // k += 4
    b .L_k_loop

.L_scalar_start:
    cmp x3, x0
    b.ge .L_exit

.L_scalar_k:
    mov x5, #0                  // j = 0
.L_scalar_j:
    mov x6, #0                  // i = 0
.L_scalar_i:
    // abc_idx = (i*4 + j)*size_c + k
    mov x7, x6
    lsl x7, x7, #2              // i*4
    add x7, x7, x5              // i*4 + j
    mul x7, x7, x0              // (i*4 + j) * size_c
    add x7, x7, x3              // + k
    ldr w8, [x1, x7, lsl #2]    // Load float

    // cba_idx = k*32 + j*8 + i
    mov x9, x3
    lsl x9, x9, #5              // k * 32
    lsl x10, x5, #3             // j * 8
    add x9, x9, x10
    add x9, x9, x6
    str w8, [x2, x9, lsl #2]    // Store float

    add x6, x6, #1
    cmp x6, #8
    b.lt .L_scalar_i
    add x5, x5, #1
    cmp x5, #4
    b.lt .L_scalar_j
    add x3, x3, #1
    cmp x3, x0
    b.lt .L_scalar_k

.L_exit:
    ret
