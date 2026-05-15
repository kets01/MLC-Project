#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(zero_16_16_asm)

.text
.align 2
SYMBOL(zero_16_16_asm):
// x0: a (target matrix), x1: ld_a
zero_16_16:
    smstart sm
    ptrue   p0.s, vl16
    dup     z0.s, #0       
    mov     x2, #4         // Unroll: 4 iterations * 4 stores = 16 columns
1:
    st1w    {z0.s}, p0, [x0]
    add     x3, x0, x1, lsl #2
    st1w    {z0.s}, p0, [x3]
    add     x4, x3, x1, lsl #2
    st1w    {z0.s}, p0, [x4]
    add     x5, x4, x1, lsl #2
    st1w    {z0.s}, p0, [x5]

    add     x0, x5, x1, lsl #2 // Jump to start of next block
    subs    x2, x2, #1
    b.ne    1b
    smstop  sm
    ret
