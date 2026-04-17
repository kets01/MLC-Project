
#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(fmadd_asm)

.text
.align 2
SYMBOL(fmadd_asm):
    // x0 the number of iteration
    fmov s0, #1.0
    fmov s1, #2.0
    fmov s2, #3.0
    fmov s3, #4.0
    fmov s4, #5.0
    fmov s5, #6.0
    fmov s6, #7.0
    fmov s7, #8.0

.loop:
    // Execute 8 independent FMADDS to saturate CPU pipelines
    // d = a * b + c
    fmadd s0, s0, s1, s2
    fmadd s1, s1, s2, s3
    fmadd s2, s2, s3, s4
    fmadd s3, s3, s4, s5
    fmadd s4, s4, s5, s6
    fmadd s5, s5, s6, s7
    fmadd s6, s6, s7, s0
    fmadd s7, s7, s0, s1

    subs x0, x0, #1
    bne .loop
    ret
