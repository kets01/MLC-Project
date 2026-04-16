
#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(inner_product_asm)
.global SYMBOL(outer_product_asm)

.text
.align 2

SYMBOL(inner_product_asm):
    mov x0, #0
    ret

SYMBOL(outer_product_asm):
    // x0 = pointer to a
    // x1 = pointer to b
    // w2 = size (uint32_t const i_size)
    // x3 = pointer to c
    cbz w2, .done               // If size is 0, exit
    mov x10, x1                 // Save the base address of i_b in x10
    mov w4, w2                  // w4 = outer loop counter (starts at i_size)

.loop_i:
    ldr w5, [x0], #4            // load uint32_t a[i], post-increment
    
    // Prepare for the inner loop 
    mov x11, x10                // x11 = current pointer for i_b (reset to start of b)
    mov w6, w2                  // w6 = inner loop counter (starts at i_size)

.loop_j:
    ldr w7, [x11], #4           // w7 = b[j], x11 += 4
    
    umull x8, w5, w7            // x8 = (uint64_t)w5 * w7
    
    // Because we write row by row, we can just move x3 forward by 8 bytes
    str x8, [x3], #8            // *x3 = x8, x3 += 8 (post-increment)
    
    subs w6, w6, #1             // w6 = w6 - 1
    cbnz w6, .loop_j            // If j counter != 0, repeat inner
    
    subs w4, w4, #1             // w4 = w4 - 1
    cbnz w4, .loop_i            // If i counter != 0, repeat outer

.done:
    ret
