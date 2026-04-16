/* Macro to handle the underscore difference between macOS and Linux */
#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(inner_product_asm)
.global SYMBOL(outer_product_asm)

SYMBOL(inner_product_asm):
    // x0 = pointer to a
    // x1 = pointer to b
    // w2 = size (uint32_t const i_size)
    
    mov x3, #0                  // initialize x3 to 0 (int64_t o_result = 0)

.loop:
    cbz w2, .done               // if size is 0, exit loop

    ldr w4, [x0], #4            // load uint32_t a[i], post-increment
    ldr w5, [x1], #4            // load uint32_t b[i], post-increment

    umaddl x3, w4, w5, x3       // x3 = x3 + (w4 *w5)  (Unsigned Multiply-Add Long)

    sub w2, w2, #1              // size--
    b .loop                     // repeat
.done:
    mov x0, x3                  // Move
    ret

SYMBOL(outer_product_asm):
    ret
