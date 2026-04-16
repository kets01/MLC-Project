/* Macro to handle the underscore difference between macOS and Linux */
#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(inner_product_asm)
.global SYMBOL(outer_product_asm)

SYMBOL(inner_product_asm):
    mov x0, #0
    ret

SYMBOL(outer_product_asm):
    ret
