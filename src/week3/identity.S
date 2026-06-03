#ifdef __APPLE__
    #define SYMBOL(s) _##s
#else
    #define SYMBOL(s) s
#endif

.global SYMBOL(identity_16_16)

.text
.align 2
//

