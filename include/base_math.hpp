#ifndef BASE_MATH_H
#define BASE_MATH_H
#include <cstdint>

extern "C" {
    int64_t inner_product_asm(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size);
    void outer_product_asm(uint32_t const *i_a, uint32_t const *i_b, uint32_t i_size, uint64_t *o_c);
}
#endif
