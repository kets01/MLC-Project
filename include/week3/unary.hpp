#ifndef WEEK3_UNARY_HPP
#define WEEK3_UNARY_HPP

#include <cstdint>

extern "C" {
    void identity_16_16_asm(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b);
    void zero_16_16_asm(float * a, int64_t ld_a); // Assuming 'a' is the target to be zeroed
    void relu_16_16_asm(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b);

    void identity_16_16_cpp(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b);
    void zero_16_16_cpp(float * a, int64_t ld_a);
    void relu_16_16_cpp(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b);
}


#endif