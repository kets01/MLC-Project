
#include "base_math.hpp"

extern "C" {

    int64_t inner_product_cpp(uint32_t const *i_a,
                              uint32_t const *i_b,
                              uint32_t const i_size) {
        int64_t o_result = 0;

        for (uint32_t l_i = 0; l_i < i_size; l_i++) {
        
            o_result += i_a[l_i] * i_b[l_i];
        }

        return o_result;
    }
}