#include "week2/perm.hpp"

// Reference code to verify your Assembly
void perm_cpp_abc_cba(int64_t size_c, float const * abc, float * cba) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < size_c; ++k) {
                int64_t in_idx = i * (4 * size_c) + j * size_c + k;
                int64_t out_idx = k * 32 + j * 8 + i;
                cba[out_idx] = abc[in_idx];
            }
        }
    }
}