#include "week3/unary.hpp"
#include <algorithm>

extern "C" {
    void identity_16_16_cpp(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b) {
        for (int j = 0; j < 16; ++j) { // Columns
            for (int i = 0; i < 16; ++i) { // Rows
                float val = a[i + j * ld_a];
                if (trans_b == 0) {
                    b[i + j * ld_b] = val; // Column-Major
                } else {
                    b[i * ld_b + j] = val; // Row-Major
                }
            }
        }
    }

    void zero_16_16_cpp(float * a, int64_t ld_a) {
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                a[i + j * ld_a] = 0.0f;
            }
        }
    }

    void relu_16_16_cpp(float const * a, float * b, int64_t ld_a, int64_t ld_b, int32_t trans_b) {
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 16; ++i) {
                float val = std::max(0.0f, a[i + j * ld_a]);
                if (trans_b == 0) {
                    b[i + j * ld_b] = val;
                } else {
                    b[i * ld_b + j] = val;
                }
            }
        }
    }
}