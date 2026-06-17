#include "norm/norm.hpp"

namespace mini_jit::norm {

// Identity copy: confirms build wiring and the GiB/s path.
// Replaced by real LayerNorm / RMSNorm reference in Sprint 1.
void norm_placeholder(const float* a, float* b, int64_t m, int64_t n, int64_t ld) {
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            b[i + j * ld] = a[i + j * ld];
        }
    }
}

} // namespace mini_jit::norm
