#include "norm/norm.hpp"

// The assembly kernel is a plain C symbol (no name mangling).
// We declare it here with extern "C" so the linker finds it,
// then wrap it in the mini_jit::norm namespace so C++ callers
// use the canonical signature from norm.hpp.
extern "C" void layer_norm_ssve(const float* a,
                                 float*       b,
                                 const float* gamma,
                                 const float* beta,
                                 int64_t      m,
                                 int64_t      n,
                                 int64_t      ld_a,
                                 int64_t      ld_b,
                                 float        epsilon);

namespace mini_jit::norm {

void layer_norm_ssve(const float* a,
                     float*       b,
                     const float* gamma,
                     const float* beta,
                     int64_t      m,
                     int64_t      n,
                     int64_t      ld_a,
                     int64_t      ld_b,
                     float        epsilon) {
    ::layer_norm_ssve(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

} // namespace mini_jit::norm
