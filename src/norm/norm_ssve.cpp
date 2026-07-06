#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

// The assembly kernel is compiled as a plain C function (no C++ name mangling).
extern "C" void rms_norm_ssve(const float* a,
                               float*       b,
                               const float* gamma,
                               int64_t      m,
                               int64_t      n,
                               int64_t      ld_a,
                               int64_t      ld_b,
                               float        epsilon);

namespace mini_jit::norm {

void rms_norm_ssve(const float* a,
                   float*       b,
                   const float* gamma,
                   int64_t      m,
                   int64_t      n,
                   int64_t      ld_a,
                   int64_t      ld_b,
                   float        epsilon) {
    // Guard: the assembly uses SMSTART/SMSTOP and SVE instructions that trap
    // on hardware without SME.  Return silently so callers can check
    // cpu_supports_sme() and SKIP the test rather than getting SIGILL.
    if (!cpu_supports_sme()) return;

    ::rms_norm_ssve(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

} // namespace mini_jit::norm
