#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

// The assembly kernels are compiled as plain C functions (no C++ name mangling).
extern "C" void rms_norm_ssve(const float*, float*, const float*,
                               int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v1(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v2(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v3(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);

namespace mini_jit::norm {

// Guard: each wrapper returns silently when SME is absent (SMSTART traps on
// hardware without SME), so callers can SKIP the test rather than get SIGILL.

void rms_norm_ssve(const float* a, float* b, const float* gamma,
                   int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v1(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v1(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v2(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v3(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v3(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

} // namespace mini_jit::norm
