#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

// The assembly kernels are compiled as plain C functions (no C++ name mangling).
extern "C" void layer_norm_ssve(const float*, float*, const float*, const float*,
                                 int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v1(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v2(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v4(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v5(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v6(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_welford(const float*, float*, const float*, const float*,
                                         int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v7(const float*, float*, const float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_za_sme2(const float*, float*, const float*, const float*,
                                   int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_za(const float*, float*, const float*, const float*,
                               int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve(const float*, float*, const float*,
                               int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v1(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v2(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v3(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v4(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v5(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v6(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v7(const float*, float*, const float*,
                                  int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_ssve_v7x2(const float*, float*, const float*,
                                    int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_za_sme2(const float*, float*, const float*,
                                 int64_t, int64_t, int64_t, int64_t, float);
extern "C" void rms_norm_za(const float*, float*, const float*,
                            int64_t, int64_t, int64_t, int64_t, float);
extern "C" void bw_probe_ssve(float*, const float*, int64_t);

namespace mini_jit::norm {

// Guard: each wrapper returns silently when SME is absent (SMSTART traps on
// hardware without SME), so callers can SKIP the test rather than get SIGILL.

void layer_norm_ssve(const float* a, float* b, const float* gamma, const float* beta,
                     int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v1(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_v1(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v2(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_v2(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v4(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_v4(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v5(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_v5(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v6(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_welford(const float* a, float* b, const float* gamma, const float* beta,
                              int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_ssve_welford(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

// V7 needs SME2; on an SME1-only machine fall back to V6, the variant it is
// derived from, so the SME2 path stays a pure opt-in optimisation.
void layer_norm_ssve_v7(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    if (!cpu_supports_sme2()) {
        ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::layer_norm_ssve_v7(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_za_sme2(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    if (!cpu_supports_sme2()) {
        ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::layer_norm_za_sme2(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_za(const float* a, float* b, const float* gamma, const float* beta,
                   int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::layer_norm_za(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

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

void rms_norm_ssve_v4(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v4(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v5(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v5(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v6(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

// V7 needs SME2 for its multi-vector loads/stores.  Where SME2 is absent but
// SME is present (an SME1-only machine), fall back to V6 — the variant V7 is
// derived from — so callers get a correct result everywhere and the SME2 path
// is a pure opt-in optimisation (ROADMAP Sprint 6: "guard the SME2 path
// behind a FEAT_SME2 runtime check so it degrades to the SME1 kernel").
void rms_norm_ssve_v7(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_ssve_v7(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v7x2(const float* a, float* b, const float* gamma,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_ssve_v7x2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_za_sme2(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_za_sme2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_za(const float* a, float* b, const float* gamma,
                 int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (!cpu_supports_sme()) return;
    ::rms_norm_za(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void bw_probe_ssve(float* d, const float* s, int64_t n) {
    if (!cpu_supports_sme()) return;
    ::bw_probe_ssve(d, s, n);
}

} // namespace mini_jit::norm
