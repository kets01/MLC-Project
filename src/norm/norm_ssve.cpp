#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme(), cpu_supports_sme2()

#include <cstdio>    // fprintf on the precondition path
#include <cstdlib>   // abort
#include <cstring>   // strstr

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
extern "C" void smstart_probe_pairs(int64_t);
extern "C" void smstart_probe_sm_only(int64_t);
extern "C" void smstart_probe_empty(int64_t);
extern "C" int64_t svl_fp32_lanes();

namespace mini_jit::norm {

namespace {

// Sprint 8.  These wrappers used to `return;` when SME was absent, which meant
// a caller on an M1/M2 got no computation, no status and no diagnostic — it
// could not distinguish success from a no-op, and would then read whatever was
// in the output buffer.  CLAUDE.md §4 requires the opposite: fail fast and
// clearly at the boundary rather than produce silent wrong numbers.
//
// So an ISA-specific entry point now aborts with a message naming itself and
// the missing feature, and callers who want a result on any CPU use the
// layer_norm()/rms_norm() dispatchers below.
[[noreturn]] void isa_precondition_failed(const char* fn, const char* feature) {
    std::fprintf(stderr,
        "\nFATAL: %s requires %s, which this CPU does not report.\n"
        "  This is an ISA-specific entry point with a hard precondition; it\n"
        "  will not silently return an uncomputed buffer.\n"
        "  Either guard the call with cpu_supports_sme(), or call\n"
        "  mini_jit::norm::%s() instead, which dispatches to the best\n"
        "  available implementation and falls back to the scalar reference.\n\n",
        fn, feature,
        std::strstr(fn, "layer") != nullptr ? "layer_norm" : "rms_norm");
    std::abort();
}

inline void require_sme(const char* fn) {
    if (!cpu_supports_sme()) isa_precondition_failed(fn, "FEAT_SME");
}

} // namespace

// ---------------------------------------------------------------------------
// Public dispatchers — always compute a correct result on any CPU.
// ---------------------------------------------------------------------------

void layer_norm(const float* a, float* b, const float* gamma, const float* beta,
                int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (cpu_supports_sme2())
        ::layer_norm_ssve_v7(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
    else if (cpu_supports_sme())
        ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
    else
        layer_norm_ref(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void rms_norm(const float* a, float* b, const float* gamma,
              int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    if (cpu_supports_sme2())
        ::rms_norm_ssve_v7(a, b, gamma, m, n, ld_a, ld_b, epsilon);
    else if (cpu_supports_sme())
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
    else
        rms_norm_ref(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

const char* norm_dispatch_target() {
    if (cpu_supports_sme2()) return "SME2 (V7 multi-vector)";
    if (cpu_supports_sme())  return "SME (V6 contiguity-grouped SSVE)";
    return "scalar reference (no SME on this CPU)";
}

// ---------------------------------------------------------------------------
// ISA-specific entry points.  Precondition enforced, not concealed.
// ---------------------------------------------------------------------------

void layer_norm_ssve(const float* a, float* b, const float* gamma, const float* beta,
                     int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve");
    ::layer_norm_ssve(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v1(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v1");
    ::layer_norm_ssve_v1(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v2(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v2");
    ::layer_norm_ssve_v2(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v4(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v4");
    ::layer_norm_ssve_v4(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v5(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v5");
    ::layer_norm_ssve_v5(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_v6(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v6");
    ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_ssve_welford(const float* a, float* b, const float* gamma, const float* beta,
                              int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_welford");
    ::layer_norm_ssve_welford(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

// V7 needs SME2; on an SME1-only machine fall back to V6, the variant it is
// derived from, so the SME2 path stays a pure opt-in optimisation.
void layer_norm_ssve_v7(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_ssve_v7");
    if (!cpu_supports_sme2()) {
        ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::layer_norm_ssve_v7(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_za_sme2(const float* a, float* b, const float* gamma, const float* beta,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_za_sme2");
    if (!cpu_supports_sme2()) {
        ::layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::layer_norm_za_sme2(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void layer_norm_za(const float* a, float* b, const float* gamma, const float* beta,
                   int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("layer_norm_za");
    ::layer_norm_za(a, b, gamma, beta, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve(const float* a, float* b, const float* gamma,
                   int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve");
    ::rms_norm_ssve(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v1(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v1");
    ::rms_norm_ssve_v1(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v2(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v2");
    ::rms_norm_ssve_v2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v3(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v3");
    ::rms_norm_ssve_v3(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v4(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v4");
    ::rms_norm_ssve_v4(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v5(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v5");
    ::rms_norm_ssve_v5(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v6(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v6");
    ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

// V7 needs SME2 for its multi-vector loads/stores.  Where SME2 is absent but
// SME is present (an SME1-only machine), fall back to V6 — the variant V7 is
// derived from — so callers get a correct result everywhere and the SME2 path
// is a pure opt-in optimisation (ROADMAP Sprint 6: "guard the SME2 path
// behind a FEAT_SME2 runtime check so it degrades to the SME1 kernel").
void rms_norm_ssve_v7(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v7");
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_ssve_v7(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_ssve_v7x2(const float* a, float* b, const float* gamma,
                        int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_ssve_v7x2");
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_ssve_v7x2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_za_sme2(const float* a, float* b, const float* gamma,
                      int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_za_sme2");
    if (!cpu_supports_sme2()) {
        ::rms_norm_ssve_v6(a, b, gamma, m, n, ld_a, ld_b, epsilon);
        return;
    }
    ::rms_norm_za_sme2(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void rms_norm_za(const float* a, float* b, const float* gamma,
                 int64_t m, int64_t n, int64_t ld_a, int64_t ld_b, float epsilon) {
    require_sme("rms_norm_za");
    ::rms_norm_za(a, b, gamma, m, n, ld_a, ld_b, epsilon);
}

void bw_probe_ssve(float* d, const float* s, int64_t n) {
    require_sme("bw_probe_ssve");
    ::bw_probe_ssve(d, s, n);
}

// Sprint 7b.  The `empty` control is guarded too, even though a loop with no
// streaming instruction would run fine without SME: the two paths must differ
// in ONE thing (the transition), and a guard that applies to one but not the
// other would reintroduce a difference into the control.
void smstart_probe_pairs(int64_t iters) {
    require_sme("smstart_probe_pairs");
    ::smstart_probe_pairs(iters);
}

void smstart_probe_sm_only(int64_t iters) {
    require_sme("smstart_probe_sm_only");
    ::smstart_probe_sm_only(iters);
}

void smstart_probe_empty(int64_t iters) {
    require_sme("smstart_probe_empty");
    ::smstart_probe_empty(iters);
}

// Returns 0 rather than a guessed default when SME is absent: a caller that
// silently got "16" on a machine with no SME would be reintroducing exactly
// the hard-coded SVL decision D forbids.
int64_t svl_fp32_lanes() {
    if (!cpu_supports_sme()) return 0;
    return ::svl_fp32_lanes();
}

} // namespace mini_jit::norm
