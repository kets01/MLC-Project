#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

using namespace mini_jit::norm;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill a column-major matrix [M rows x N cols, leading dim ld] with
// values produced by f(row, col).
template<typename F>
static std::vector<float> make_matrix(int64_t m, int64_t n, int64_t ld, F f) {
    std::vector<float> v(ld * n, 0.0f);
    for (int64_t col = 0; col < n; ++col)
        for (int64_t row = 0; row < m; ++row)
            v[row + col * ld] = f(row, col);
    return v;
}

// Compute LayerNorm in plain scalar double for independent ground-truth.
static std::vector<float> scalar_layer_norm(const std::vector<float>& a,
                                            const std::vector<float>& gamma,
                                            const std::vector<float>& beta,
                                            int64_t m, int64_t n, int64_t ld,
                                            float epsilon) {
    std::vector<float> out(ld * n, 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        double sum = 0.0;
        for (int64_t col = 0; col < n; ++col) sum += a[row + col * ld];
        double mean = sum / n;
        double var  = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            double d = a[row + col * ld] - mean; var += d * d;
        }
        var /= n;
        double inv_std = 1.0 / std::sqrt(var + epsilon);
        for (int64_t col = 0; col < n; ++col) {
            double xhat = (a[row + col * ld] - mean) * inv_std;
            out[row + col * ld] = static_cast<float>(xhat * gamma[col] + beta[col]);
        }
    }
    return out;
}

// Compute RMSNorm in plain scalar double.
static std::vector<float> scalar_rms_norm(const std::vector<float>& a,
                                          const std::vector<float>& gamma,
                                          int64_t m, int64_t n, int64_t ld,
                                          float epsilon) {
    std::vector<float> out(ld * n, 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        double sumsq = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            double x = a[row + col * ld]; sumsq += x * x;
        }
        double inv_rms = 1.0 / std::sqrt(sumsq / n + epsilon);
        for (int64_t col = 0; col < n; ++col) {
            out[row + col * ld] = static_cast<float>(a[row + col * ld] * inv_rms * gamma[col]);
        }
    }
    return out;
}

// Check that every element of b matches ref within the given relative+absolute
// tolerance.  We use a slightly wider margin than Approx's default (1e-5)
// because reference.cpp converts through float→double→float.
static constexpr float kTol = 1e-5f;

static void check_close(const std::vector<float>& got,
                        const std::vector<float>& ref,
                        int64_t m, int64_t n, int64_t ld) {
    for (int64_t col = 0; col < n; ++col)
        for (int64_t row = 0; row < m; ++row)
            REQUIRE(got[row + col * ld] == Approx(ref[row + col * ld]).epsilon(kTol));
}

// ---------------------------------------------------------------------------
// LayerNorm tests
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm ref: identity input (all same value)", "[norm][sprint1][layernorm]") {
    // If all x are equal, mean == x and var == 0 → output == beta.
    const int64_t M = 4, N = 8, ld = M;
    std::vector<float> gamma(N, 1.0f), beta(N, 0.5f);
    auto a = make_matrix(M, N, ld, [](int64_t, int64_t) { return 3.0f; });
    std::vector<float> b(ld * N, 0.0f);

    layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);

    // var == 0, so output should be beta (within numerical noise from eps).
    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b[row + col * ld] == Approx(0.5f).epsilon(1e-4f));
}

TEST_CASE("LayerNorm ref: normal random-ish input, matches scalar reference", "[norm][sprint1][layernorm]") {
    const int64_t M = 16, N = 32, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 0.5f + 0.1f * i; beta[i] = -0.1f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 7 + c * 13) % 17) - 8.0f;
    });
    std::vector<float> b(ld * N, 0.0f);

    layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    auto ref = scalar_layer_norm(a, gamma, beta, M, N, ld, 1e-5f);

    check_close(b, ref, M, N, ld);
}

TEST_CASE("LayerNorm ref: non-square with different ld_a and ld_b", "[norm][sprint1][layernorm]") {
    const int64_t M = 5, N = 12, ld_a = 8, ld_b = 8;
    std::vector<float> gamma(N, 2.0f), beta(N, 1.0f);
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>(r + c) * 0.3f;
    });
    std::vector<float> b(ld_b * N, 0.0f);

    layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(), M, N, ld_a, ld_b, 1e-5f);
    auto ref = scalar_layer_norm(a, gamma, beta, M, N, ld_a, 1e-5f);

    // ref was built with ld_a; b was written with ld_b (== ld_a here)
    check_close(b, ref, M, N, ld_b);
}

// Stability-stress: large-magnitude values with a shift.
// A naive single-pass variance (E[x^2] - mean^2) would lose precision here.
// The two-pass reference must stay accurate even on this input.
TEST_CASE("LayerNorm ref: large-magnitude stress input (stability)", "[norm][sprint1][layernorm][stress]") {
    const int64_t M = 8, N = 64, ld = M;
    const float   SHIFT = 1e4f;   // large DC offset
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b(ld * N, 0.0f);

    layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    auto ref = scalar_layer_norm(a, gamma, beta, M, N, ld, 1e-5f);

    check_close(b, ref, M, N, ld);
}

// ---------------------------------------------------------------------------
// RMSNorm tests
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm ref: unit vector input (gamma=1 → output == input/rms)", "[norm][sprint1][rmsnorm]") {
    const int64_t M = 4, N = 4, ld = M;
    std::vector<float> gamma(N, 1.0f);
    // Input: [1,0,0,0] per row → rms = 1/sqrt(N), output = x * sqrt(N)
    auto a = make_matrix(M, N, ld, [](int64_t, int64_t col) {
        return col == 0 ? 1.0f : 0.0f;
    });
    std::vector<float> b(ld * N, 0.0f);

    rms_norm_ref(a.data(), b.data(), gamma.data(), M, N, ld, ld, 0.0f);
    auto ref = scalar_rms_norm(a, gamma, M, N, ld, 0.0f);

    check_close(b, ref, M, N, ld);
}

TEST_CASE("RMSNorm ref: normal random-ish input, matches scalar reference", "[norm][sprint1][rmsnorm]") {
    const int64_t M = 16, N = 32, ld = M;
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 0.5f + 0.05f * i;
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });
    std::vector<float> b(ld * N, 0.0f);

    rms_norm_ref(a.data(), b.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    auto ref = scalar_rms_norm(a, gamma, M, N, ld, 1e-5f);

    check_close(b, ref, M, N, ld);
}

// Stability-stress: show where a naive single-pass approach would lose bits.
// This test documents that the reference itself is accurate; future SSVE
// kernels must also pass it.
TEST_CASE("RMSNorm ref: large-magnitude stress input (stability)", "[norm][sprint1][rmsnorm][stress]") {
    const int64_t M = 8, N = 64, ld = M;
    const float   SHIFT = 1e4f;
    std::vector<float> gamma(N, 1.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b(ld * N, 0.0f);

    rms_norm_ref(a.data(), b.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    auto ref = scalar_rms_norm(a, gamma, M, N, ld, 1e-5f);

    check_close(b, ref, M, N, ld);
}

TEST_CASE("RMSNorm ref: non-square with stride", "[norm][sprint1][rmsnorm]") {
    const int64_t M = 7, N = 16, ld_a = 8, ld_b = 8;
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 1.0f + 0.1f * i;
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>(r * c % 13) - 6.0f;
    });
    std::vector<float> b(ld_b * N, 0.0f);

    rms_norm_ref(a.data(), b.data(), gamma.data(), M, N, ld_a, ld_b, 1e-5f);
    auto ref = scalar_rms_norm(a, gamma, M, N, ld_a, 1e-5f);

    check_close(b, ref, M, N, ld_b);
}

// ===========================================================================
// Sprint 2 — RMSNorm SSVE kernel tests (V0 baseline + V1/V2/V3 ablation)
//
// Every test skips gracefully on M1/M2 CI runners (no SME).
// On M4 they run fully and verify rms_norm_ssve against rms_norm_ref.
//
// Tolerance kTol = 1e-5 (same as Sprint 1).  The SSVE kernel uses FP32
// arithmetic throughout so its error vs the double-precision reference is
// similar to rms_norm_ref itself.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: run rms_norm_ssve and compare to rms_norm_ref over the same input.
// ---------------------------------------------------------------------------
static void check_ssve_vs_ref(int64_t M, int64_t N, int64_t ld_a, int64_t ld_b,
                               float epsilon) {
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 0.5f + 0.1f * static_cast<float>(i % 17);

    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });

    std::vector<float> b_ref(ld_b * N, 0.0f);
    rms_norm_ref(a.data(), b_ref.data(), gamma.data(), M, N, ld_a, ld_b, epsilon);

    std::vector<float> b_ssve(ld_b * N, 0.0f);
    rms_norm_ssve(a.data(), b_ssve.data(), gamma.data(), M, N, ld_a, ld_b, epsilon);

    check_close(b_ssve, b_ref, M, N, ld_b);
}

// Test 1: N = multiple of 16 (M4 SVL/4) — no tail predication.  Basic gate.
TEST_CASE("RMSNorm SSVE: small square, N multiple of VL", "[norm][sprint2][ssve][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_ssve uses smstart/smstop)");
    check_ssve_vs_ref(16, 32, 16, 16, 1e-5f);
}

// Test 2: N = 50 → tail of 2 elements (50 % 16).  Exercises WHILELO partial
// predicate: the last WHILELO activates only 2 lanes, FMLA/FMUL/ST1W see a
// mask with 14 inactive lanes that must not be written or read.
TEST_CASE("RMSNorm SSVE: N not a multiple of vector width (tail predication)", "[norm][sprint2][ssve][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_ssve uses smstart/smstop)");
    check_ssve_vs_ref(8, 50, 8, 8, 1e-5f);
}

// Test 3: multiple rows, non-square, ld_a != ld_b — row-advance path.
TEST_CASE("RMSNorm SSVE: multi-row, non-square, different leading dims", "[norm][sprint2][ssve][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_ssve uses smstart/smstop)");
    check_ssve_vs_ref(7, 40, 8, 16, 1e-5f);
}

// Test 4: practical transformer shape (hidden_dim = 2048, 128 full vectors).
// Spot-check a sample of elements rather than all 131K to keep test time
// under a second; correctness of the full shape is covered by the benchmark.
TEST_CASE("RMSNorm SSVE: large shape (M=64, N=2048)", "[norm][sprint2][ssve][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_ssve uses smstart/smstop)");

    const int64_t M = 64, N = 2048, ld = M;
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 0.5f + 0.1f * static_cast<float>(i % 17);
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });
    std::vector<float> b_ref(ld * N, 0.0f), b_ssve(ld * N, 0.0f);
    rms_norm_ref (a.data(), b_ref.data(),  gamma.data(), M, N, ld, ld, 1e-5f);
    rms_norm_ssve(a.data(), b_ssve.data(), gamma.data(), M, N, ld, ld, 1e-5f);

    // Check every 8th element: covers all rows and a spread of columns.
    for (int64_t col = 0; col < N; col += 8)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ssve[row + col * ld] == Approx(b_ref[row + col * ld]).epsilon(kTol));
}

// Test 5: large-magnitude stress — same input as Sprint 1 reference stress
// test.  The SSVE kernel accumulates sumsq in FP32 (vs double in the
// reference), so the tolerance is widened to 5e-4 to account for that.
// This documents the accuracy the kernel actually achieves (decision B).
TEST_CASE("RMSNorm SSVE: large-magnitude stress input (stability)", "[norm][sprint2][ssve][rmsnorm][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_ssve uses smstart/smstop)");

    const int64_t M = 8, N = 64, ld = M;
    const float   SHIFT = 1e4f;
    std::vector<float> gamma(N, 1.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });

    std::vector<float> b_ref(ld * N, 0.0f), b_ssve(ld * N, 0.0f);
    rms_norm_ref (a.data(), b_ref.data(),  gamma.data(), M, N, ld, ld, 1e-5f);
    rms_norm_ssve(a.data(), b_ssve.data(), gamma.data(), M, N, ld, ld, 1e-5f);

    constexpr float kStressTol = 5e-4f;
    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ssve[row + col * ld] == Approx(b_ref[row + col * ld]).epsilon(kStressTol));
}

// ===========================================================================
// Sprint 2 ablation — V1, V2, V3 correctness tests
//
// All three variants share the same interface as V0 and must pass the same
// correctness and stress gates.  A generic helper drives each variant.
// ===========================================================================

using KernelFn = void(*)(const float*, float*, const float*,
                          int64_t, int64_t, int64_t, int64_t, float);

// Run kernel against rms_norm_ref on a given shape and compare within kTol.
static void check_variant(KernelFn kernel,
                           int64_t M, int64_t N, int64_t ld_a, int64_t ld_b,
                           float eps) {
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 0.5f + 0.1f * static_cast<float>(i % 17);
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });
    std::vector<float> b_ref(ld_b * N, 0.0f), b_ker(ld_b * N, 0.0f);
    rms_norm_ref(a.data(), b_ref.data(), gamma.data(), M, N, ld_a, ld_b, eps);
    kernel(a.data(), b_ker.data(), gamma.data(), M, N, ld_a, ld_b, eps);
    check_close(b_ker, b_ref, M, N, ld_b);
}

// Large-magnitude stress check — same input as the V0 stress test.
// All variants use FP32 arithmetic so the tolerance stays at 5e-4.
static void check_variant_stress(KernelFn kernel) {
    const int64_t M = 8, N = 64, ld = M;
    const float   SHIFT = 1e4f;
    std::vector<float> gamma(N, 1.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b_ref(ld * N, 0.0f), b_ker(ld * N, 0.0f);
    rms_norm_ref(a.data(), b_ref.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    kernel(a.data(), b_ker.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    constexpr float kStressTol = 5e-4f;
    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ker[row + col * ld] == Approx(b_ref[row + col * ld]).epsilon(kStressTol));
}

// ---------------------------------------------------------------------------
// V1 — FRSQRTE + FRSQRTS (no FSQRT/FDIV for inv_rms)
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V1: small square, N multiple of VL", "[norm][sprint2][ablation][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v1, 16, 32, 16, 16, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V1: N not a multiple of vector width (tail)", "[norm][sprint2][ablation][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v1, 8, 50, 8, 8, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V1: large-magnitude stress input", "[norm][sprint2][ablation][v1][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v1);
}

// ---------------------------------------------------------------------------
// V2 — V1 + pre-computed 1/N (no inner vector FDIV)
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V2: small square, N multiple of VL", "[norm][sprint2][ablation][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v2, 16, 32, 16, 16, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V2: N not a multiple of vector width (tail)", "[norm][sprint2][ablation][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v2, 8, 50, 8, 8, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V2: large-magnitude stress input", "[norm][sprint2][ablation][v2][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v2);
}

// ---------------------------------------------------------------------------
// V3 — V2 + 2x column-loop unroll in both passes
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V3: small square, N multiple of VL", "[norm][sprint2][ablation][v3]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v3, 16, 32, 16, 16, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V3: N not a multiple of vector width (tail)", "[norm][sprint2][ablation][v3]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v3, 8, 50, 8, 8, 1e-5f);
}

// Tail peel path: N=1 (single column, no pairs in the main loop).
TEST_CASE("RMSNorm SSVE V3: N=1 (peel-only, no pairs)", "[norm][sprint2][ablation][v3]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v3, 16, 1, 16, 16, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V3: large-magnitude stress input", "[norm][sprint2][ablation][v3][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v3);
}

// ===========================================================================
// Sprint 2a — roofline-probe correctness (bw_probe_ssve)
//
// The probe is a STREAM scale-add (d[i] = s[i] + 1.0f), not a norm kernel,
// but it feeds every % -of-peak figure, so it gets the same treatment: verify
// the streaming-mode LD1W/ST1W path against the scalar loop before trusting
// any bandwidth number derived from it.  Scale-add is a single exact FP32
// operation, so vector and scalar results must match bit-for-bit (== 0 diff).
// ===========================================================================

static void check_probe(int64_t n) {
    std::vector<float> s(n), d(n, 0.0f), expect(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) s[i] = static_cast<float>((i % 251) - 125) * 0.25f;
    for (int64_t i = 0; i < n; ++i) expect[i] = s[i] + 1.0f;

    bw_probe_ssve(d.data(), s.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        INFO("i=" << i);
        REQUIRE(d[i] == expect[i]);
    }
}

// Covers the 4-vector main loop, the per-VL tail loop, and a partial final
// predicate — n values chosen to hit all three regardless of SVL.
TEST_CASE("BW probe SSVE: main loop + tail + partial predicate", "[norm][sprint2][roofline]") {
    if (!cpu_supports_sme()) SKIP("SME required (bw_probe_ssve uses smstart/smstop)");
    check_probe(1024);   // multiple of 4*VL for any SVL up to 2048 bits
    check_probe(259);    // odd: main loop + tail + partial last predicate
    check_probe(3);      // below one vector: tail-only path
}

TEST_CASE("BW probe SSVE: n = 0 is a safe no-op", "[norm][sprint2][roofline]") {
    if (!cpu_supports_sme()) SKIP("SME required (bw_probe_ssve uses smstart/smstop)");
    std::vector<float> s(4, 1.0f), d(4, -7.0f);
    bw_probe_ssve(d.data(), s.data(), 0);
    for (int i = 0; i < 4; ++i) REQUIRE(d[i] == -7.0f);  // untouched
}
