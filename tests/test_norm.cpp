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

// Multi-accumulator variants (V4+) reassociate the sum-of-squares: four
// partial sums combined in tree order instead of the reference's sequential
// order.  Neither order is more correct, but a single FP32 rounding
// difference in sumsq can move the output by up to ~1.1e-5 relative (worst
// observed, N=4 where each accumulator holds exactly one term).  Those
// variants are verified against the honest tolerance they actually meet —
// documented here rather than silently widening the gate for V0-V3.
static constexpr float kTolReassoc = 2e-5f;

static void check_close(const std::vector<float>& got,
                        const std::vector<float>& ref,
                        int64_t m, int64_t n, int64_t ld,
                        float tol = kTol) {
    for (int64_t col = 0; col < n; ++col)
        for (int64_t row = 0; row < m; ++row)
            REQUIRE(got[row + col * ld] == Approx(ref[row + col * ld]).epsilon(tol));
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
// Sprint 2c — LayerNorm SSVE kernel tests
//
// Every test calls layer_norm_ssve and compares its output element-by-element
// against layer_norm_ref (the verified C++ reference from Sprint 1).
//
// All tests are guarded with cpu_supports_sme(): on CI (M1/M2) they are
// skipped gracefully; on M4 they run in full.
// ===========================================================================

// Helper: run both kernels and compare.
// Accepts separate ld_a / ld_b so we can test padding.
static void check_lnssve_vs_ref(int64_t m, int64_t n, int64_t ld_a, int64_t ld_b,
                                 float epsilon,
                                 std::vector<float>& a,
                                 std::vector<float>& gamma,
                                 std::vector<float>& beta,
                                 float tol = kTol) {
    std::vector<float> b_ref(ld_b * n, 0.0f);
    std::vector<float> b_ssve(ld_b * n, 0.0f);

    layer_norm_ref (a.data(), b_ref.data(),  gamma.data(), beta.data(),
                    m, n, ld_a, ld_b, epsilon);
    layer_norm_ssve(a.data(), b_ssve.data(), gamma.data(), beta.data(),
                    m, n, ld_a, ld_b, epsilon);

    for (int64_t col = 0; col < n; ++col)
        for (int64_t row = 0; row < m; ++row)
            REQUIRE(b_ssve[row + col * ld_b] ==
                    Approx(b_ref[row + col * ld_b]).epsilon(tol));
}

// Case 1: N < SVL (N=4, SVL=16 on M4)
// The main loop body never fires — only the tail path executes.
// This tests WHILELO + tail path in isolation.
TEST_CASE("LayerNorm SSVE: N smaller than SVL (tail-only path)", "[norm][sprint2c][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 4, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f + 0.1f * i; beta[i] = 0.05f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 7 + c * 3) % 11) - 5.0f;
    });

    check_lnssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 2: N = SVL (N=16 on M4)
// Exactly one full vector, no tail.  Tests the clean single-iteration path.
TEST_CASE("LayerNorm SSVE: N equals SVL (no tail)", "[norm][sprint2c][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 16, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 0.5f + 0.05f * i; beta[i] = -0.1f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 5) % 17) - 8.0f;
    });

    check_lnssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 3: N = SVL + 3 (N=19 on M4)
// One full vector iteration + a tail of 3 elements.
// Tests that both the full-vector and tail paths run in the same row.
TEST_CASE("LayerNorm SSVE: N = SVL + 3 (full vector + small tail)", "[norm][sprint2c][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 6, N = 19, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f; beta[i] = 0.0f; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 13 + c * 7) % 19) - 9.0f;
    });

    check_lnssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 4: multiple rows, N = 2*SVL + 5 (N=37 on M4)
// Tests the outer row loop (x20/x19 row-base advancement) and
// multiple full-vector chunks + tail within each row.
TEST_CASE("LayerNorm SSVE: multiple rows, N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 37, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 0.8f + 0.02f * i; beta[i] = 0.1f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 5 + c * 11) % 23) - 11.0f;
    });

    check_lnssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 5: padded matrix — ld_a > M and ld_b > M
// Tests that the column stride (stride_col_a = ld_a*4) correctly skips the
// padding elements when ld_a != M.
TEST_CASE("LayerNorm SSVE: padded matrix (ld_a > M, ld_b > M)", "[norm][sprint2c][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 5, N = 19, ld_a = 8, ld_b = 8;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f + 0.1f * i; beta[i] = -0.05f * i; }
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 9 + c * 4) % 13) - 6.0f;
    });

    check_lnssve_vs_ref(M, N, ld_a, ld_b, 1e-5f, a, gamma, beta);
}

// Case 6: large-magnitude stress input
// Values clustered around 1e4 with small variation (±5 around the DC offset).
// The reference accumulates in double; the SSVE kernel in FP32.
// With SHIFT=1e4, mean ≈ 1e4 and (x - mean) ≈ ±5: ~4 significant digits survive
// after cancellation.  The absolute error in the normalized output is bounded by
// roughly eps_float * SHIFT * inv_std ≈ 1.2e-7 * 1e4 * 0.45 ≈ 5e-4.
// We use an ABSOLUTE margin (not relative epsilon) because some normalized values
// are near zero, making a relative tolerance too tight for those elements.
TEST_CASE("LayerNorm SSVE: large-magnitude stress input (stability)", "[norm][sprint2c][ssve][layernorm][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 4, N = 37, ld = M;
    const float   SHIFT   = 1e4f;
    const float   kMargin = 1e-3f;  // absolute tolerance; wider than default kTol
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });

    std::vector<float> b_ref (ld * N, 0.0f);
    std::vector<float> b_ssve(ld * N, 0.0f);
    layer_norm_ref (a.data(), b_ref.data(),  gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    layer_norm_ssve(a.data(), b_ssve.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);

    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ssve[row + col * ld] ==
                    Approx(b_ref[row + col * ld]).margin(kMargin));
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
                           float eps, float tol = kTol) {
    std::vector<float> gamma(N);
    for (int64_t i = 0; i < N; ++i) gamma[i] = 0.5f + 0.1f * static_cast<float>(i % 17);
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });
    std::vector<float> b_ref(ld_b * N, 0.0f), b_ker(ld_b * N, 0.0f);
    rms_norm_ref(a.data(), b_ref.data(), gamma.data(), M, N, ld_a, ld_b, eps);
    kernel(a.data(), b_ker.data(), gamma.data(), M, N, ld_a, ld_b, eps);
    check_close(b_ker, b_ref, M, N, ld_b, tol);
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

// ---------------------------------------------------------------------------
// V4 — four independent FMLA accumulator chains in the reduction (Sprint 2b)
//
// New code paths vs V2: the 4-column main loop, the 0-3 column remainder
// loop, and the accumulator combine.  N is chosen to hit every remainder
// class (N%4 = 0,1,2,3) and the below-one-iteration case (N < 4).
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V4: small square, N multiple of 4 and of VL", "[norm][sprint2][ablation][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v4, 16, 32, 16, 16, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V4: remainder classes N%4 = 1, 2, 3", "[norm][sprint2][ablation][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v4, 8, 49, 8, 8, 1e-5f, kTolReassoc);   // 49 % 4 == 1
    check_variant(rms_norm_ssve_v4, 8, 50, 8, 8, 1e-5f, kTolReassoc);   // 50 % 4 == 2 (+ VL tail)
    check_variant(rms_norm_ssve_v4, 8, 51, 8, 8, 1e-5f, kTolReassoc);   // 51 % 4 == 3
}

TEST_CASE("RMSNorm SSVE V4: N < 4 (remainder-only, main loop skipped)", "[norm][sprint2][ablation][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v4, 16, 1, 16, 16, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v4, 16, 3, 16, 16, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V4: multi-row-block, different leading dims", "[norm][sprint2][ablation][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v4, 40, 37, 48, 40, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V4: large-magnitude stress input", "[norm][sprint2][ablation][v4][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v4);
}


// ---------------------------------------------------------------------------
// V5 — software-pipelined reduction loads (Sprint 2b)
//
// New code paths vs V4: the rotating A/B pipeline has TWO exits (group A or
// group B left in flight, depending on the number of 4-column groups), a
// preload-only path for 4 <= N < 8, and the same remainder loop.  N values
// below hit each exit and each remainder class.
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V5: pipeline exits — B-in-flight (N=32) and A-in-flight (N=28)", "[norm][sprint2][ablation][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v5, 16, 32, 16, 16, 1e-5f, kTolReassoc);  // even groups -> tail B
    check_variant(rms_norm_ssve_v5, 16, 28, 16, 16, 1e-5f, kTolReassoc);  // odd groups  -> tail A
}

TEST_CASE("RMSNorm SSVE V5: preload-only path (4 <= N < 8) and N < 4", "[norm][sprint2][ablation][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v5, 16, 4, 16, 16, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 16, 5, 16, 16, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 16, 7, 16, 16, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 16, 3, 16, 16, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 16, 1, 16, 16, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V5: remainder classes N%4 = 1, 2, 3", "[norm][sprint2][ablation][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v5, 8, 49, 8, 8, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 8, 50, 8, 8, 1e-5f, kTolReassoc);
    check_variant(rms_norm_ssve_v5, 8, 51, 8, 8, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V5: multi-row-block, different leading dims", "[norm][sprint2][ablation][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v5, 40, 37, 48, 40, 1e-5f, kTolReassoc);
}

TEST_CASE("RMSNorm SSVE V5: large-magnitude stress input", "[norm][sprint2][ablation][v5][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v5);
}


// ---------------------------------------------------------------------------
// V6 — 4-row-block contiguity grouping (Sprint 2b)
//
// New code paths vs V4: the unpredicated 4-block group loop (needs M >= 4*VL)
// and the predicated single-block tail (up to 3 blocks + partial).  On the
// M4 (VL = 16 FP32) M=64 is exactly one group; M=100 is one group + two full
// tail blocks + a 4-row partial; M<64 exercises tail-only.  V6 keeps the
// reference's sequential per-row summation order (one accumulator per row
// block), so it is verified at the strict kTol, not kTolReassoc.
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm SSVE V6: exactly one full group (M=4*VL on M4)", "[norm][sprint2][ablation][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v6, 64, 32, 64, 64, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V6: group + full tail blocks + partial block (M=100)", "[norm][sprint2][ablation][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v6, 100, 50, 100, 100, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V6: tail-only path (M < 4*VL)", "[norm][sprint2][ablation][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v6, 8, 50, 8, 8, 1e-5f);
    check_variant(rms_norm_ssve_v6, 40, 37, 48, 40, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V6: multiple groups, different leading dims, N=1", "[norm][sprint2][ablation][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant(rms_norm_ssve_v6, 192, 37, 200, 224, 1e-5f);
    check_variant(rms_norm_ssve_v6, 128, 1, 128, 128, 1e-5f);
}

TEST_CASE("RMSNorm SSVE V6: large-magnitude stress input", "[norm][sprint2][ablation][v6][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_variant_stress(rms_norm_ssve_v6);
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
