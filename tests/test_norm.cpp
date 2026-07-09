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

// FRSQRTE+NR vs IEEE FSQRT+FDIV: one NR step (FRSQRTS) gives ~5e-6 relative
// accuracy in inv_std (half the squared initial FRSQRTE error of ~2^-8.25).
// LayerNorm adds a beta term that can partially cancel gamma*x_hat.  When
// output y is near zero, a relative epsilon is inappropriate because the
// absolute error in y from the inv_std approximation stays bounded even as y
// approaches zero.  Bound: |delta_y| ≤ gamma * |x_hat| * 5e-6.  For our test
// shapes (gamma ≤ 1.5, |x_hat| ≤ 3) this is ≤ ~2.25e-5; worst observed in
// practice is ~5e-6.  We use an absolute margin of 5e-5 — documented here so
// it is clear this is the actual accuracy of FRSQRTE+NR, not a silenced gate.
static constexpr float kAbsMarginNR = 5e-5f;

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
// Sprint 2c ablation — LayerNorm SSVE V1 tests
//
// V1 replaces FSQRT+FDIV with FRSQRTE+NR in pass 2.  All other passes are
// identical to V0.  The same six correctness cases are used so a regression
// in any pass is caught even when V1 is only expected to change pass 2.
// ===========================================================================

using LNKernelFn = void(*)(const float*, float*, const float*, const float*,
                            int64_t, int64_t, int64_t, int64_t, float);

static void check_ln_variant(LNKernelFn kernel,
                              int64_t m, int64_t n, int64_t ld_a, int64_t ld_b,
                              float epsilon) {
    std::vector<float> gamma(n), beta(n);
    for (int64_t i = 0; i < n; ++i) { gamma[i] = 0.5f + 0.05f * static_cast<float>(i % 17);
                                       beta[i]  = 0.1f  * static_cast<float>(i % 13) - 0.3f; }
    auto a = make_matrix(m, n, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
    });
    std::vector<float> b_ref(ld_b * n, 0.0f), b_ker(ld_b * n, 0.0f);
    layer_norm_ref(a.data(), b_ref.data(), gamma.data(), beta.data(), m, n, ld_a, ld_b, epsilon);
    kernel(a.data(), b_ker.data(), gamma.data(), beta.data(), m, n, ld_a, ld_b, epsilon);
    // Absolute margin: see kAbsMarginNR comment above.
    for (int64_t col = 0; col < n; ++col)
        for (int64_t row = 0; row < m; ++row)
            REQUIRE(b_ker[row + col * ld_b] ==
                    Approx(b_ref[row + col * ld_b]).margin(kAbsMarginNR));
}

TEST_CASE("LayerNorm SSVE V1: N smaller than SVL (tail-only path)", "[norm][sprint2c][ssve][layernorm][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v1, 8, 4, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V1: N equals SVL (no tail)", "[norm][sprint2c][ssve][layernorm][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v1, 8, 16, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V1: N = SVL + 3 (full vector + small tail)", "[norm][sprint2c][ssve][layernorm][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v1, 6, 19, 6, 6, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V1: multiple rows, N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v1, 8, 37, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V1: padded matrix (ld_a > M, ld_b > M)", "[norm][sprint2c][ssve][layernorm][v1]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v1, 5, 19, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V1: large-magnitude stress input (stability)", "[norm][sprint2c][ssve][layernorm][v1][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 4, N = 37, ld = M;
    const float   SHIFT   = 1e4f;
    const float   kMargin = 1e-3f;
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b_ref (ld * N, 0.0f);
    std::vector<float> b_v1  (ld * N, 0.0f);
    layer_norm_ref      (a.data(), b_ref.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    layer_norm_ssve_v1  (a.data(), b_v1.data(),  gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);

    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_v1[row + col * ld] ==
                    Approx(b_ref[row + col * ld]).margin(kMargin));
}

// ===========================================================================
// Sprint 2c ablation — LayerNorm SSVE V2, V4, V5, V6, Welford tests
//
// All variants are verified against layer_norm_ref using check_ln_variant
// (absolute margin kAbsMarginNR = 5e-5).  V4/V5 add boundary-case tests for
// the 4-column loop (N < 4, N%4 remainder classes).  V6 adds tests for the
// group path (M = 4*VL = 64 on M4) and the tail-only path (M < 64).
// ===========================================================================

// ---------------------------------------------------------------------------
// V2 — pre-computed 1/N; RMSNorm-style ADDVL/INCW outer loop
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm SSVE V2: N smaller than SVL", "[norm][sprint2c][ssve][layernorm][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v2, 8, 4, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V2: N equals SVL (no tail)", "[norm][sprint2c][ssve][layernorm][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v2, 8, 16, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V2: N = SVL + 3", "[norm][sprint2c][ssve][layernorm][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v2, 6, 19, 6, 6, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V2: multiple rows, N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v2, 8, 37, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V2: padded matrix (ld > M)", "[norm][sprint2c][ssve][layernorm][v2]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v2, 5, 19, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V2: large-magnitude stress input", "[norm][sprint2c][ssve][layernorm][v2][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    const int64_t M = 4, N = 37, ld = M;
    const float SHIFT = 1e4f;
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b_ref(ld * N, 0.0f), b_ker(ld * N, 0.0f);
    layer_norm_ref(a.data(), b_ref.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    layer_norm_ssve_v2(a.data(), b_ker.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ker[row + col * ld] == Approx(b_ref[row + col * ld]).margin(1e-3f));
}

// ---------------------------------------------------------------------------
// V4 — four independent accumulator chains in both reduction passes
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm SSVE V4: N smaller than SVL", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 8, 4, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V4: N equals SVL", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 8, 16, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V4: N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 8, 37, 8, 8, 1e-5f);
}

// N%4 remainder classes — each exercises a different remainder-loop iteration count.
TEST_CASE("LayerNorm SSVE V4: N%4 remainder classes (1, 2, 3)", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 8, 33, 8, 8, 1e-5f);   // N%4 == 1
    check_ln_variant(layer_norm_ssve_v4, 8, 34, 8, 8, 1e-5f);   // N%4 == 2
    check_ln_variant(layer_norm_ssve_v4, 8, 35, 8, 8, 1e-5f);   // N%4 == 3
}

// N < 4: the 4-column main loop is skipped entirely (remainder loop only).
TEST_CASE("LayerNorm SSVE V4: N < 4 (remainder-only, main loop skipped)", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 8, 1, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v4, 8, 3, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V4: padded matrix (ld > M)", "[norm][sprint2c][ssve][layernorm][v4]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v4, 5, 37, 8, 8, 1e-5f);
}

// ---------------------------------------------------------------------------
// V5 — software-pipelined loads in both reduction passes
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm SSVE V5: N smaller than SVL", "[norm][sprint2c][ssve][layernorm][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v5, 8, 4, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V5: N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v5, 8, 37, 8, 8, 1e-5f);
}

// Pipeline boundary: preload-only path (4 <= N < 8) and N < 4.
TEST_CASE("LayerNorm SSVE V5: pipeline boundary (4 <= N < 8) and N < 4", "[norm][sprint2c][ssve][layernorm][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v5, 8, 4, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 5, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 7, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 3, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 1, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V5: N%4 remainder classes", "[norm][sprint2c][ssve][layernorm][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v5, 8, 33, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 34, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v5, 8, 35, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE V5: padded matrix (ld > M)", "[norm][sprint2c][ssve][layernorm][v5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v5, 5, 37, 8, 8, 1e-5f);
}

// ---------------------------------------------------------------------------
// V6 — 4-row-block contiguity grouping
//
// On M4 (VL=16): 4*VL = 64.  Tests below hit the group path (M >= 64),
// the group+tail path (M=100), and the tail-only path (M < 64).
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm SSVE V6: tail-only path (M < 4*VL)", "[norm][sprint2c][ssve][layernorm][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_v6, 8, 4, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v6, 8, 37, 8, 8, 1e-5f);
    check_ln_variant(layer_norm_ssve_v6, 5, 19, 8, 8, 1e-5f);
}

// Exactly one full group: M = 4*VL = 64 on M4.  Exercises the unpredicated
// group path (p0) with no tail blocks.
TEST_CASE("LayerNorm SSVE V6: exactly one full group (M = 4*VL)", "[norm][sprint2c][ssve][layernorm][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    {
        const int64_t M = 64, N = 32, ld = M;
        std::vector<float> gamma(N), beta(N);
        for (int64_t i = 0; i < N; ++i) {
            gamma[i] = 0.5f + 0.05f * static_cast<float>(i % 17);
            beta[i]  = 0.1f  * static_cast<float>(i % 13) - 0.3f;
        }
        auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
            return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
        });
        std::vector<float> b_ref(ld * N, 0.0f), b_ker(ld * N, 0.0f);
        layer_norm_ref(a.data(), b_ref.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
        layer_norm_ssve_v6(a.data(), b_ker.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
        for (int64_t col = 0; col < N; ++col)
            for (int64_t row = 0; row < M; ++row)
                REQUIRE(b_ker[row + col * ld] ==
                        Approx(b_ref[row + col * ld]).margin(kAbsMarginNR));
    }
}

// Group + tail: M=100 means one full group (64 rows) + 36-row tail on M4.
TEST_CASE("LayerNorm SSVE V6: group + tail blocks (M=100)", "[norm][sprint2c][ssve][layernorm][v6]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    {
        const int64_t M = 100, N = 50, ld = M;
        std::vector<float> gamma(N), beta(N);
        for (int64_t i = 0; i < N; ++i) {
            gamma[i] = 0.5f + 0.05f * static_cast<float>(i % 17);
            beta[i]  = 0.1f  * static_cast<float>(i % 13) - 0.3f;
        }
        auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
            return static_cast<float>((r * 11 + c * 7) % 19) - 9.0f;
        });
        std::vector<float> b_ref(ld * N, 0.0f), b_ker(ld * N, 0.0f);
        layer_norm_ref(a.data(), b_ref.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
        layer_norm_ssve_v6(a.data(), b_ker.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
        for (int64_t col = 0; col < N; ++col)
            for (int64_t row = 0; row < M; ++row)
                REQUIRE(b_ker[row + col * ld] ==
                        Approx(b_ref[row + col * ld]).margin(kAbsMarginNR));
    }
}

// ---------------------------------------------------------------------------
// Welford — online single-pass mean+variance
//
// The Welford algorithm is numerically stable but uses FP32 throughout.
// For our test shapes the error is within kAbsMarginNR.  The Welford kernel
// reduces traffic from 3R+1W to 2R+1W while incurring N scalar FDIVs
// per block; tests verify correctness, not performance.
// ---------------------------------------------------------------------------

TEST_CASE("LayerNorm SSVE Welford: N smaller than SVL", "[norm][sprint2c][ssve][layernorm][welford]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_welford, 8, 4, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE Welford: N equals SVL", "[norm][sprint2c][ssve][layernorm][welford]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_welford, 8, 16, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE Welford: N = 2*SVL + 5", "[norm][sprint2c][ssve][layernorm][welford]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_welford, 8, 37, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE Welford: padded matrix (ld > M)", "[norm][sprint2c][ssve][layernorm][welford]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_welford, 5, 19, 8, 8, 1e-5f);
}

TEST_CASE("LayerNorm SSVE Welford: N=1 (single column, no loop body)", "[norm][sprint2c][ssve][layernorm][welford]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant(layer_norm_ssve_welford, 8, 1, 8, 8, 1e-5f);
}

// ---------------------------------------------------------------------------
// Stress coverage for the remaining LayerNorm variants (Sprint 2c gap-fill).
//
// V0/V1/V2 already have large-magnitude stress cases; V4/V5/V6 change the
// reduction structure (multi-accumulator, pipelining, 4-block grouping) and
// Welford changes the ALGORITHM — exactly the cases where cancellation
// behaviour could differ, so each gets the same SHIFT=1e4 input.  Welford's
// claim to numerical stability (vs the catastrophic naive E[x2]-mean2) is
// verified here, not assumed: the ablation verdict "slower than two-pass"
// is only meaningful if its accuracy actually holds.
// ---------------------------------------------------------------------------

static void check_ln_variant_stress(LNKernelFn kernel) {
    const int64_t M = 4, N = 37, ld = M;
    const float   SHIFT   = 1e4f;
    const float   kMargin = 1e-3f;  // same bound as the V0 stress case
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    auto a = make_matrix(M, N, ld, [&](int64_t r, int64_t c) {
        return SHIFT + static_cast<float>((r * 3 + c * 5) % 11) - 5.0f;
    });
    std::vector<float> b_ref(ld * N, 0.0f), b_ker(ld * N, 0.0f);
    layer_norm_ref(a.data(), b_ref.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    kernel(a.data(), b_ker.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < M; ++row)
            REQUIRE(b_ker[row + col * ld] ==
                    Approx(b_ref[row + col * ld]).margin(kMargin));
}

TEST_CASE("LayerNorm SSVE V4: large-magnitude stress input", "[norm][sprint2c][ssve][layernorm][v4][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant_stress(layer_norm_ssve_v4);
}

TEST_CASE("LayerNorm SSVE V5: large-magnitude stress input", "[norm][sprint2c][ssve][layernorm][v5][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant_stress(layer_norm_ssve_v5);
}

TEST_CASE("LayerNorm SSVE V6: large-magnitude stress input", "[norm][sprint2c][ssve][layernorm][v6][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant_stress(layer_norm_ssve_v6);
}

TEST_CASE("LayerNorm SSVE Welford: large-magnitude stress input (the stability claim)", "[norm][sprint2c][ssve][layernorm][welford][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_ln_variant_stress(layer_norm_ssve_welford);
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
// Sprint 3 — RMSNorm ZA-tile residency kernel (rms_norm_za)
//
// The ZA-resident fast path is taken when the row fits in ZA (N <= 4*SVL;
// 64 on the M4's 512-bit SVL).  These cases deliberately walk the tile
// boundaries so the four ZA_LOAD/ZA_STORE macro expansions and the
// column/row tails are all exercised; the N > 4*SVL cases take the
// streaming fallback.  rms_norm_za shares the canonical signature, so the
// Sprint-2 check_variant / check_variant_stress helpers drive it directly.
//
// The kernel sums squares in a single accumulator in strict column order
// (reference order), so the standard kTol = 1e-5 applies — no reassociation
// widening.  All cases skip on CI (no SME) and run fully on the M4.
// ===========================================================================

// ZA path, one tile: N < SVL (column tail inside tile 0) and N == SVL.
TEST_CASE("RMSNorm ZA: single tile, N < SVL and N == SVL", "[norm][sprint3][za][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_za uses smstart/smstop + ZA)");
    check_variant(rms_norm_za, 16, 8,  16, 16, 1e-5f);   // N < SVL: tile 0 partial
    check_variant(rms_norm_za, 16, 16, 16, 16, 1e-5f);   // N == SVL: tile 0 full
}

// ZA path, multiple tiles: N spanning 2-3 tiles with a partial last tile,
// and N == 4*SVL exactly (all four tiles full — the residency boundary).
TEST_CASE("RMSNorm ZA: multi-tile, partial last tile and 4*SVL boundary", "[norm][sprint3][za][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_za uses smstart/smstop + ZA)");
    check_variant(rms_norm_za, 16, 24, 16, 16, 1e-5f);   // tiles 0 full + 1 partial
    check_variant(rms_norm_za, 16, 40, 16, 16, 1e-5f);   // tiles 0,1 full + 2 partial
    check_variant(rms_norm_za, 16, 64, 16, 16, 1e-5f);   // all four tiles full (=4*SVL)
}

// ZA path with a row tail: M not a multiple of SVL exercises the WHILELO
// row predicate through the mova into/out of ZA (inactive lanes must not be
// stored), across several row blocks and mismatched leading dims.
TEST_CASE("RMSNorm ZA: row tail + multiple row blocks + mismatched ld", "[norm][sprint3][za][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_za uses smstart/smstop + ZA)");
    check_variant(rms_norm_za, 7,  40, 8,   8,  1e-5f);  // single partial block
    check_variant(rms_norm_za, 100, 48, 128, 112, 1e-5f); // full blocks + row tail, ld>M
}

// Fallback path: N just past the ZA capacity (4*SVL + 1) and a realistic
// large transformer width — both take the streaming two-pass, so correctness
// there must hold too even though ZA is not used.
TEST_CASE("RMSNorm ZA: fallback path (N > 4*SVL)", "[norm][sprint3][za][rmsnorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_za uses smstart/smstop + ZA)");
    check_variant(rms_norm_za, 16, 65,   16, 16, 1e-5f); // one past capacity
    check_variant(rms_norm_za, 64, 2048, 64, 64, 1e-5f); // large width, fallback
}

// Stress: large-magnitude shifted input (N=64 -> ZA path).  Same 5e-4 FP32
// tolerance as the SSVE variants (sumsq accumulates in FP32 vs the double ref).
TEST_CASE("RMSNorm ZA: large-magnitude stress input (stability)", "[norm][sprint3][za][rmsnorm][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required (rms_norm_za uses smstart/smstop + ZA)");
    check_variant_stress(rms_norm_za);
}


// ===========================================================================
// Sprint 3 — LayerNorm ZA-tile residency kernel (layer_norm_za), gated
//
// Unlike rms_norm_za (2R+1W -> 1R+1W), this prototype stages x in ZA once
// during the mean pass and reuses it from ZA for BOTH the variance pass and
// the normalize pass: a full 3R+1W -> 1R+1W fusion. Same tile-boundary
// walk as the RMSNorm ZA suite (single tile, multi-tile, the 4*SVL
// residency boundary, row tails, mismatched leading dims, and the N > 4*SVL
// fallback), reusing the Sprint-2c check_ln_variant / check_ln_variant_stress
// helpers since layer_norm_za shares the canonical LayerNorm signature.
//
// Mean and variance each use a single accumulator in strict column order
// (reference order) across all four tile sections, so no reassociation
// tolerance is needed beyond the FRSQRTE+NR margin already used by V1/V6
// (kAbsMarginNR). All cases skip on CI (no SME) and run fully on the M4.
// ===========================================================================

// ZA path, one tile: N < SVL (column tail inside tile 0) and N == SVL.
TEST_CASE("LayerNorm ZA: single tile, N < SVL and N == SVL", "[norm][sprint3][za][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (layer_norm_za uses smstart/smstop + ZA)");
    check_ln_variant(layer_norm_za, 16, 8,  16, 16, 1e-5f);   // N < SVL: tile 0 partial
    check_ln_variant(layer_norm_za, 16, 16, 16, 16, 1e-5f);   // N == SVL: tile 0 full
}

// ZA path, multiple tiles: N spanning 2-3 tiles with a partial last tile,
// and N == 4*SVL exactly (all four tiles full — the residency boundary).
TEST_CASE("LayerNorm ZA: multi-tile, partial last tile and 4*SVL boundary", "[norm][sprint3][za][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (layer_norm_za uses smstart/smstop + ZA)");
    check_ln_variant(layer_norm_za, 16, 24, 16, 16, 1e-5f);   // tiles 0 full + 1 partial
    check_ln_variant(layer_norm_za, 16, 40, 16, 16, 1e-5f);   // tiles 0,1 full + 2 partial
    check_ln_variant(layer_norm_za, 16, 64, 16, 16, 1e-5f);   // all four tiles full (=4*SVL)
}

// ZA path with a row tail: M not a multiple of SVL exercises the WHILELO
// row predicate through the mova into/out of ZA, across several row blocks
// and mismatched leading dims.
TEST_CASE("LayerNorm ZA: row tail + multiple row blocks + mismatched ld", "[norm][sprint3][za][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (layer_norm_za uses smstart/smstop + ZA)");
    check_ln_variant(layer_norm_za, 7,  40, 8,   8,  1e-5f);   // single partial block
    check_ln_variant(layer_norm_za, 100, 48, 128, 112, 1e-5f); // full blocks + row tail, ld>M
}

// Fallback path: N just past the ZA capacity (4*SVL + 1) and a realistic
// large transformer width — both take the streaming three-pass, so
// correctness there must hold too even though ZA is not used.
TEST_CASE("LayerNorm ZA: fallback path (N > 4*SVL)", "[norm][sprint3][za][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required (layer_norm_za uses smstart/smstop + ZA)");
    check_ln_variant(layer_norm_za, 16, 65,   16, 16, 1e-5f);  // one past capacity
    check_ln_variant(layer_norm_za, 64, 2048, 64, 64, 1e-5f);  // large width, fallback
}

// Stress: large-magnitude shifted input (N=37 -> ZA path on the M4).
TEST_CASE("LayerNorm ZA: large-magnitude stress input (stability)", "[norm][sprint3][za][layernorm][stress]") {
    if (!cpu_supports_sme()) SKIP("SME required (layer_norm_za uses smstart/smstop + ZA)");
    check_ln_variant_stress(layer_norm_za);
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


// ===========================================================================
// Regression — eps register-stash clobber (found during Sprint 3, LayerNorm ZA)
//
// Every SSVE/ZA kernel stashed epsilon in the callee-saved S8/D8 register
// before SMSTART, intending to read it back afterwards. Two shapes of that
// pattern shipped, and BOTH were broken:
//   - RMSNorm V1-V6 and rms_norm_za reloaded eps from [sp,#NN] — but that
//     stack slot held the CALLER's old d8 (saved by "str d8" *before* the
//     eps fmov overwrote the register), never eps itself.
//   - RMSNorm V0 read eps directly back from S8/Z8 after SMSTART, no stack
//     detour at all — but SMSTART clobbers D8 too, not just D9-D15 as
//     assumed, so the register itself doesn't survive the transition either.
// Every existing tolerance-based test passed regardless, because eps's
// effect on non-degenerate data is far below the FP32 comparison tolerance.
// The bug is only observable when sumsq/variance is EXACTLY zero, so eps is
// the only thing standing between a valid reciprocal and 1/0 = inf, and
// inf * 0 = NaN propagates through the normalize step.
//
// The fix (all 13 affected kernels): stash eps to a dedicated stack slot
// BEFORE smstart and reload from that same address after — memory, unlike
// any register, is unaffected by a PSTATE.SM transition. These cases lock
// that fix in place: an all-zero RMSNorm row (sumsq == 0 exactly) and an
// all-constant LayerNorm row (var == 0 exactly) must produce a finite,
// correct result, not NaN.
// ===========================================================================

TEST_CASE("RMSNorm SSVE kernels: all-zero row is finite (eps regression)",
          "[norm][sprint3][regression][eps]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    const int64_t M = 16, N = 16, ld = M;
    std::vector<float> a(ld * N, 0.0f);          // sumsq == 0 exactly per row
    std::vector<float> gamma(N, 1.0f);

    auto check = [&](KernelFn kernel, const char* name) {
        std::vector<float> b(ld * N, 123.0f);
        kernel(a.data(), b.data(), gamma.data(), M, N, ld, ld, 1e-5f);
        for (int64_t i = 0; i < ld * N; ++i) {
            INFO(name << " b[" << i << "]=" << b[i]);
            REQUIRE(std::isfinite(b[i]));
            REQUIRE(b[i] == Approx(0.0f).margin(1e-6f));
        }
    };
    check(rms_norm_ssve,    "rms_norm_ssve");
    check(rms_norm_ssve_v1, "rms_norm_ssve_v1");
    check(rms_norm_ssve_v2, "rms_norm_ssve_v2");
    check(rms_norm_ssve_v3, "rms_norm_ssve_v3");
    check(rms_norm_ssve_v4, "rms_norm_ssve_v4");
    check(rms_norm_ssve_v5, "rms_norm_ssve_v5");
    check(rms_norm_ssve_v6, "rms_norm_ssve_v6");
    check(rms_norm_za,      "rms_norm_za");
}

TEST_CASE("LayerNorm SSVE kernels: all-constant row is finite (eps regression)",
          "[norm][sprint3][regression][eps]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    const int64_t M = 16, N = 16, ld = M;
    std::vector<float> a(ld * N, 3.0f);          // var == 0 exactly per row
    std::vector<float> gamma(N, 1.0f), beta(N, 0.5f);

    auto check = [&](LNKernelFn kernel, const char* name) {
        std::vector<float> b(ld * N, 123.0f);
        kernel(a.data(), b.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
        for (int64_t i = 0; i < ld * N; ++i) {
            INFO(name << " b[" << i << "]=" << b[i]);
            REQUIRE(std::isfinite(b[i]));
            REQUIRE(b[i] == Approx(0.5f).margin(1e-6f));   // (x-mean)=0 -> output == beta
        }
    };
    check(layer_norm_ssve,          "layer_norm_ssve");
    check(layer_norm_ssve_v1,       "layer_norm_ssve_v1");
    check(layer_norm_ssve_v2,       "layer_norm_ssve_v2");
    check(layer_norm_ssve_v4,       "layer_norm_ssve_v4");
    check(layer_norm_ssve_v5,       "layer_norm_ssve_v5");
    check(layer_norm_ssve_v6,       "layer_norm_ssve_v6");
    check(layer_norm_ssve_welford,  "layer_norm_ssve_welford");
}

// ===========================================================================
// Sprint 4 — mini_jit::Norm JIT generator
//
// Three layers of verification, weakest to strongest:
//   1. [encoders]      every new InstGen encoder reproduces the golden word
//                      the toolchain assembled for the same instruction
//                      (words taken from objdump of rms/layer_norm_ssve_v6.o).
//   2. [encoding-diff] the generator's whole buffer is diffed word-by-word
//                      against the LINKED hand-written kernel, read straight
//                      from its function address at runtime — so the diff is
//                      always against what the toolchain actually assembled,
//                      never a stale copy.  A green diff proves the JIT
//                      kernel byte-identical to a kernel that already passed
//                      the full Sprint-2/3 suite.
//   3. [jit]           the emitted kernel is executed and verified against
//                      the C++ reference anyway (SME-guarded) — belt and
//                      suspenders, and it also validates the JitEngine
//                      buffer path (mmap/JIT-protect/icache-invalidate).
//
// 1 and 2 are host-portable and run on CI; 3 skips without SME.
// ===========================================================================

#include "norm/jit_norm.hpp"
#include "week6/Instgen.hpp"
#include <cstring>

// The raw assembled kernels (not the SME-guard wrappers in mini_jit::norm) —
// the encoding diff reads their instruction words from the linked binary.
extern "C" void rms_norm_ssve_v6(const float*, float*, const float*,
                                 int64_t, int64_t, int64_t, int64_t, float);
extern "C" void layer_norm_ssve_v6(const float*, float*, const float*, const float*,
                                   int64_t, int64_t, int64_t, int64_t, float);

using IG = mini_jit::InstGen;
using mini_jit::Norm;

TEST_CASE("Sprint4 InstGen: base loads/stores match golden words", "[norm][sprint4][encoders]") {
    // golden words: objdump of rms_norm_ssve_v6.o / layer_norm_ssve_v6.o
    REQUIRE(IG::base_stp_pre_x(IG::x19, IG::x20, IG::sp, -80) == 0xa9bb53f3u); // stp x19,x20,[sp,#-0x50]!
    REQUIRE(IG::base_stp_off_x(IG::x21, IG::x22, IG::sp,  16) == 0xa9015bf5u); // stp x21,x22,[sp,#0x10]
    REQUIRE(IG::base_stp_off_x(IG::x25, IG::x26, IG::sp,  48) == 0xa9036bf9u); // stp x25,x26,[sp,#0x30]
    REQUIRE(IG::base_ldp_off_x(IG::x23, IG::x24, IG::sp,  32) == 0xa94263f7u); // ldp x23,x24,[sp,#0x20]
    REQUIRE(IG::base_ldp_post_x(IG::x19, IG::x20, IG::sp, 80) == 0xa8c553f3u); // ldp x19,x20,[sp],#0x50
    REQUIRE(IG::base_str_imm_x(IG::x25, IG::sp, 48) == 0xf9001bf9u);           // str x25,[sp,#0x30]
    REQUIRE(IG::base_ldr_imm_x(IG::x25, IG::sp, 48) == 0xf9401bf9u);           // ldr x25,[sp,#0x30]
    REQUIRE(IG::simd_str_imm_d(IG::v8, IG::sp, 56)  == 0xfd001fe8u);           // str d8,[sp,#0x38]
    REQUIRE(IG::simd_ldr_imm_d(IG::v8, IG::sp, 56)  == 0xfd401fe8u);           // ldr d8,[sp,#0x38]
    REQUIRE(IG::simd_str_imm_s(IG::v0, IG::sp, 64)  == 0xbd0043e0u);           // str s0,[sp,#0x40]
    REQUIRE(IG::simd_ldr_imm_s(IG::v0, IG::sp, 64)  == 0xbd4043e0u);           // ldr s0,[sp,#0x40]
}

TEST_CASE("Sprint4 InstGen: base moves/branches match golden words", "[norm][sprint4][encoders]") {
    REQUIRE(IG::base_mov_reg_x(IG::x19, IG::x0) == 0xaa0003f3u);               // mov x19,x0
    REQUIRE(IG::base_mov_reg_x(IG::x0,  IG::x6) == 0xaa0603e0u);               // mov x0,x6
    REQUIRE(IG::base_cmp_reg_x(IG::x6, IG::x22) == 0xeb1600dfu);               // cmp x6,x22
    REQUIRE(IG::base_b_cond(IG::cond_hi,  72) == 0x54000908u);                 // b.hi +72
    REQUIRE(IG::base_b_cond(IG::cond_eq,  35) == 0x54000460u);                 // b.eq +35 (b.none)
    REQUIRE(IG::base_b_ne(-10)            == 0x54fffec1u);                     // b.ne -10
    REQUIRE(IG::base_b(-73)               == 0x17ffffb7u);                     // b -73
    REQUIRE(IG::base_br_cbz_x(IG::x22, 123) == 0xb4000f76u);                   // cbz x22,+123
    // existing encoders reused by the generator, pinned here too
    REQUIRE(IG::base_lsl_x(IG::x24, IG::x5, 2)     == 0xd37ef4b8u);            // lsl x24,x5,#2
    REQUIRE(IG::base_movz_x(IG::x0, 0)             == 0xd2800000u);            // mov x0,#0
    REQUIRE(IG::base_add_reg_x(IG::x6, IG::x0, IG::x14) == 0x8b0e0006u);       // add x6,x0,x14
    REQUIRE(IG::base_add_imm_x(IG::x10, IG::x10, 4)     == 0x9100114au);       // add x10,x10,#4
    REQUIRE(IG::base_subs_imm_x(IG::x9, IG::x9, 1)      == 0xf1000529u);       // subs x9,x9,#1
}

TEST_CASE("Sprint4 InstGen: fp scalar ops match golden words", "[norm][sprint4][encoders]") {
    REQUIRE(IG::fp_fmov_s(IG::v8, IG::v0)      == 0x1e204008u);                // fmov s8,s0
    REQUIRE(IG::fp_fmov_imm_s(IG::v0, 0x70)    == 0x1e2e1000u);                // fmov s0,#1.0
    REQUIRE(IG::fp_scvtf_s_x(IG::v1, IG::x23)  == 0x9e2202e1u);                // scvtf s1,x23
    REQUIRE(IG::fp_fdiv_s(IG::v0, IG::v0, IG::v1) == 0x1e211800u);             // fdiv s0,s0,s1
}

TEST_CASE("Sprint4 InstGen: SVE/SME ops match golden words", "[norm][sprint4][encoders]") {
    REQUIRE(IG::sme_smstart_sm_only() == 0xd503437fu);                         // smstart sm
    REQUIRE(IG::sme_smstop_sm_only()  == 0xd503427fu);                         // smstop sm
    // the Sprint-4 ptrue fix: fp32 must be .S (size=10), not .B
    REQUIRE(IG::sve_ptrue_all(IG::p0, IG::dtype_t::fp32) == 0x2598e3e0u);      // ptrue p0.s
    REQUIRE(IG::sve_cntw_x(IG::x14)   == 0x04a0e3eeu);                         // cntw x14
    REQUIRE(IG::sve_incw_x(IG::x0)    == 0x04b0e3e0u);                         // incw x0
    REQUIRE(IG::sve_addvl(IG::x19, IG::x19, 4) == 0x04335093u);                // addvl x19,x19,#4
    REQUIRE(IG::sve_addvl(IG::x20, IG::x20, 1) == 0x04345034u);                // addvl x20,x20,#1
    REQUIRE(IG::sve_dup_imm_s(IG::z2, 0)       == 0x25b8c002u);                // mov z2.s,#0
    REQUIRE(IG::sve_dup_elem_s(IG::z8, IG::z0, 0) == 0x05242008u);             // mov z8.s,s0
    REQUIRE(IG::sve_dup_elem_s(IG::z3, IG::z8, 0) == 0x05242103u);             // mov z3.s,s8
    REQUIRE(IG::sve_ld1w_imm(IG::z0,  IG::p0, IG::x8, 0) == 0xa540a100u);      // ld1w {z0.s},p0/z,[x8]
    REQUIRE(IG::sve_ld1w_imm(IG::z24, IG::p0, IG::x8, 1) == 0xa541a118u);      // ld1w {z24.s},[x8,#1,mul vl]
    REQUIRE(IG::sve_ld1w_imm(IG::z0,  IG::p1, IG::x8, 0) == 0xa540a500u);      // ld1w {z0.s},p1/z,[x8]
    REQUIRE(IG::sve_st1w_imm(IG::z26, IG::p0, IG::x9, 3) == 0xe543e13au);      // st1w {z26.s},[x9,#3,mul vl]
    REQUIRE(IG::sve_st1w_imm(IG::z0,  IG::p1, IG::x9, 0) == 0xe540e520u);      // st1w {z0.s},p1,[x9]
    REQUIRE(IG::sve_ld1rw_s(IG::z1, IG::p0, IG::x10) == 0x8540c141u);          // ld1rw {z1.s},p0/z,[x10]
    REQUIRE(IG::sve_ld1rw_s(IG::z1, IG::p1, IG::x10) == 0x8540c541u);          // ld1rw {z1.s},p1/z,[x10]
    REQUIRE(IG::sve_fmla_s(IG::z16, IG::p0, IG::z24, IG::z24) == 0x65b80310u); // fmla z16.s,p0/m,z24.s,z24.s
    REQUIRE(IG::sve_fmla_s(IG::z2,  IG::p1, IG::z0,  IG::z0)  == 0x65a00402u); // fmla z2.s,p1/m,z0.s,z0.s
    REQUIRE(IG::sve_fadd_p_s(IG::z2, IG::p0, IG::z8) == 0x65808102u);          // fadd z2.s,p0/m,z2.s,z8.s
    REQUIRE(IG::sve_fsub_p_s(IG::z0, IG::p0, IG::z8) == 0x65818100u);          // fsub z0.s,p0/m,z0.s,z8.s
    REQUIRE(IG::sve_fmul_p_s(IG::z2, IG::p0, IG::z5) == 0x658280a2u);          // fmul z2.s,p0/m,z2.s,z5.s
    REQUIRE(IG::sve_fmul_s(IG::z3, IG::z4, IG::z2)   == 0x65820883u);          // fmul z3.s,z4.s,z2.s
    REQUIRE(IG::sve_frsqrte_s(IG::z4, IG::z2)        == 0x658f3044u);          // frsqrte z4.s,z2.s
    REQUIRE(IG::sve_frsqrts_s(IG::z3, IG::z4, IG::z3) == 0x65831c83u);         // frsqrts z3.s,z4.s,z3.s
    REQUIRE(IG::sve_whilelo_s_x(IG::p1, IG::x0, IG::x22) == 0x25b61c01u);      // whilelo p1.s,x0,x22
    REQUIRE(IG::sve_sel_s(IG::z2, IG::p1, IG::z2, IG::z3) == 0x05a3c442u);     // sel z2.s,p1,z2.s,z3.s
}

// --- encoding diff ---------------------------------------------------------

// Read the instruction words of a linked function.  Code pages are readable
// on macOS arm64 (no execute-only mappings), and the .S kernels are a single
// straight-line symbol, so &fn is the first word and fn's RET is the last.
template <typename Fn>
static const uint32_t* linked_words(Fn* fn) {
    return reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(fn));
}

static void encoding_diff(const std::vector<uint32_t>& jit,
                          const uint32_t* golden) {
    REQUIRE(!jit.empty());
    // anchor: the hand-written kernel ends in exactly one RET; if the length
    // of the two sequences disagreed, this word would not be RET.
    REQUIRE(golden[jit.size() - 1] == 0xd65f03c0u);
    for (size_t i = 0; i < jit.size(); ++i) {
        INFO("word " << i << ": jit=" << IG::to_string_hex(jit[i])
                     << " golden=" << IG::to_string_hex(golden[i]));
        REQUIRE(jit[i] == golden[i]);
    }
}

TEST_CASE("Sprint4 encoding diff: JIT RMSNorm == assembled rms_norm_ssve_v6",
          "[norm][sprint4][encoding-diff]") {
    Norm gen;
    REQUIRE(gen.generate(Norm::ntype_t::rms) == Norm::error_t::success);
    REQUIRE(gen.words().size() == 143);        // instruction count of the .S
    encoding_diff(gen.words(), linked_words(&::rms_norm_ssve_v6));
}

TEST_CASE("Sprint4 encoding diff: JIT LayerNorm == assembled layer_norm_ssve_v6",
          "[norm][sprint4][encoding-diff]") {
    Norm gen;
    REQUIRE(gen.generate(Norm::ntype_t::layer) == Norm::error_t::success);
    REQUIRE(gen.words().size() == 194);
    encoding_diff(gen.words(), linked_words(&::layer_norm_ssve_v6));
}

// --- execution: JIT kernel vs C++ reference (same cases as the V6 tests) ----

TEST_CASE("Sprint4 JIT RMSNorm: executes correctly vs reference",
          "[norm][sprint4][jit]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    Norm gen;
    REQUIRE(gen.generate(Norm::ntype_t::rms) == Norm::error_t::success);
    Norm::rms_kernel_t k = gen.get_rms_kernel();
    REQUIRE(k != nullptr);
    check_variant(k,  64, 32,  64,  64, 1e-5f);   // exactly one full group
    check_variant(k, 100, 50, 100, 100, 1e-5f);   // group + tail blocks + partial
    check_variant(k,   8, 50,   8,   8, 1e-5f);   // tail-only path
    check_variant(k,  40, 37,  48,  40, 1e-5f);   // mismatched leading dims
    check_variant(k, 192, 37, 200, 224, 1e-5f);   // multiple groups, padded lds
    check_variant(k, 128,  1, 128, 128, 1e-5f);   // N=1
    check_variant_stress(k);                      // large-magnitude stress
}

TEST_CASE("Sprint4 JIT LayerNorm: executes correctly vs reference",
          "[norm][sprint4][jit]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    Norm gen;
    REQUIRE(gen.generate(Norm::ntype_t::layer) == Norm::error_t::success);
    Norm::layer_kernel_t k = gen.get_layer_kernel();
    REQUIRE(k != nullptr);
    check_ln_variant(k,  64, 32,  64,  64, 1e-5f);
    check_ln_variant(k, 100, 50, 100, 100, 1e-5f);
    check_ln_variant(k,   8, 50,   8,   8, 1e-5f);
    check_ln_variant(k,  40, 37,  48,  40, 1e-5f);
    check_ln_variant(k, 192, 37, 200, 224, 1e-5f);
    check_ln_variant(k, 128,  1, 128, 128, 1e-5f);
}

TEST_CASE("Sprint4 JIT kernels: all-constant row is finite (eps regression)",
          "[norm][sprint4][jit][eps]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    const int64_t M = 16, N = 16, ld = M;
    std::vector<float> a(ld * N, 3.0f);
    std::vector<float> gamma(N, 1.0f), beta(N, 0.5f);

    Norm gr;  REQUIRE(gr.generate(Norm::ntype_t::rms)   == Norm::error_t::success);
    Norm gl;  REQUIRE(gl.generate(Norm::ntype_t::layer) == Norm::error_t::success);

    std::vector<float> br(ld * N, 123.0f);
    gr.get_rms_kernel()(a.data(), br.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    for (int64_t i = 0; i < ld * N; ++i) REQUIRE(std::isfinite(br[i]));

    std::vector<float> bl(ld * N, 123.0f);
    gl.get_layer_kernel()(a.data(), bl.data(), gamma.data(), beta.data(), M, N, ld, ld, 1e-5f);
    for (int64_t i = 0; i < ld * N; ++i) {
        REQUIRE(std::isfinite(bl[i]));
        REQUIRE(bl[i] == Approx(0.5f).margin(1e-6f));
    }
}
