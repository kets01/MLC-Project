#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include "norm/norm.hpp"
#include "week3/utility.hpp"

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

// ---------------------------------------------------------------------------
// LayerNorm SSVE kernel tests (Sprint 2)
//
// Every test calls layer_norm_ssve and compares its output element-by-element
// against layer_norm_ref (the verified C++ reference from Sprint 1).
//
// All tests are guarded with cpu_supports_sme(): on CI (M1/M2) they are
// skipped gracefully; on M4 they run in full.
// ---------------------------------------------------------------------------

// Helper: run both kernels and compare.
// Accepts separate ld_a / ld_b so we can test padding.
static void check_ssve_vs_ref(int64_t m, int64_t n, int64_t ld_a, int64_t ld_b,
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
// This tests WHILELO + tail gather/scatter in isolation.
TEST_CASE("LayerNorm SSVE: N smaller than SVL (tail-only path)", "[norm][sprint2][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 4, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f + 0.1f * i; beta[i] = 0.05f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 7 + c * 3) % 11) - 5.0f;
    });

    check_ssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 2: N = SVL (N=16 on M4)
// Exactly one full vector, no tail.  Tests the clean single-iteration path.
TEST_CASE("LayerNorm SSVE: N equals SVL (no tail)", "[norm][sprint2][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 16, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 0.5f + 0.05f * i; beta[i] = -0.1f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 11 + c * 5) % 17) - 8.0f;
    });

    check_ssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 3: N = SVL + 3 (N=19 on M4)
// One full vector iteration + a tail of 3 elements.
// Tests that both the full-vector and tail paths run in the same row.
TEST_CASE("LayerNorm SSVE: N = SVL + 3 (full vector + small tail)", "[norm][sprint2][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 6, N = 19, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f; beta[i] = 0.0f; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 13 + c * 7) % 19) - 9.0f;
    });

    check_ssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 4: multiple rows, N = 2*SVL + 5 (N=37 on M4)
// Tests the outer row loop (x20/x19 row-base advancement) and
// multiple full-vector chunks + tail within each row.
TEST_CASE("LayerNorm SSVE: multiple rows, N = 2*SVL + 5", "[norm][sprint2][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 8, N = 37, ld = M;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 0.8f + 0.02f * i; beta[i] = 0.1f * i; }
    auto a = make_matrix(M, N, ld, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 5 + c * 11) % 23) - 11.0f;
    });

    check_ssve_vs_ref(M, N, ld, ld, 1e-5f, a, gamma, beta);
}

// Case 5: padded matrix — ld_a > M and ld_b > M
// Tests that the column stride (stride_col_a = ld_a*4) correctly skips the
// padding elements when ld_a != M.
TEST_CASE("LayerNorm SSVE: padded matrix (ld_a > M, ld_b > M)", "[norm][sprint2][ssve][layernorm]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const int64_t M = 5, N = 19, ld_a = 8, ld_b = 8;
    std::vector<float> gamma(N), beta(N);
    for (int64_t i = 0; i < N; ++i) { gamma[i] = 1.0f + 0.1f * i; beta[i] = -0.05f * i; }
    auto a = make_matrix(M, N, ld_a, [](int64_t r, int64_t c) {
        return static_cast<float>((r * 9 + c * 4) % 13) - 6.0f;
    });

    check_ssve_vs_ref(M, N, ld_a, ld_b, 1e-5f, a, gamma, beta);
}

// Case 6: large-magnitude stress input
// Values clustered around 1e4 with small variation (±5 around the DC offset).
// The reference accumulates in double; the SSVE kernel in FP32.
// With SHIFT=1e4, mean ≈ 1e4 and (x - mean) ≈ ±5: ~4 significant digits survive
// after cancellation.  The absolute error in the normalized output is bounded by
// roughly eps_float * SHIFT * inv_std ≈ 1.2e-7 * 1e4 * 0.45 ≈ 5e-4.
// We use an ABSOLUTE margin (not relative epsilon) because some normalized values
// are near zero, making a relative tolerance too tight for those elements.
TEST_CASE("LayerNorm SSVE: large-magnitude stress input (stability)", "[norm][sprint2][ssve][layernorm][stress]") {
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
