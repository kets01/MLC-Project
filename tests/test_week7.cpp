#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>
#include "week7/TeirRuntime.h"
#include "week3/utility.hpp"
#include "norm/norm.hpp"

using namespace mini_jit::teir;
using Catch::Approx;

// ===========================================================================
// Sprint 5 — the TEIR runtime executing the REAL data/*.teir files.
//
// Every tree below is parsed from its file (TeirParser), and every result is
// verified element-wise against a scalar reference that replays the file's
// own axis strides on non-uniform data. The previous tests used constant
// data (all-ones / all-fives) and spot-checked single elements — checks that
// cannot distinguish a correct kernel from an addressing bug, which is
// exactly how the week6 GEMM's broken B-operand path stayed invisible.
//
// Helpers walk the parsed tree to shrink outer loop ranges where the file's
// full problem (8192^3 matmul) is too large for a unit test; the reference
// then replays the SAME shrunken iteration space.
// ===========================================================================

namespace {

// Count elementwise mismatches instead of one REQUIRE per element (a
// REQUIRE per element is prohibitively slow at tens of millions).
int64_t count_mismatch(const std::vector<float>& got,
                       const std::vector<float>& expected, float tol,
                       int64_t* first = nullptr) {
    int64_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - expected[i]) > tol) {
            if (bad == 0 && first) *first = (int64_t)i;
            ++bad;
        }
    return bad;
}

Iteration* as_iter(Node* n) { return dynamic_cast<Iteration*>(n); }

// Follow the tree to the iteration node for a named axis.
Iteration* find_iter(Node* n, const std::string& axis_name) {
    if (auto* it = as_iter(n)) {
        if (it->axis->name == axis_name) return it;
        return find_iter(it->body.get(), axis_name);
    }
    if (auto* seq = dynamic_cast<Sequence*>(n)) {
        for (auto& c : seq->children)
            if (auto* found = find_iter(c.get(), axis_name)) return found;
    }
    return nullptr;
}

} // namespace

TEST_CASE("TEIR: transposition.teir (faithful strides, full verification)", "[teir][sprint5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    TeirRuntime runtime;

    // in: [a][b][c][d] (96,128,48,32) row-major; out strides from the file.
    const uint64_t A = 96, B = 128, C = 48, D = 32;
    const uint64_t total = A * B * C * D;
    const uint64_t in_a = 196608, in_b = 1536, in_c = 32, in_d = 1;      // elements
    const uint64_t out_a = 48, out_b = 4416, out_c = 1, out_d = 565248;  // elements

    std::vector<float> in(total), out(total, -3.0f), expected(total, -3.0f);
    for (uint64_t i = 0; i < total; ++i) in[i] = float(i % 977) * 0.5f - 100.0f;

    // Faithful replay in the runtime's traversal order (a, then b, both
    // sequential in the file). Note: the file's out strides make writes
    // overlap for a >= 92 (out_b = 4416 < 96*48); sequential replay in the
    // same order reproduces the same last-writer-wins result.
    for (uint64_t a = 0; a < A; ++a)
        for (uint64_t b = 0; b < B; ++b)
            for (uint64_t c = 0; c < C; ++c)
                for (uint64_t d = 0; d < D; ++d)
                    expected[a * out_a + b * out_b + c * out_c + d * out_d] =
                        in[a * in_a + b * in_b + c * in_c + d * in_d];

    auto root = runtime.load_teir("transposition.teir");
    runtime.execute(root.get(), in.data(), out.data(), nullptr);

    int64_t first = -1;
    int64_t bad = count_mismatch(out, expected, 0.0f, &first);
    INFO("first mismatch at flat index " << first);
    REQUIRE(bad == 0);
}

TEST_CASE("TEIR: matmul.teir (shrunk ranges, guarded zero + accumulation)", "[teir][sprint5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    TeirRuntime runtime;

    auto root = runtime.load_teir("matmul.teir");
    // Shrink: k0 16->1, m0 256->1, n0 128->4. Untouched regions of C keep
    // their sentinel, which the reference replicates — this also verifies
    // the schedule's Sequence order (zero before the k0=0 gemms) and the
    // guard, since the zero tile covers only block (m0=0, n0=0).
    Iteration* it_k0 = find_iter(root.get(), "k0");  REQUIRE(it_k0);
    Iteration* it_m0 = find_iter(root.get(), "m0");  REQUIRE(it_m0);
    Iteration* it_n0 = find_iter(root.get(), "n0");  REQUIRE(it_n0);
    it_k0->axis->range = 1;
    it_m0->axis->range = 1;
    it_n0->axis->range = 4;

    const uint64_t M1 = 32, N1 = 64, K1 = 512, N0 = 4;
    // element strides from the file
    const uint64_t a_m1 = 512, a_k1 = 1;
    const uint64_t b_n0 = 32768, b_k1 = 64, b_n1 = 1;
    const uint64_t c_n0 = 2048, c_m1 = 64, c_n1 = 1;

    std::vector<float> a(M1 * K1), b(N0 * b_n0), c(N0 * c_n0, 5.0f), expected;
    for (size_t i = 0; i < a.size(); ++i) a[i] = float(i % 23) * 0.25f - 2.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = float(i % 19) * 0.5f - 4.0f;

    expected = c;
    // zero fires once (k0==0) on block (m0=0, n0=0) BEFORE the gemms
    for (uint64_t m1 = 0; m1 < M1; ++m1)
        for (uint64_t n1 = 0; n1 < N1; ++n1)
            expected[m1 * c_m1 + n1 * c_n1] = 0.0f;
    for (uint64_t n0 = 0; n0 < N0; ++n0)
        for (uint64_t m1 = 0; m1 < M1; ++m1)
            for (uint64_t n1 = 0; n1 < N1; ++n1) {
                double acc = expected[n0 * c_n0 + m1 * c_m1 + n1 * c_n1];
                for (uint64_t k1 = 0; k1 < K1; ++k1)
                    acc += (double)a[m1 * a_m1 + k1 * a_k1] *
                           (double)b[n0 * b_n0 + k1 * b_k1 + n1 * b_n1];
                expected[n0 * c_n0 + m1 * c_m1 + n1 * c_n1] = (float)acc;
            }

    runtime.execute(root.get(), a.data(), b.data(), c.data());

    int64_t first = -1;
    int64_t bad = count_mismatch(c, expected, 1e-2f, &first);
    INFO("first mismatch at flat index " << first);
    REQUIRE(bad == 0);
}

TEST_CASE("TEIR: contraction.teir (shrunk ranges, strided zero + t-accumulation)", "[teir][sprint5]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    TeirRuntime runtime;

    auto root = runtime.load_teir("contraction.teir");
    // Shrink: p 128->1, r 96->2, t 32->2 (t sequential: exercises the
    // guarded zero at t=0 and accumulation across the second t step).
    Iteration* it_p = find_iter(root.get(), "p");  REQUIRE(it_p);
    Iteration* it_r = find_iter(root.get(), "r");  REQUIRE(it_r);
    Iteration* it_t = find_iter(root.get(), "t");  REQUIRE(it_t);
    it_p->axis->range = 1;
    it_r->axis->range = 2;
    it_t->axis->range = 2;

    const uint64_t Q = 96, S = 64, U = 256, R = 2, T = 2;
    // element strides from the file
    const uint64_t i0_q = 8192, i0_t = 256, i0_u = 1;
    const uint64_t i1_t = 1572864, i1_r = 16384, i1_u = 64, i1_s = 1;
    const uint64_t o_q = 6144, o_r = 64, o_s = 1;

    std::vector<float> in0(Q * i0_q), in1(T * i1_t), out(Q * o_q, 5.0f), expected;
    for (size_t i = 0; i < in0.size(); ++i) in0[i] = float(i % 17) * 0.25f - 2.0f;
    for (size_t i = 0; i < in1.size(); ++i) in1[i] = float(i % 29) * 0.125f - 1.0f;

    expected = out;
    for (uint64_t r = 0; r < R; ++r) {
        // t=0: guarded zero wipes the q x s tile at this (p=0, r) position
        for (uint64_t q = 0; q < Q; ++q)
            for (uint64_t s = 0; s < S; ++s)
                expected[q * o_q + r * o_r + s * o_s] = 0.0f;
        for (uint64_t t = 0; t < T; ++t)
            for (uint64_t q = 0; q < Q; ++q)
                for (uint64_t s = 0; s < S; ++s) {
                    double acc = expected[q * o_q + r * o_r + s * o_s];
                    for (uint64_t u = 0; u < U; ++u)
                        acc += (double)in0[q * i0_q + t * i0_t + u * i0_u] *
                               (double)in1[t * i1_t + r * i1_r + u * i1_u + s * i1_s];
                    expected[q * o_q + r * o_r + s * o_s] = (float)acc;
                }
    }

    runtime.execute(root.get(), in0.data(), in1.data(), out.data());

    int64_t first = -1;
    int64_t bad = count_mismatch(out, expected, 1e-2f, &first);
    INFO("first mismatch at flat index " << first);
    REQUIRE(bad == 0);
}

TEST_CASE("TEIR: rmsnorm.teir vs C++ reference", "[teir][sprint5][norm]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    TeirRuntime runtime;

    const int64_t M = 512, N = 512, ld = 512;
    std::vector<float> a(ld * N), b(ld * N, -1.0f), gamma(N);
    for (int64_t i = 0; i < ld * N; ++i) a[i] = 0.01f * float(i % 97);
    for (int64_t j = 0; j < N; ++j) gamma[j] = 1.0f + 0.001f * float(j % 13);

    auto root = runtime.load_teir("rmsnorm.teir");
    runtime.execute(root.get(), a.data(), b.data(), gamma.data());

    std::vector<float> expected(ld * N, 0.0f);
    mini_jit::norm::rms_norm_ref(a.data(), expected.data(), gamma.data(),
                                 M, N, ld, ld, 1e-5f);

    int64_t first = -1;
    int64_t bad = count_mismatch(b, expected, 1e-4f, &first);
    INFO("first mismatch at flat index " << first);
    REQUIRE(bad == 0);
}

TEST_CASE("TEIR: layernorm.teir vs C++ reference", "[teir][sprint5][norm]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    TeirRuntime runtime;

    const int64_t M = 512, N = 512, ld = 512;
    std::vector<float> a(ld * N), b(ld * N, -1.0f), gamma(N), beta(N);
    for (int64_t i = 0; i < ld * N; ++i) a[i] = 0.01f * float(i % 97);
    for (int64_t j = 0; j < N; ++j) {
        gamma[j] = 1.0f + 0.001f * float(j % 13);
        beta[j]  = 0.01f * float(j % 7);
    }

    auto root = runtime.load_teir("layernorm.teir");
    runtime.execute(root.get(), a.data(), b.data(), gamma.data(), beta.data());

    std::vector<float> expected(ld * N, 0.0f);
    mini_jit::norm::layer_norm_ref(a.data(), expected.data(), gamma.data(),
                                   beta.data(), M, N, ld, ld, 1e-5f);

    int64_t first = -1;
    int64_t bad = count_mismatch(b, expected, 1e-4f, &first);
    INFO("first mismatch at flat index " << first);
    REQUIRE(bad == 0);
}
