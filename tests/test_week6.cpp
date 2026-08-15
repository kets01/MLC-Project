#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "week6/gemm.hpp"
#include "week6/unary.hpp"
#include "week3/utility.hpp" // This header contains cpu_supports_sme()
              
 using namespace mini_jit;

TEST_CASE("Unary JIT Generator Functional and Performance Test", "[unary]") {
    if (!cpu_supports_sme()) SKIP("SME required (Unary kernels emit smstart/smstop)");
    mini_jit::Unary generator;
    std::vector<uint32_t> sizes = {64, 128, 512};

    // Print Table Header for the report
    std::cout << "\n" << std::left << std::setw(10) << "Op" 
              << std::setw(8) << "M" 
              << std::setw(8) << "N" 
              << std::setw(15) << "Status" 
              << "Performance (GiB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (uint32_t M : sizes) {
        for (uint32_t N : sizes) {
            
            // --- 1. IDENTITY TEST ---
            SECTION("Identity " + std::to_string(M) + "x" + std::to_string(N)) {
                generator.generate(M, N, 0, mini_jit::Unary::dtype_t::fp32, mini_jit::Unary::ptype_t::identity);
                auto kernel = generator.get_kernel();
                REQUIRE(kernel != nullptr);

                std::vector<float> src(M * N, 3.14f);
                std::vector<float> dst(M * N, 0.0f);
                
                // Call the JIT kernel
                kernel(src.data(), dst.data(), M, M);

                // Verify correctness
                bool correct = true;
                for (float f : dst) {
                    if (std::abs(f - 3.14f) > 1e-5) { 
                        correct = false; 
                        break; 
                    }
                }
                REQUIRE(correct);

                // Benchmark performance
                auto start = std::chrono::high_resolution_clock::now();
                int iters = 1000;
                for(int i = 0; i < iters; ++i) {
                    kernel(src.data(), dst.data(), M, M);
                }
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count() / iters;
                double bytes = (double)M * N * sizeof(float) * 2; // Read A + Write B
                double gibs = (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

                std::cout << std::left << std::setw(10) << "Identity" 
                          << std::setw(8) << M 
                          << std::setw(8) << N 
                          << std::setw(15) << "PASSED" 
                          << std::fixed << std::setprecision(2) << gibs << std::endl;
            }

            // --- 2. RELU TEST ---
            SECTION("ReLU " + std::to_string(M) + "x" + std::to_string(N)) {
                generator.generate(M, N, 0, mini_jit::Unary::dtype_t::fp32, mini_jit::Unary::ptype_t::relu);
                auto kernel = generator.get_kernel();
                REQUIRE(kernel != nullptr);

                std::vector<float> src(M * N);
                for(size_t i = 0; i < src.size(); ++i) {
                    src[i] = (i % 2 == 0) ? 5.0f : -5.0f;
                }
                std::vector<float> dst(M * N, 0.0f);

                kernel(src.data(), dst.data(), M, M);

                // Verify correctness
                bool correct = true;
                for(size_t i = 0; i < dst.size(); ++i) {
                    float expected = (src[i] > 0) ? src[i] : 0.0f;
                    if (std::abs(dst[i] - expected) > 1e-5) {
                        correct = false;
                        break;
                    }
                }
                REQUIRE(correct);

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                int iters = 1000;
                for(int i = 0; i < iters; ++i) {
                    kernel(src.data(), dst.data(), M, M);
                }
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count() / iters;
                double bytes = (double)M * N * sizeof(float) * 2; 
                double gibs = (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

                std::cout << std::left << std::setw(10) << "ReLU" 
                          << std::setw(8) << M 
                          << std::setw(8) << N 
                          << std::setw(15) << "PASSED" 
                          << std::fixed << std::setprecision(2) << gibs << std::endl;
            }
        }
    }
}

TEST_CASE("Gemm Functional Verification", "[gemm]") {
    if (!cpu_supports_sme()) SKIP("SME Required for testing");

    uint32_t dims[] = {64, 128, 512};

    SECTION("GEMM Primitive Verification (27 settings)") {
        Gemm gen;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    uint32_t M = dims[i];
                    uint32_t N = dims[j];
                    uint32_t K = dims[k];

                    REQUIRE(gen.generate(M, N, K, 0, 0, 0, Gemm::dtype_t::fp32) == Gemm::error_t::success);
                    auto kernel = gen.get_kernel();

                    std::vector<float> MA(M * K, 1.0f), MB(K * N, 1.0f), MC(M * N, 0.0f);
                    kernel(MA.data(), MB.data(), MC.data(), M, K, M);

                    // For matrices of 1.0f, every element in the output should be K
                    REQUIRE(MC[0] == (float)K);
                }
            }
        }
    }
}

// ===========================================================================
// GEMM layout correctness on non-uniform data.
//
// The 27-setting test above is blind to addressing bugs: any 16 contiguous
// floats of an all-ones matrix ARE the correct operand vector, and it checks
// only C[0].  These verify every element against a scalar reference with
// distinct data, across all (trans_a, trans_b, trans_c) combinations —
// including the ZA-staged transpose paths and the accumulate semantics the
// TEIR schedules rely on.
// ===========================================================================

#include "week6/Instgen.hpp"

TEST_CASE("Sprint5 InstGen: MOVA vector-to-tile inserts match golden words",
          "[sprint5][encoders]") {
    // golden words: clang -march=armv9-a+sme assembly of each instruction
    using IG = InstGen;
    REQUIRE(IG::sme_mova_vec_to_tile_h_s(0, 12, 0, IG::p0, IG::z0)  == 0xc0800000u); // mova za0h.s[w12,0],p0/m,z0.s
    REQUIRE(IG::sme_mova_vec_to_tile_h_s(0, 12, 3, IG::p0, IG::z5)  == 0xc08000a3u); // mova za0h.s[w12,3],p0/m,z5.s
    REQUIRE(IG::sme_mova_vec_to_tile_h_s(1, 13, 1, IG::p1, IG::z2)  == 0xc0802445u); // mova za1h.s[w13,1],p1/m,z2.s
    REQUIRE(IG::sme_mova_vec_to_tile_h_s(3, 15, 3, IG::p0, IG::z31) == 0xc08063efu); // mova za3h.s[w15,3],p0/m,z31.s
    REQUIRE(IG::sme_mova_vec_to_tile_v_s(0, 12, 0, IG::p0, IG::z0)  == 0xc0808000u); // mova za0v.s[w12,0],p0/m,z0.s
    REQUIRE(IG::sme_mova_vec_to_tile_v_s(2, 14, 2, IG::p2, IG::z9)  == 0xc080c92au); // mova za2v.s[w14,2],p2/m,z9.s
    REQUIRE(IG::sme_mova_vec_to_tile_v_s(3, 12, 3, IG::p0, IG::z1)  == 0xc080802fu); // mova za3v.s[w12,3],p0/m,z1.s
}

namespace {

// Element accessors per layout flag (ld in elements).
inline float& at(std::vector<float>& buf, uint32_t trans,
                 uint32_t row, uint32_t col, uint32_t ld) {
    return trans ? buf[row * ld + col] : buf[row + col * ld];
}

// Scalar reference: C += A * B, element-exact FMA order irrelevant here
// (double accumulator, tolerance covers reassociation).
void gemm_ref(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C,
              uint32_t M, uint32_t N, uint32_t K,
              uint32_t ta, uint32_t tb, uint32_t tc,
              uint32_t lda, uint32_t ldb, uint32_t ldc) {
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c) {
            double acc = at(C, tc, r, c, ldc);
            for (uint32_t k = 0; k < K; ++k)
                acc += (double)at(A, ta, r, k, lda) * (double)at(B, tb, k, c, ldb);
            at(C, tc, r, c, ldc) = (float)acc;
        }
}

void check_gemm_layout(uint32_t M, uint32_t N, uint32_t K,
                       uint32_t ta, uint32_t tb, uint32_t tc) {
    // lds per layout: leading dim = the non-contiguous direction's extent
    uint32_t lda = ta ? K : M;
    uint32_t ldb = tb ? N : K;
    uint32_t ldc = tc ? N : M;

    std::vector<float> A(M * K), B(K * N), C(M * N), expected;
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t k = 0; k < K; ++k)
            at(A, ta, r, k, lda) = 0.25f * float((r * 7 + k * 3) % 11) - 1.0f;
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t c = 0; c < N; ++c)
            at(B, tb, k, c, ldb) = 0.5f * float((k * 5 + c) % 13) - 2.0f;
    // Non-zero initial C so the accumulate semantics are verified too.
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c)
            at(C, tc, r, c, ldc) = 0.125f * float((r + 2 * c) % 5);

    expected = C;
    gemm_ref(A, B, expected, M, N, K, ta, tb, tc, lda, ldb, ldc);

    Gemm gen;
    REQUIRE(gen.generate(M, N, K, ta, tb, tc, Gemm::dtype_t::fp32) == Gemm::error_t::success);
    auto kernel = gen.get_kernel();
    REQUIRE(kernel != nullptr);
    kernel(A.data(), B.data(), C.data(), lda, ldb, ldc);

    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c) {
            INFO("layout (" << ta << "," << tb << "," << tc << ") element ("
                 << r << "," << c << ")");
            REQUIRE(std::fabs(at(C, tc, r, c, ldc) - at(expected, tc, r, c, ldc))
                    <= 1e-2f);
        }
}

} // namespace

TEST_CASE("Sprint5 GEMM: all layout combinations vs scalar reference",
          "[sprint5][gemm][layouts]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    // 48x48 exercises multi-tile M, the 16-wide n-tail (48 = 32+16), and a
    // multi-chunk K; every (trans_a, trans_b, trans_c) combination covers
    // both direct-load paths and both ZA-staged transpose paths. K=40 adds
    // the predicated remainder chunk (2 full chunks + 8) per combination.
    for (uint32_t ta = 0; ta <= 1; ++ta)
        for (uint32_t tb = 0; tb <= 1; ++tb)
            for (uint32_t tc = 0; tc <= 1; ++tc) {
                check_gemm_layout(48, 48, 32, ta, tb, tc);
                check_gemm_layout(48, 48, 40, ta, tb, tc);
            }
}

TEST_CASE("Sprint5 GEMM: arbitrary K (course spec) incl. remainder-only and K=1",
          "[sprint5][gemm][arbitrary-k]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_gemm_layout(32, 32, 8, 1, 1, 1);    // remainder-only, staged A
    check_gemm_layout(32, 32, 8, 0, 0, 0);    // remainder-only, staged B
    check_gemm_layout(32, 32, 17, 1, 0, 1);   // 1 chunk + rem 1, both staged
    check_gemm_layout(32, 32, 1, 0, 1, 0);    // K=1, direct paths
}

TEST_CASE("Sprint5 GEMM: .teir shapes (row-major, accumulating)",
          "[sprint5][gemm][teir-shapes]") {
    if (!cpu_supports_sme()) SKIP("SME required");
    check_gemm_layout(32, 64, 512, 1, 1, 1);   // matmul.teir per-invocation shape
    check_gemm_layout(96, 64, 256, 1, 1, 1);   // contraction.teir shape
}

TEST_CASE("Sprint5 GEMM: shape/dtype validation fails loudly",
          "[sprint5][gemm]") {
    Gemm gen;
    REQUIRE(gen.generate(15, 32, 32, 0, 0, 0, Gemm::dtype_t::fp32) == Gemm::error_t::err_shape);
    REQUIRE(gen.generate(32, 8, 32, 0, 0, 0, Gemm::dtype_t::fp32)  == Gemm::error_t::err_shape);
    REQUIRE(gen.generate(32, 32, 0, 0, 0, 0, Gemm::dtype_t::fp32)  == Gemm::error_t::err_shape);
    REQUIRE(gen.generate(32, 32, 32, 0, 0, 0, Gemm::dtype_t::fp64) == Gemm::error_t::err_unsupported_dtype);
}

TEST_CASE("Sprint5 Unary trans_b=1: transposing copy vs scalar reference",
          "[sprint5][unary][transb]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    // Unary.h has always specified trans_b as "1 if B is row-major" while A is
    // column-major — i.e. the memory transpose — and this implementation once
    // ignored the flag.  32x48 covers multi-tile both ways; padded lds
    // exercise the ld-awareness of that path.
    struct { uint32_t m, n, lda_pad, ldb_pad; } shapes[] = {
        {16, 16, 0, 0}, {32, 48, 0, 0}, {32, 48, 5, 3},
    };
    for (auto& s : shapes) {
        uint32_t lda = s.m + s.lda_pad;      // A col-major: ld >= m
        uint32_t ldb = s.n + s.ldb_pad;      // B row-major: ld >= n
        std::vector<float> A(lda * s.n), B(s.m * ldb, -7.0f);
        for (uint32_t i = 0; i < s.m; ++i)
            for (uint32_t j = 0; j < s.n; ++j)
                A[i + j * lda] = float(i * 100 + j) - 500.0f;

        mini_jit::Unary gen;
        REQUIRE(gen.generate(s.m, s.n, 1, Unary::dtype_t::fp32,
                             Unary::ptype_t::identity) == Unary::error_t::success);
        auto kernel = gen.get_kernel();
        REQUIRE(kernel != nullptr);
        kernel(A.data(), B.data(), lda, ldb);

        for (uint32_t i = 0; i < s.m; ++i)
            for (uint32_t j = 0; j < s.n; ++j) {
                INFO("shape " << s.m << "x" << s.n << " element (" << i << "," << j << ")");
                REQUIRE(B[i * ldb + j] == A[i + j * lda]);
            }
    }
}

TEST_CASE("Sprint5 Unary trans_b=1: transposing relu", "[sprint5][unary][transb]") {
    if (!cpu_supports_sme()) SKIP("SME required");

    const uint32_t m = 32, n = 16;
    std::vector<float> A(m * n), B(m * n, -7.0f);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j)
            A[i + j * m] = float(i) - float(j * 2);   // mix of signs

    mini_jit::Unary gen;
    REQUIRE(gen.generate(m, n, 1, Unary::dtype_t::fp32,
                         Unary::ptype_t::relu) == Unary::error_t::success);
    gen.get_kernel()(A.data(), B.data(), m, n);

    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j)
            REQUIRE(B[i * n + j] == std::max(A[i + j * m], 0.0f));
}