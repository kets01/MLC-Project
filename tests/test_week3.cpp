#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include "week3/unary.hpp"
#include "week3/gemm_sme.hpp"
#include "utility.hpp"


// ------------------------------------------------------------
// Pretty printer for 16x16 matrices (used by unary tests)
// ------------------------------------------------------------
void print_16x16(const std::string& label, const std::vector<float>& data, int64_t ld) {
    std::cout << "\n--- " << label << " ---\n";
    for (int i = 0; i < 16; i++) {
        std::cout << "Row " << std::setw(2) << i << ": ";
        for (int j = 0; j < 16; j++) {
            // Column-major access: i + j*ld
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(6) << data[i + j * ld] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}



// ============================================================================
//                           TEST SUITE 1 — GEMM
// ============================================================================
// Small epsilon for float comparison 
const float EPS = 0.5f;

TEST_CASE("SME Kernel Verification against C++ Reference", "[gemm]") {
    if (!cpu_supports_sme()) {
        SKIP("CPU does not support SME instructions (Likely running on M1/M2 runner).");
    }

    const int64_t M = 512;
    const int64_t N = 512;
    const int64_t K = 512;

    // Initialize with test data
    std::vector<float> A(M * K, 0.1f);
    std::vector<float> B(K * N, 0.2f);
    
    // Fill with some gradients to catch indexing errors
    for(size_t i=0; i<A.size(); ++i) A[i] = (float)(i % 100) * 0.01f;
    for(size_t i=0; i<B.size(); ++i) B[i] = (float)(i % 100) * 0.01f;

    SECTION("Level 1: 32x32 result, K=1") {
        std::vector<float> C_asm(M * N, 0.0f);
        std::vector<float> C_ref(M * N, 0.0f);

        gemm_32_32_1(A.data(), B.data(), C_asm.data(), M, K, M);
        ref_gemm_32_32_1(A.data(), B.data(), C_ref.data(), M, K, M);

        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 32; ++i) {
                float diff = std::abs(C_asm[j * M + i] - C_ref[j * M + i]);
                REQUIRE(diff < EPS);
            }
        }
    }

    SECTION("Level 2: 32x32 result, K=512") {
        std::vector<float> C_asm(M * N, 0.0f);
        std::vector<float> C_ref(M * N, 0.0f);

        gemm_32_32_512(A.data(), B.data(), C_asm.data(), M, K, M);
        ref_gemm_32_32_512(A.data(), B.data(), C_ref.data(), M, K, M);

        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 32; ++i) {
                float diff = std::abs(C_asm[j * M + i] - C_ref[j * M + i]);
                REQUIRE(diff < EPS);
            }
        }
    }

    SECTION("Level 3: 512x32 result, K=512") {
        std::vector<float> C_asm(M * N, 0.0f);
        std::vector<float> C_ref(M * N, 0.0f);

        gemm_512_32_512(A.data(), B.data(), C_asm.data(), M, K, M);
        ref_gemm_512_32_512(A.data(), B.data(), C_ref.data(), M, K, M);

        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 512; ++i) {
                float diff = std::abs(C_asm[j * M + i] - C_ref[j * M + i]);
                REQUIRE(diff < EPS);
            }
        }
    }

    SECTION("Level 4: 512x512 result, K=512") {
        std::vector<float> C_asm(M * N, 0.0f);
        std::vector<float> C_ref(M * N, 0.0f);

        gemm_512_512_512(A.data(), B.data(), C_asm.data(), M, K, M);
        ref_gemm_512_512_512(A.data(), B.data(), C_ref.data(), M, K, M);

        // Spot check the whole matrix (every 13th element to save time)
        for (int j = 0; j < 512; j += 13) {
            for (int i = 0; i < 512; i += 13) {
                float diff = std::abs(C_asm[j * M + i] - C_ref[j * M + i]);
                REQUIRE(diff < EPS);
            }
        }
    }
}



// ============================================================================
//                     TEST SUITE 2 — Unary Ops (ReLU, Zero, Identity)
// ============================================================================
TEST_CASE("SVE Unary Kernels Verification", "[week3][unary]") {
    const int64_t ld = 16;
    std::vector<float> A(256);
    std::vector<float> B_asm(256, 0.0f);
    std::vector<float> B_cpp(256, 0.0f);

    // Initialize A with a pattern
    for (int i = 0; i < 256; i++) A[i] = (float)i - 128.0f;


    SECTION("ReLU Test") {
        relu_16_16_asm(A.data(), B_asm.data(), ld, ld, 0);
        relu_16_16_cpp(A.data(), B_cpp.data(), ld, ld, 0);

        print_16x16("Input Matrix A", A, ld);
        print_16x16("Output Matrix B (ReLU ASM)", B_asm, ld);

        REQUIRE(B_asm == B_cpp);
    }


    SECTION("Zero Test") {
        std::vector<float> Z_asm(256, 1.0f);
        std::vector<float> Z_ref(256, 0.0f);

        zero_16_16_asm(Z_asm.data(), ld);

        print_16x16("Zero Kernel Result", Z_asm, ld);

        REQUIRE(Z_asm == Z_ref);
    }


    SECTION("Identity Transpose Test") {
        for (int i = 0; i < 256; i++) A[i] = (float)i;

        identity_16_16_asm(A.data(), B_asm.data(), ld, ld, 1);
        identity_16_16_cpp(A.data(), B_cpp.data(), ld, ld, 1);

        print_16x16("Identity Input A", A, ld);
        print_16x16("Identity Output B (ASM Transpose)", B_asm, ld);

        REQUIRE(B_asm == B_cpp);
    }
}
