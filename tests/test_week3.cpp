#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include "week3/unary.hpp"
#include "week3/gemm_sme.hpp"


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
TEST_CASE("SME GEMM 512x512x512", "[week3][gemm]") {
    const int64_t DIM = 512;

    std::vector<float> A(DIM * DIM, 1.0f);
    std::vector<float> B(DIM * DIM, 1.0f);
    std::vector<float> C(DIM * DIM, 0.0f);

    // A: Col-major, B: Row-major, C: Col-major
    gemm_512_512_512(A.data(), B.data(), C.data(), DIM, DIM, DIM);

    // If A and B are all 1s, every element of C should be 512
    for (int i = 0; i < 100; ++i) {
        REQUIRE(C[i] == 512.0f);
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
