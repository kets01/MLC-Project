#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp> // Added for vector matching
#include <iostream>
#include <vector>
#include <iomanip>
#include "week3/unary.hpp"

// Improved printer using only std::cout to avoid buffer mixing
void print_16x16(const std::string& label, const std::vector<float>& data, int64_t ld) {
    std::cout << "\n--- " << label << " ---\n";
    for (int i = 0; i < 16; i++) {
        std::cout << "Row " << std::setw(2) << i << ": ";
        for (int j = 0; j < 16; j++) {
            // Column-major access: i + j*ld
            std::cout << std::fixed << std::setprecision(1) << std::setw(6) << data[i + j*ld] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

TEST_CASE("SVE Unary Kernels Verification", "[week3]") {
    const int64_t ld = 16;
    std::vector<float> A(256);
    std::vector<float> B_asm(256, 0.0f);
    std::vector<float> B_cpp(256, 0.0f);

    // Initialize A with a pattern
    for (int i = 0; i < 256; i++) A[i] = (float)i - 128.0f;

    SECTION("ReLU Test") {
        relu_16_16_asm(A.data(), B_asm.data(), ld, ld, 0);
        relu_16_16_cpp(A.data(), B_cpp.data(), ld, ld, 0);

        // Printing BEFORE the requirement
        print_16x16("Input Matrix A", A, ld);
        print_16x16("Output Matrix B (ReLU ASM)", B_asm, ld);

        // Comparing whole vectors prevents 256 lines of success messages
        REQUIRE(B_asm == B_cpp);
    }

    SECTION("Zero Test") {
        std::vector<float> Z_asm(256, 1.0f);
        std::vector<float> Z_ref(256, 0.0f); // Comparison reference

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