#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "week7/TeirRuntime.h"
#include "week3/utility.hpp"

using namespace mini_jit::teir;

TEST_CASE("TEIR Runtime Functional Verification", "[teir]") {
    if (!cpu_supports_sme()) SKIP("SME required (identity and GEMM kernels use smstart/smstop)");
    TeirRuntime runtime;

    SECTION("Task 1: Transposition Verification") {
        uint64_t total = 96 * 128 * 48 * 32; 
        std::vector<float> A(total, 5.0f), B(total, 0.0f);
        
        auto root = runtime.build_transposition_tree();
        runtime.execute(root.get(), A.data(), B.data(), nullptr);

        // Check first and last element. Should be 5.0f if pointers moved correctly.
        REQUIRE(B[0] == 5.0f);
        REQUIRE(B[total - 1] == 5.0f); 
    }

    SECTION("Task 2: Matmul Verification") {
        size_t size = 512 * 512; 
        std::vector<float> A(size, 1.0f), B(size, 1.0f), C(size, 0.0f);
        auto root = runtime.build_matmul_tree();
        
        // Temporarily reduce range for quick unit test
        dynamic_cast<Iteration*>(root.get())->axis->range = 2; 
        
        runtime.execute(root.get(), A.data(), B.data(), C.data());
        REQUIRE(C[0] > 0);
    }

    SECTION("Task 3: Contraction Verification") {
        // Large buffer to ensure strides don't Segfault
        size_t size = 256 * 256 * 16;
        std::vector<float> A(size, 1.0f), B(size, 1.0f), C(size, 0.0f);
        auto root = runtime.build_contraction_tree();
        
        // Confirm no crash occurs during high-dimensional traversal
        REQUIRE_NOTHROW(runtime.execute(root.get(), A.data(), B.data(), C.data()));
    }
}