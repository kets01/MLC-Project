#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "week3/gemm_sme.hpp"

TEST_CASE("SME GEMM 512x512x512", "[week3]") {
    const int64_t DIM = 512;
    std::vector<float> A(DIM * DIM, 1.0f);
    std::vector<float> B(DIM * DIM, 1.0f);
    std::vector<float> C(DIM * DIM, 0.0f);

    // Correct Leading Dimensions
    // A: Col-major, B: Row-major, C: Col-major
    gemm_512_512_512(A.data(), B.data(), C.data(), DIM, DIM, DIM);

    // If A and B are all 1s, every element of C should be 512
    for(int i = 0; i < 100; ++i) { // Check first 100 elements
        REQUIRE(C[i] == 512.0f);
    }
}