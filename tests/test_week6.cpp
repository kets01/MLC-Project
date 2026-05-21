#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "week3/utility.hpp"
//#include "week6/Unary.h"
#include "week6/gemm.hpp"

using namespace mini_jit;

TEST_CASE("Week 6 Functional Verification", "[jit_week6]") {
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
                    REQUIRE(MC[M * N - 1] == (float)K);
                }
            }
        }
    }
}