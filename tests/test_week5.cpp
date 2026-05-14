#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "week5/jit_engine.hpp"
#include "week5/jit_kernel.hpp"
#include "week3/utility.hpp"

TEST_CASE("JIT Kernel Functional Verification", "[jit]") {
    if (!cpu_supports_sme()) SKIP("SME Required");

    SECTION("Identity Kernel") {
        auto jit_id = JitEngine::generate<void(*)(const float*, float*, int64_t, int64_t, int32_t)>(get_identity_jit_opcodes());
        std::vector<float> A(256, 3.14f), B(256, 0.0f);
        jit_id(A.data(), B.data(), 16, 16, 0);
        for(float f : B) REQUIRE(f == 3.14f);
    }

    SECTION("Zero Kernel") {
        auto jit_zero = JitEngine::generate<void(*)(float*, int64_t)>(get_zero_jit_opcodes());
        std::vector<float> data(256, 1.0f);
        jit_zero(data.data(), 16);
        for(float f : data) REQUIRE(f == 0.0f);
    }

    SECTION("ReLU Kernel") {
        auto jit_relu = JitEngine::generate<void(*)(const float*, float*, int64_t, int64_t, int32_t)>(get_relu_jit_opcodes());
        std::vector<float> A(256), B(256, 1.0f);
        for(int i=0; i<256; i++) A[i] = (float)i - 128.0f;
        jit_relu(A.data(), B.data(), 16, 16, 0);
        REQUIRE(B[0] == 0.0f);   // -128 clipped
        REQUIRE(B[128] == 0.0f); // 0 preserved
        REQUIRE(B[129] == 1.0f); // 1 preserved
    }

    SECTION("GEMM Kernel") {
        auto jit_gemm = JitEngine::generate<void(*)(const float*, const float*, float*, int64_t, int64_t, int64_t)>(get_gemm_jit_opcodes());
        const int64_t DIM = 512;
        std::vector<float> MA(DIM*DIM, 1.0f), MB(DIM*DIM, 1.0f), MC(DIM*DIM, 0.0f);
        jit_gemm(MA.data(), MB.data(), MC.data(), DIM, DIM, DIM);
        REQUIRE(MC[0] == 512.0f);
        REQUIRE(MC[DIM*DIM-1] == 512.0f);
    }
}