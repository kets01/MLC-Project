#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "norm/norm.hpp"

using namespace mini_jit::norm;

TEST_CASE("Norm placeholder: identity copy", "[norm][sprint0]") {
    const int64_t M = 16, N = 16, ld = 16;
    std::vector<float> a(M * N), b(M * N, 0.0f);
    for (int64_t i = 0; i < M * N; ++i) a[i] = static_cast<float>(i);

    norm_placeholder(a.data(), b.data(), M, N, ld);

    for (int64_t i = 0; i < M * N; ++i) {
        REQUIRE(b[i] == a[i]);
    }
}
