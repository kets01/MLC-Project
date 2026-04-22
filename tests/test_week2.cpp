#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>
#include "week2/perm.hpp"
#include "week2/fmadd.hpp"


// Reference code to verify your Assembly
void perm_cpp_abc_cba(int64_t size_c, float const * abc, float * cba) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < size_c; ++k) {
                int64_t in_idx = i * (4 * size_c) + j * size_c + k;
                int64_t out_idx = k * 32 + j * 8 + i;
                cba[out_idx] = abc[in_idx];
            }
        }
    }
}

TEST_CASE("FMADD Benchmark", "[fmadd]") {
    SECTION("Benchmark FMADD") {
        fmadd_asm(1000000000); // Run 1 billion iterations
        REQUIRE(true); // If we reach this point, the benchmark ran successfully.
    }
}

TEST_CASE("Verification: TRN logic", "[perm]") {
    const int64_t size_c = 4; // Test a perfect 4x4 block
    std::vector<float> abc(8 * 4 * size_c);
    std::vector<float> ref(8 * 4 * size_c, 0.0f);
    std::vector<float> neon(8 * 4 * size_c, 0.0f);

    for (size_t i = 0; i < abc.size(); ++i) abc[i] = (float)i;

    perm_cpp_abc_cba(size_c, abc.data(), ref.data());
    perm_neon_abc_cba(size_c, abc.data(), neon.data());

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(neon[i] == ref[i]);
    }
}