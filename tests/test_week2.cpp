#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "week2/fmadd.hpp"

TEST_CASE("FMADD Benchmark", "[fmadd]") {
    SECTION("Benchmark FMADD") {
        // This test will run the benchmark for FMA and ensure it completes without errors.
        // Note: We won't check the performance here, just that it runs.
        fmadd_asm(1000000000); // Run 1 billion iterations
        REQUIRE(true); // If we reach this point, the benchmark ran successfully.
    }
}