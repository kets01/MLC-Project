#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "base_math.hpp"

TEST_CASE("Infrastructure Check", "[stub]") {
    SECTION("Basic Linkage") {
        uint32_t a = 0;
        // Verify we can call the function without crashing
        // (Since it's a stub that returns 0, this should pass)
        REQUIRE(inner_product_asm(&a, &a, 0) == 0);
    }
}

TEST_CASE("Outer Product Functional Test", "[outer]") {
    uint32_t a[] = {1, 2};
    uint32_t b[] = {3, 4};
    uint64_t c[4] = {0}; // 2x2 matrix
    
    outer_product_asm(a, b, 2, c);
    
    // Matrix should be:
    // [ 1*3  1*4 ]  => [ 3  4 ]
    // [ 2*3  2*4 ]     [ 6  8 ]
    REQUIRE(c[0] == 3);
    REQUIRE(c[1] == 4);
    REQUIRE(c[2] == 6);
    REQUIRE(c[3] == 8);
}
