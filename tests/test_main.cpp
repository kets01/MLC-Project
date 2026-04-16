#include <catch2/catch_test_macros.hpp>
#include "base_math.hpp"

TEST_CASE("Infrastructure Check", "[stub]") {
    SECTION("Basic Linkage") {
        uint32_t a = 0;
        // Verify we can call the function without crashing
        // (Since it's a stub that returns 0, this should pass)
        REQUIRE(inner_product_asm(&a, &a, 0) == 0);
    }
}

TEST_CASE("Math logic stubs", "[math]") {
    uint32_t vec_a[] = {1, 2};
    uint32_t vec_b[] = {3, 4};
    
    SECTION("Inner Product Stub return value") {
        // Our stub currently just returns 0
        REQUIRE(inner_product_asm(vec_a, vec_b, 2) == 0);
    }
}
