#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "base_math.hpp" 

TEST_CASE("Inner Product Calculation", "[inner_product]") {
    
    SECTION("Standard 4-element vectors") {
        uint32_t a[] = {1, 2, 3, 4};
        uint32_t b[] = {5, 6, 7, 8};
        uint32_t size = 4;
        int64_t expected = 70; // (1*5) + (2*6) + (3*7) + (4*8) = 70
        
        int64_t result = inner_product_asm(a, b, size);
        
        // Catch2 uses REQUIRE with standard C++ operators (==)
        REQUIRE(result == expected);
    }

    SECTION("Empty arrays return zero") {
        uint32_t a[] = {0};
        uint32_t b[] = {0};
        int64_t expected = 0;
        
        REQUIRE(inner_product_asm(a, b, 0) == expected);

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
