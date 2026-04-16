#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "base_math.hpp" // Use your header for the prototypes

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