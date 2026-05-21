#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "week6/Unary.h"
#include "week3/utility.hpp" // This header contains cpu_supports_sme()

TEST_CASE("Unary JIT Generator Functional and Performance Test", "[unary]") {
    // Use the correct hardware check function from your utility library
    if (!cpu_supports_sme()) {
        SKIP("SME/SVE Hardware not supported on this host");
    }

    mini_jit::Unary generator;
    std::vector<uint32_t> sizes = {64, 128, 512};

    // Print Table Header for the report
    std::cout << "\n" << std::left << std::setw(10) << "Op" 
              << std::setw(8) << "M" 
              << std::setw(8) << "N" 
              << std::setw(15) << "Status" 
              << "Throughput (GiB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (uint32_t M : sizes) {
        for (uint32_t N : sizes) {
            
            // --- 1. IDENTITY TEST ---
            SECTION("Identity " + std::to_string(M) + "x" + std::to_string(N)) {
                generator.generate(M, N, 0, mini_jit::Unary::dtype_t::fp32, mini_jit::Unary::ptype_t::identity);
                auto kernel = generator.get_kernel();
                REQUIRE(kernel != nullptr);

                std::vector<float> src(M * N, 3.14f);
                std::vector<float> dst(M * N, 0.0f);
                
                // Call the JIT kernel
                kernel(src.data(), dst.data(), M, M);

                // Verify correctness
                bool correct = true;
                for (float f : dst) {
                    if (std::abs(f - 3.14f) > 1e-5) { 
                        correct = false; 
                        break; 
                    }
                }
                REQUIRE(correct);

                // Benchmark performance
                auto start = std::chrono::high_resolution_clock::now();
                int iters = 1000;
                for(int i = 0; i < iters; ++i) {
                    kernel(src.data(), dst.data(), M, M);
                }
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count() / iters;
                double bytes = (double)M * N * sizeof(float) * 2; // Read A + Write B
                double gibs = (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

                std::cout << std::left << std::setw(10) << "Identity" 
                          << std::setw(8) << M 
                          << std::setw(8) << N 
                          << std::setw(15) << "PASSED" 
                          << std::fixed << std::setprecision(2) << gibs << std::endl;
            }

            // --- 2. RELU TEST ---
            SECTION("ReLU " + std::to_string(M) + "x" + std::to_string(N)) {
                generator.generate(M, N, 0, mini_jit::Unary::dtype_t::fp32, mini_jit::Unary::ptype_t::relu);
                auto kernel = generator.get_kernel();
                REQUIRE(kernel != nullptr);

                std::vector<float> src(M * N);
                for(size_t i = 0; i < src.size(); ++i) {
                    src[i] = (i % 2 == 0) ? 5.0f : -5.0f;
                }
                std::vector<float> dst(M * N, 0.0f);

                kernel(src.data(), dst.data(), M, M);

                // Verify correctness
                bool correct = true;
                for(size_t i = 0; i < dst.size(); ++i) {
                    float expected = (src[i] > 0) ? src[i] : 0.0f;
                    if (std::abs(dst[i] - expected) > 1e-5) {
                        correct = false;
                        break;
                    }
                }
                REQUIRE(correct);

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                int iters = 1000;
                for(int i = 0; i < iters; ++i) {
                    kernel(src.data(), dst.data(), M, M);
                }
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count() / iters;
                double bytes = (double)M * N * sizeof(float) * 2; 
                double gibs = (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

                std::cout << std::left << std::setw(10) << "ReLU" 
                          << std::setw(8) << M 
                          << std::setw(8) << N 
                          << std::setw(15) << "PASSED" 
                          << std::fixed << std::setprecision(2) << gibs << std::endl;
            }
        }
    }
}