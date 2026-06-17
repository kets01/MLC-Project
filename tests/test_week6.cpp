#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "week6/gemm.hpp"
#include "week6/unary.hpp"
#include "week3/utility.hpp" // This header contains cpu_supports_sme()
              
 using namespace mini_jit;

TEST_CASE("Unary JIT Generator Functional and Performance Test", "[unary]") {
    if (!cpu_supports_sme()) SKIP("SME required (Unary kernels emit smstart/smstop)");
    mini_jit::Unary generator;
    std::vector<uint32_t> sizes = {64, 128, 512};

    // Print Table Header for the report
    std::cout << "\n" << std::left << std::setw(10) << "Op" 
              << std::setw(8) << "M" 
              << std::setw(8) << "N" 
              << std::setw(15) << "Status" 
              << "Performance (GiB/s)" << std::endl;
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

TEST_CASE("Gemm Functional Verification", "[gemm]") {
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
                }
            }
        }
    }
}