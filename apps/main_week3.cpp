#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week3/unary.hpp"
#include "week3/gemm_sme.hpp"



// ============================================================================
//                         Benchmark: Unary Ops (16x16)
// ============================================================================
//
// Calculations for 16x16 FP32 Matrix
// Total elements = 256
// Bytes per matrix = 256 * 4 = 1024 bytes (1 KiB)
// ============================================================================
void benchmark_unary() {
    const int64_t ld = 16;
    const int64_t iterations = 10'000'000; // 10 million runs

    std::vector<float> A(256, 1.0f);
    std::vector<float> B(256, 0.0f);

    std::cout << "\n============================================\n";
    std::cout << "      Benchmarking Unary Kernels\n";
    std::cout << "============================================\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    // --- Benchmark Identity ---
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        identity_16_16_asm(A.data(), B.data(), ld, ld, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double gib = (double)iterations * 2.0 * 1024.0 / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Identity: " << std::fixed << std::setprecision(2)
              << gib / diff.count() << " GiB/s\n";

    // --- Benchmark ReLU ---
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        relu_16_16_asm(A.data(), B.data(), ld, ld, 0);
    }
    end = std::chrono::high_resolution_clock::now();

    diff = end - start;
    std::cout << "ReLU:     " << std::fixed << std::setprecision(2)
              << gib / diff.count() << " GiB/s\n";

    // --- Benchmark Zero ---
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        zero_16_16_asm(B.data(), ld);
    }
    end = std::chrono::high_resolution_clock::now();

    diff = end - start;
    double gib_zero = (double)iterations * 1024.0 / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Zero:     " << std::fixed << std::setprecision(2)
              << gib_zero / diff.count() << " GiB/s\n";
}



// ============================================================================
//                         Benchmark: Binary-GEMM 
// ============================================================================
template <typename Func>
void run_bench(const char* name, int iters, double flops_per_call, Func func) {
    // Warmup
    func();
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    double seconds = duration<double>(end - start).count() / iters;
    double gflops = (flops_per_call / seconds) * 1e-9;

    std::cout << std::left << std::setw(20) << name 
              << " | Performance: " << std::fixed << std::setprecision(2) 
              << std::right << std::setw(8) << gflops << " GFLOPS" << std::endl;
}

int benchmark_binary() {
    const int iters = 100;
    const int64_t DIM = 512;
    const int64_t TILE = 32;

    std::vector<float> A(DIM * DIM, 1.1f);
    std::vector<float> B(DIM * DIM, 1.2f);
    std::vector<float> C(DIM * DIM, 0.0f);

    std::cout << "SME GEMM Performance Benchmark\n";
    std::cout << "========================================\n";

    // Bench Level 1 (Expected to be slowest due to Load/Store ratio)
    run_bench("gemm_32_32_1", 10000, 2.0 * TILE * TILE * 1, [&]() {
        gemm_32_32_1(A.data(), B.data(), C.data(), TILE, TILE, TILE);
    });

    // Bench Level 2 (Middle ground)
    run_bench("gemm_32_32_512", 1000, 2.0 * TILE * TILE * DIM, [&]() {
        gemm_32_32_512(A.data(), B.data(), C.data(), TILE, TILE, TILE);
    });

    // Bench Level 3 (Tiled M)
    run_bench("gemm_512_32_512", 200, 2.0 * DIM * TILE * DIM, [&]() {
        gemm_512_32_512(A.data(), B.data(), C.data(), DIM, TILE, DIM);
    });

    // Bench Level 4 (Full Matrix - Should be fastest)
    run_bench("gemm_512_512_512", iters, 2.0 * DIM * DIM * DIM, [&]() {
        gemm_512_512_512(A.data(), B.data(), C.data(), DIM, DIM, DIM);
    });

    std::cout << "========================================\n";
    return 0;
}



// ============================================================================
//                                   MAIN
// ============================================================================
int main() {
    benchmark_unary();
    benchmark_binary();
    return 0;
}
