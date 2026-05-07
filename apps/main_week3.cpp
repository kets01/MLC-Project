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
//                         Benchmark: GEMM (512x512x512)
// ============================================================================
void benchmark_gemm() {
    const int64_t M = 512;
    const int64_t N = 512;
    const int64_t K = 512;
    const int num_iterations = 100;

    std::vector<float> A(M * K, 1.1f);
    std::vector<float> B(K * N, 1.2f);
    std::vector<float> C(M * N, 0.0f);

    std::cout << "\n============================================\n";
    std::cout << "          Benchmarking SME GEMM\n";
    std::cout << "============================================\n";
    std::cout << "Matrix size: " << M << " x " << N << " x " << K << "\n";
    std::cout << "Iterations:  " << num_iterations << "\n\n";

    // Warm-up run (important for SME/streaming mode)
    gemm_512_512_512(A.data(), B.data(), C.data(), M, K, M);

    // Timed benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        gemm_512_512_512(A.data(), B.data(), C.data(), M, K, M);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    double seconds = diff.count();
    double avg_seconds = seconds / num_iterations;

    // Total FLOPs = 2 * M * N * K
    double total_flops = 2.0 * M * N * K;
    double gflops = (total_flops / avg_seconds) * 1e-9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Time:    " << seconds << " s\n";
    std::cout << "Avg Time:      " << avg_seconds << " s\n";
    std::cout << "Performance:   " << gflops << " GFLOPS\n";
    std::cout << "--------------------------------------------\n";
}



// ============================================================================
//                                   MAIN
// ============================================================================
int main() {
    benchmark_unary();
    benchmark_gemm();
    return 0;
}
