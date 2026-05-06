#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week3/gemm_sme.hpp"

int main() {
    const int64_t M = 512;
    const int64_t N = 512;
    const int64_t K = 512;
    const int num_iterations = 100;

    // Allocate matrices
    std::vector<float> A(M * K, 1.1f);
    std::vector<float> B(K * N, 1.2f);
    std::vector<float> C(M * N, 0.0f);

    std::cout << "Benchmarking SME GEMM: " << M << "x" << N << "x" << K << std::endl;

    // 1. Warm-up run (crucial for SME/Streaming mode and CPU frequency)
    gemm_512_512_512(A.data(), B.data(), C.data(), M, K, M);

    // 2. Timed Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        gemm_512_512_512(A.data(), B.data(), C.data(), M, K, M);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 3. Calculate Performance
    double seconds = diff.count();
    double avg_seconds = seconds / num_iterations;
    
    // Total FLOPs = 2 * M * N * K
    double total_flops = 2.0 * M * N * K;
    double gflops = (total_flops / avg_seconds) * 1e-9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Total Time:    " << seconds << " s" << std::endl;
    std::cout << "Avg Time:      " << avg_seconds << " s" << std::endl;
    std::cout << "Performance:   " << gflops << " GFLOPS" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    return 0;
}