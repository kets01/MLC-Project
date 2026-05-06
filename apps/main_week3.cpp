#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week3/unary.hpp"

// Calculations for 16x16 FP32 Matrix
// Total elements = 256
// Bytes per matrix = 256 * 4 = 1024 bytes (1 KiB)

void benchmark_unary() {
    const int64_t ld = 16;
    const int64_t iterations = 10000000; // 10 million runs
    std::vector<float> A(256, 1.0f);
    std::vector<float> B(256, 0.0f);

    std::cout << "Benchmarking Unary Kernels (" << iterations << " iterations)..." << std::endl;

    // --- Benchmark Identity ---
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        identity_16_16_asm(A.data(), B.data(), ld, ld, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    // Bandwidth: (Read A + Write B) * iterations
    double gib = (double)iterations * 2.0 * 1024.0 / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Identity: " << std::fixed << std::setprecision(2) 
              << gib / diff.count() << " GiB/s" << std::endl;

    // --- Benchmark ReLU ---
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        relu_16_16_asm(A.data(), B.data(), ld, ld, 0);
    }
    end = std::chrono::high_resolution_clock::now();
    
    diff = end - start;
    std::cout << "ReLU:     " << std::fixed << std::setprecision(2) 
              << gib / diff.count() << " GiB/s" << std::endl;

    // --- Benchmark Zero ---
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        zero_16_16_asm(B.data(), ld);
    }
    end = std::chrono::high_resolution_clock::now();
    
    diff = end - start;
    // Bandwidth: (Write B only) * iterations
    double gib_zero = (double)iterations * 1024.0 / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Zero:     " << std::fixed << std::setprecision(2) 
              << gib_zero / diff.count() << " GiB/s" << std::endl;
}

int main() {
    benchmark_unary();
    return 0;
}