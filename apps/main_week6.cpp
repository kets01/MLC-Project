#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "Unary.h"

void run_unary_bench(uint32_t M, uint32_t N) {
    mini_jit::Unary generator;
    // Generate Identity kernel
    generator.generate(M, N, 0, mini_jit::Unary::dtype_t::fp32, mini_jit::Unary::ptype_t::identity);
    auto kernel = generator.get_kernel();

    std::vector<float> A(M * N, 1.1f), B(M * N, 0.0f);
    
    // Timing loop
    auto start = std::chrono::high_resolution_clock::now();
    int reps = 1000;
    for(int i = 0; i < reps; ++i) {
        kernel(A.data(), B.data(), M, M);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate GiB/s
    double seconds = std::chrono::duration<double>(end - start).count() / reps;
    double bytes_processed = (double)M * N * sizeof(float) * 2; // Read A + Write B
    double gibs = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / seconds;

    std::cout << "Unary " << M << "x" << N << " | Performance: " 
              << std::fixed << std::setprecision(2) << gibs << " GiB/s" << std::endl;
}

int main() {
    uint32_t dims[] = {64, 128, 512};

    std::cout << "--- Unary Benchmarks (9 settings) ---" << std::endl;
    for (uint32_t m : dims) {
        for (uint32_t n : dims) {
            run_unary_bench(m, n);
        }
    }

    return 0;
}