

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "week6/unary.hpp"
#include "week6/gemm.hpp"

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

using namespace std;
void run_gemm_bench() {
    uint32_t dims[] = {64, 128, 512};

    // --- GEMM Benchmarks 
    cout << "\n--- GEMM Benchmarks (GFLOPS) ---" << endl;
    mini_jit::Gemm g_gen;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                uint32_t M = dims[i];
                uint32_t N = dims[j];
                uint32_t K = dims[k];
                
                g_gen.generate(M, N, K, 0, 0, 0, mini_jit::Gemm::dtype_t::fp32);
                auto kernel = g_gen.get_kernel();

                vector<float> A(M*K, 1.0f), B(K*N, 1.0f), C(M*N, 0.0f);
                auto start = chrono::high_resolution_clock::now();
                for(int iter=0; iter<50; iter++) kernel(A.data(), B.data(), C.data(), M, K, M);
                auto end = chrono::high_resolution_clock::now();

                double secs = chrono::duration<double>(end - start).count() / 50.0;
                double gflops = (2.0 * M * N * K) / (secs * 1e9);
                cout << "M=" << M << ", N=" << N << ", K=" << K << " | " << gflops << " GFLOPS" << endl;
            }
        }
    }
}
int main() {
    uint32_t dims[] = {64, 128, 512};

    std::cout << "--- Unary Benchmarks (9 settings) ---" << std::endl;
    for (uint32_t m : dims) {
        for (uint32_t n : dims) {
            run_unary_bench(m, n);
        }
    }
    run_gemm_bench();

    return 0;
}