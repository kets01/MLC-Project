#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week7/TeirRuntime.h"
#include "week3/utility.hpp"

using namespace mini_jit::teir;

int main() {
    TeirRuntime runtime;
    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "           WEEK 7: TEIR RUNTIME PERFORMANCE REPORT\n";
    std::cout << std::string(75, '=') << "\n";
    std::cout << std::left << std::setw(22) << "TASK" 
              << std::setw(25) << "DIMENSIONS" 
              << "PERFORMANCE" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    // --- Task 1: Transposition ---
    {
        size_t size = 96 * 128 * 48 * 32;
        std::vector<float> A(size, 1.1f), B(size, 0.0f);
        auto tree = runtime.build_transposition_tree();
        auto start = std::chrono::high_resolution_clock::now();
        runtime.execute(tree.get(), A.data(), B.data(), nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end-start).count();
        double gibs = (size * 8.0 / 1e9 / sec);
        std::cout << std::left << std::setw(22) << "Transposition" 
                  << std::setw(25) << "96x128x48x32" 
                  << std::fixed << std::setprecision(2) << gibs << " GiB/s\n";
    }

    // --- Task 2: Matmul ---
    if (cpu_supports_sme()) {
        size_t N = 8192;
        std::vector<float> A(N*N, 1.0f), B(N*N, 1.0f), C(N*N, 0.0f);
        auto tree = runtime.build_matmul_tree();
        auto start = std::chrono::high_resolution_clock::now();
        runtime.execute(tree.get(), A.data(), B.data(), C.data());
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end-start).count();
        double gflops = (2.0 * N * N * N / 1e9 / sec);
        std::cout << std::left << std::setw(22) << "Matrix Mult" 
                  << std::setw(25) << "8192^3" 
                  << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    }

    // --- Task 3: Contraction ---
    {
        // pqtu, trus -> pqrs | (128, 96, 96, 64, 32, 256)
        const uint64_t p=128, q=96, t=96, u=64, r=32, s=256;
        size_t size_t1 = p*q*t*u;
        size_t size_t2 = t*r*u*s;
        size_t size_res = p*q*r*s;
        
        std::vector<float> T1(size_t1, 1.0f), T2(size_t2, 1.0f), Res(size_res, 0.0f);
        
        auto tree = runtime.build_contraction_tree();
        auto start = std::chrono::high_resolution_clock::now();
        runtime.execute(tree.get(), T1.data(), T2.data(), Res.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        double sec = std::chrono::duration<double>(end-start).count();
        
        // Operation count for contraction: 2 * (p*q*r*s*t*u)
        double total_ops = 2.0 * p * q * r * s * t * u;
        double gflops = (total_ops / 1e9) / sec;
        
        std::cout << std::left << std::setw(22) << "Tensor Contraction" 
                  << std::setw(25) << "128x96x...x256" 
                  << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    }

    std::cout << std::string(75, '=') << "\n";
    return 0;
}