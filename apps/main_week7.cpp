#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week7/TeirRuntime.h"
#include "week3/utility.hpp"

using namespace mini_jit::teir;

int main() {
    TeirRuntime runtime;
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "         WEEK 7 TEIR PERFORMANCE REPORT\n";
    std::cout << std::string(60, '=') << "\n";

    // Task 1: Transposition
    {
        size_t size = 96 * 128 * 48 * 32;
        std::vector<float> A(size, 1.0f), B(size, 0.0f);
        auto func = runtime.compile(runtime.build_transposition_tree());
        auto start = std::chrono::high_resolution_clock::now();
        func(A.data(), B.data(), nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end-start).count();
        std::cout << std::left << std::setw(20) << "Transposition" << ": " 
                  << std::fixed << std::setprecision(2) << (size * 8.0 / 1e9 / sec) << " GiB/s\n";
    }

    // Task 2: Matmul
    if (cpu_supports_sme()) {
        size_t N = 8192;
        std::vector<float> A(N*N, 1.0f), B(N*N, 1.0f), C(N*N, 0.0f);
        auto func = runtime.compile(runtime.build_matmul_tree());
        auto start = std::chrono::high_resolution_clock::now();
        func(A.data(), B.data(), C.data());
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end-start).count();
        std::cout << std::left << std::setw(20) << "Matrix Mult" << ": " 
                  << (2.0 * N * N * N / 1e9 / sec) << " GFLOPS\n";
    }

    std::cout << std::string(60, '=') << "\n";
    return 0;
}