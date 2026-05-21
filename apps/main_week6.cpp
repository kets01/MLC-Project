#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
//#include "week6/unary.hpp"
#include "week6/gemm.hpp"

using namespace std;

void run_benchmarks() {
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
    run_benchmarks();
    return 0;
}