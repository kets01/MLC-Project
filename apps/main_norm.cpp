#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include "norm/norm.hpp"

int main() {
    const int64_t M = 512, N = 512, ld = 512;
    std::vector<float> a(M * N, 1.0f), b(M * N, 0.0f);

    const int reps = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        mini_jit::norm::norm_placeholder(a.data(), b.data(), M, N, ld);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count() / reps;
    double bytes   = static_cast<double>(M) * N * sizeof(float) * 2; // 1 read + 1 write
    double gibs    = (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

    std::cout << std::fixed << std::setprecision(2)
              << "norm_placeholder  M=" << M << " N=" << N
              << "  " << gibs << " GiB/s  [placeholder — Sprint 0]\n";
    return 0;
}
