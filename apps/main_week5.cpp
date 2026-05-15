#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "week5/jit_engine.hpp"
#include "week3/utility.hpp"

extern std::vector<uint32_t> get_identity_jit_opcodes();
extern std::vector<uint32_t> get_zero_jit_opcodes();
extern std::vector<uint32_t> get_relu_jit_opcodes();
extern std::vector<uint32_t> get_gemm_jit_opcodes();

using namespace std::chrono;

int main() {
    if (!cpu_supports_sme()) return 0;

    auto identity = JitEngine::generate<void(*)(const float*, float*, int64_t, int64_t, int32_t)>(get_identity_jit_opcodes());
    auto relu     = JitEngine::generate<void(*)(const float*, float*, int64_t, int64_t, int32_t)>(get_relu_jit_opcodes());
    auto zero     = JitEngine::generate<void(*)(float*, int64_t)>(get_zero_jit_opcodes());
    auto gemm     = JitEngine::generate<void(*)(const float*, const float*, float*, int64_t, int64_t, int64_t)>(get_gemm_jit_opcodes());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "--- Week 5 JIT Benchmarks ---\n";

    // Unary Benchmarks (16x16)
    const int unary_iters = 10'000'000;
    std::vector<float> A(256, 1.0f), B(256, 0.0f);
    double gib_factor = (double)unary_iters * 2.0 * 1024.0 / (1024.0 * 1024.0 * 1024.0);

    auto s = high_resolution_clock::now();
    for(int i=0; i<unary_iters; i++) identity(A.data(), B.data(), 16, 16, 0);
    std::cout << "JIT Identity: " << gib_factor / duration<double>(high_resolution_clock::now()-s).count() << " GiB/s\n";

    s = high_resolution_clock::now();
    for(int i=0; i<unary_iters; i++) relu(A.data(), B.data(), 16, 16, 0);
    std::cout << "JIT ReLU:     " << gib_factor / duration<double>(high_resolution_clock::now()-s).count() << " GiB/s\n";

    s = high_resolution_clock::now();
    for(int i=0; i<unary_iters; i++) zero(B.data(), 16);
    std::cout << "JIT Zero:     " << (gib_factor/2.0) / duration<double>(high_resolution_clock::now()-s).count() << " GiB/s\n";

    // Binary Benchmark (512x512x512)
    const int64_t DIM = 512;
    std::vector<float> MA(DIM*DIM, 1.1f), MB(DIM*DIM, 1.2f), MC(DIM*DIM, 0.0f);
    s = high_resolution_clock::now();
    for(int i=0; i<50; i++) gemm(MA.data(), MB.data(), MC.data(), DIM, DIM, DIM);
    double avg_s = duration<double>(high_resolution_clock::now()-s).count() / 50.0;
    std::cout << "JIT GEMM:     " << (2.0*DIM*DIM*DIM/avg_s)*1e-9 << " GFLOPS\n";

    return 0;
}