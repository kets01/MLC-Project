#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include "week7/TeirRuntime.hpp"
#include "week7/TeirParser.hpp"

using namespace mini_jit::teir;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_sec(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char** argv) {
    // Allow overriding data directory via first argument
    std::string data_dir = (argc > 1) ? argv[1] : "data";

    TeirRuntime rt;

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "         WEEK 7: TEIR RUNTIME PERFORMANCE REPORT\n";
    std::cout << std::string(72, '=') << "\n";
    std::cout << std::left
              << std::setw(22) << "Kernel"
              << std::setw(28) << "Dimensions"
              << std::setw(14) << "Performance"
              << "Status\n";
    std::cout << std::string(72, '-') << "\n";

    // ─────────────────────────────────────────────────────────────────────
    // 1. Transposition: abcd → dbac with (a=96, b=128, c=48, d=32)
    // ─────────────────────────────────────────────────────────────────────
    {
        const std::string path = data_dir + "/transposition.teir";
        try {
            TeirObject obj = parse_file(path);
            auto result = rt.compile(obj);
            if (!result) throw std::runtime_error(result.error);

            const uint64_t a=96, b=128, c=48, d=32;
            uint64_t total = a*b*c*d;  // total float elements
            std::vector<float> in(total, 1.1f), out(total, 0.0f);
            float* ptrs[] = { in.data(), out.data() };

            // Warm-up
            result.kernel(ptrs);
            std::fill(out.begin(), out.end(), 0.0f);

            // Timed run
            auto t0 = Clock::now();
            result.kernel(ptrs);
            auto t1 = Clock::now();

            double sec  = elapsed_sec(t0, t1);
            // Bandwidth: 2 arrays of total f32 elements = 2 * total * 4 bytes
            double gibs = (2.0 * total * sizeof(float)) / (1e9 * sec);

            std::cout << std::left
                      << std::setw(22) << "Transposition"
                      << std::setw(28) << "96x128x48x32"
                      << std::fixed << std::setprecision(2)
                      << std::setw(14) << (std::to_string(static_cast<int>(gibs*100)/100.0).substr(0,5) + " GiB/s")
                      << "OK\n";
        } catch (const std::exception& e) {
            std::cout << std::setw(22) << "Transposition"
                      << "FAILED: " << e.what() << "\n";
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 2. Matmul: mk,kn→mn with (m=8192, k=8192, n=8192)
    //    blocked as m0k0m1k1,k0n0k1n1→m0n0m1n1
    //    with (m0=256, k0=16, m1=32, k1=512, n0=128, n1=64)
    // ─────────────────────────────────────────────────────────────────────
    {
        const std::string path = data_dir + "/matmul.teir";
        try {
            TeirObject obj = parse_file(path);
            auto result = rt.compile(obj);
            if (!result) throw std::runtime_error(result.error);

            const uint64_t N = 8192;
            std::vector<float> A(N*N, 1.0f), B(N*N, 1.0f), C(N*N, 0.0f);
            float* ptrs[] = { A.data(), B.data(), C.data() };

            // Warm-up (smaller would be better but the spec requires full size)
            auto t0 = Clock::now();
            result.kernel(ptrs);
            auto t1 = Clock::now();

            double sec    = elapsed_sec(t0, t1);
            double gflops = (2.0 * N * N * N) / (1e9 * sec);

            std::cout << std::left
                      << std::setw(22) << "Matmul"
                      << std::setw(28) << "8192x8192x8192"
                      << std::fixed << std::setprecision(2)
                      << std::setw(14) << (std::to_string(static_cast<int>(gflops*100)/100.0).substr(0,6) + " GFLOP/s")
                      << "OK\n";
        } catch (const std::exception& e) {
            std::cout << std::setw(22) << "Matmul"
                      << "FAILED: " << e.what() << "\n";
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 3. Contraction: pqtu,trus→pqrs with (p=128, q=96, r=96, s=64, t=32, u=256)
    // ─────────────────────────────────────────────────────────────────────
    {
        const std::string path = data_dir + "/contraction.teir";
        try {
            TeirObject obj = parse_file(path);
            auto result = rt.compile(obj);
            if (!result) throw std::runtime_error(result.error);

            // Tensor sizes derived from strides in contraction.teir:
            //   in0 (pqtu): largest offset = p*3145728 = 128*3145728 → 402MB
            //   in1 (trus): largest offset = t*6291456 = 32*6291456  → 201MB
            //   out (pqrs): largest offset = p*2359296 = 128*2359296 → 301MB
            // Allocate using the product of extents:
            const uint64_t p=128, q=96, t_=32, u=256; // t_ avoids clash with std::t
            const uint64_t r=96, s=64;
            std::vector<float> in0(p*q*t_*u, 1.0f);
            std::vector<float> in1(t_*r*u*s, 1.0f);
            std::vector<float> out(p*q*r*s, 0.0f);
            float* ptrs[] = { in0.data(), in1.data(), out.data() };

            auto t0 = Clock::now();
            result.kernel(ptrs);
            auto t1 = Clock::now();

            double sec    = elapsed_sec(t0, t1);
            // 2 * p*q*r*s*t*u multiply-adds
            double gflops = (2.0 * p*q*r*s*t_*u) / (1e9 * sec);

            std::cout << std::left
                      << std::setw(22) << "Contraction"
                      << std::setw(28) << "128x96x96x64x32x256"
                      << std::fixed << std::setprecision(2)
                      << std::setw(14) << (std::to_string(static_cast<int>(gflops*100)/100.0).substr(0,6) + " GFLOP/s")
                      << "OK\n";
        } catch (const std::exception& e) {
            std::cout << std::setw(22) << "Contraction"
                      << "FAILED: " << e.what() << "\n";
        }
    }

    std::cout << std::string(72, '=') << "\n";
    return 0;
}