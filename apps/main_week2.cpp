#include <iostream>
#include <chrono>
#include <iomanip>
#include "week2/fmadd.hpp"
#include "week2/perm.hpp"

// Helper to print input tensor (8 x 4 x size_c)
void printABC(int64_t size_c, const float* data) {
    std::cout << "Input Tensor (abc) [8 x 4 x " << size_c << "]:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "i=" << i << ":" << std::endl;
        for (int j = 0; j < 4; ++j) {
            std::cout << "  j=" << j << " | ";
            for (int k = 0; k < size_c; ++k) {
                std::cout << std::setw(4) << data[(i * 4 + j) * size_c + k] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// Helper to print output tensor (size_c x 4 x 8)
void printCBA(int64_t size_c, const float* data) {
    std::cout << "Output Tensor (cba) [" << size_c << " x 4 x 8]:" << std::endl;
    for (int k = 0; k < size_c; ++k) {
        std::cout << "k=" << k << " (New contiguous dimension is 'a'):" << std::endl;
        for (int j = 0; j < 4; ++j) {
            std::cout << "  j=" << j << " | ";
            for (int i = 0; i < 8; ++i) {
                std::cout << std::setw(4) << data[k * 32 + j * 8 + i] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "----------------------------------------------------" << std::endl;
}

void verify_permutation(int64_t size_c) {
    int64_t total = 8 * 4 * size_c;
    std::vector<float> input(total);
    std::vector<float> output(total, 0.0f);

    // Initialize: Value = 100*i + 10*j + k
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < size_c; ++k) {
                input[(i * 4 + j) * size_c + k] = (i * 100) + (j * 10) + k;
            }
        }
    }

    std::cout << ">>> RUNNING PERMUTATION FOR C = " << size_c << " <<<" << std::endl;
    printABC(size_c, input.data());

    // Execute Neon Kernel
    perm_neon_abc_cba(size_c, input.data(), output.data());

    printCBA(size_c, output.data());
}


void run_fmadd_benchmark() {
    // 1 Billion iterations. Each iteration has 8 FMADD instructions.
    // Total instructions = 8,000,000,000.
    const uint64_t iterations = 1000000000; 
    const int instructions_per_loop = 8;
    
    std::cout << "--- AArch64 FMADD Throughput Benchmark ---" << std::endl;
    std::cout << "Running " << iterations << " iterations (" 
              << (iterations * instructions_per_loop / 1e9) << " billion instructions)..." << std::endl;

    // Start high-resolution timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Call the assembly function
    fmadd_asm(iterations);
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    // Calculations
    double total_instr = (double)iterations * instructions_per_loop;
    double giga_instr_per_sec = (total_instr / seconds) / 1e9;
    
    // In ML, FMADD is counted as 2 operations (1 Multiply + 1 Add)
    double gflops = (total_instr * 2 / seconds) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Execution Time : " << seconds << " seconds" << std::endl;
    std::cout << "Throughput     : " << giga_instr_per_sec << " G-Instr/sec" << std::endl;
    std::cout << "Performance    : " << gflops << " GFLOPS" << std::endl;

}


int main() {
    run_fmadd_benchmark(); // Second, measure throughput
    verify_permutation(2);
    verify_permutation(4);
    return 0;
}
