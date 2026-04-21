#include <iostream>
#include <chrono>
#include <iomanip>
#include "week2/fmadd.hpp"

int main() {
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

    return 0;
}