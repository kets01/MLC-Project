// formatter.cpp — standalone developer tool, NOT part of the CMake build.
//
// Turns an objdump disassembly into the uint32_t opcode arrays in
// week5/jit_kernel.cpp, which is how the week-5 JIT kernels were produced
// (documented in docs/source/weeks/week5.rst):
//
//   clang -arch arm64 -march=armv9-a+sme+sve -c gemm_512_512_512.S -o temp.o
//   llvm-objdump -d temp.o | ./formatter.o
//
// It has its own main(), so it is deliberately excluded from week5_lib —
// building it into the library would collide with every test/bench main().
// Build it on demand:  c++ -std=c++17 -o formatter.o src/week5/formatter.cpp
//
// Sprint 4 superseded this workflow for new kernels: mini_jit::Norm emits
// through InstGen encoders that are unit-tested against golden words, rather
// than pasting pre-assembled opcode blobs. Kept because week5.rst documents
// it as the reproduction path for the week-5 kernels.

#include <iostream>
#include <string>
#include <regex>
#include <iomanip>

int main() {
    // Regex to match: [address]: [hex_opcode] [instruction_text]
    // Group 1: Hex Opcode
    // Group 2: Assembly Instruction
    std::regex line_regex(R"(^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.*))");
    std::string line;

    std::cout << "const uint32_t kernel_bin[] = {" << std::endl;

    while (std::getline(std::cin, line)) {
        std::smatch match;
        if (std::regex_search(line, match, line_regex)) {
            std::string opcode = match[1];
            std::string assembly = match[2];

            // Print formatted line
            std::cout << "    0x" << opcode << ",      //" << assembly << std::endl;
        } else if (line.find("<") != std::string::npos && line.find(">:") != std::string::npos) {
            // Optional: Print label names as comments
            std::cout << "    // " << line << std::endl;
        }
    }

    std::cout << "};" << std::endl;

    return 0;
}