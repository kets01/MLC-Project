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