#ifndef MINI_JIT_TEIR_PARSER_HPP
#define MINI_JIT_TEIR_PARSER_HPP

#include "Teir.hpp"
#include <string>
#include <stdexcept>

namespace mini_jit::teir {

// Parse a TEIR source string and return a populated TeirObject.
// Throws std::runtime_error on parse errors.
TeirObject parse(const std::string& source);

// Convenience: parse from file path
TeirObject parse_file(const std::string& path);

} // namespace mini_jit::teir
#endif