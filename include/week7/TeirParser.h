#ifndef MINI_JIT_TEIR_PARSER_H
#define MINI_JIT_TEIR_PARSER_H

#include "Teir.h"
#include <memory>
#include <string>

namespace mini_jit::teir {

// Parses a .teir file (tensors, axes, primitives, schedule — including
// `guard first(@axis)` and multi-child iteration steps) into the Node tree
// TeirRuntime::traverse() already knows how to execute.
//
// Pure text -> tree: this does not generate or cache any kernels. Each
// built Invocation carries the shape (m, n, k, ld_a, ld_b, ld_c, eps) its
// primitive needs; TeirRuntime resolves/caches the actual kernel_t lazily
// at execution time, keeping the parser independent of the JIT generators.
//
// Throws std::runtime_error with a descriptive message on malformed input
// or an unresolved @name reference (decision: fail loud, not silently).
std::shared_ptr<Node> parse_teir_file(const std::string& path);

} // namespace mini_jit::teir

#endif
