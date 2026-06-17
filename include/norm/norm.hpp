#pragma once
#include <cstdint>

namespace mini_jit::norm {

// Placeholder: identity copy, column-major layout.
// The canonical kernel signature (with gamma/beta/epsilon) is pinned
// in Sprint 1 once the C++ reference is written.
void norm_placeholder(const float* a, float* b, int64_t m, int64_t n, int64_t ld);

} // namespace mini_jit::norm
