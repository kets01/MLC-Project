#pragma once
#include <cstdint>

namespace mini_jit::norm {

// Canonical kernel interface (decision A): shared by the C++ reference,
// the mini_jit::Norm JIT generator, and the TEIR registration.
//
// Layout: column-major, explicit leading dimension.  data[row + col*ld]
// Normalized axis: N (each of the M rows is normalized independently).
// gamma, beta: per-feature scale/shift vectors [N elements].

// LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
// Two-pass: pass 1 computes mean and variance, pass 2 normalizes.
void layer_norm_ref(const float* a,
                    float*       b,
                    const float* gamma,
                    const float* beta,
                    int64_t      m,
                    int64_t      n,
                    int64_t      ld_a,
                    int64_t      ld_b,
                    float        epsilon);

// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
// Single-pass: no mean subtraction, no beta.
void rms_norm_ref(const float* a,
                  float*       b,
                  const float* gamma,
                  int64_t      m,
                  int64_t      n,
                  int64_t      ld_a,
                  int64_t      ld_b,
                  float        epsilon);

} // namespace mini_jit::norm
