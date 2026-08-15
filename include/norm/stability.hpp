#pragma once
#include <cstdint>

// Sprint 7a — numerical-stability instruments.
//
// These are ANALYSIS functions, not kernels and not part of the primitive
// contract in norm.hpp.  Nothing in the library calls them; they exist so the
// report's stability claims can be measured rather than asserted.
//
// The project's design (context.md §8) rests on three assertions:
//   1. single-pass variance via E[x^2] - mean^2 suffers catastrophic
//      cancellation in FP32 for large or shifted inputs,
//   2. LayerNorm's centred two-pass formulation avoids it,
//   3. RMSNorm sidesteps the problem entirely because it never forms a mean.
// Until now the project simply never implemented (1), so the claim had no
// measured counterexample.  These functions supply it.
//
// Why FP32 accumulation is deliberate: the kernels accumulate in FP32, so a
// float64 demonstrator would prove nothing about them.  The reference in
// reference.cpp accumulates in double ON PURPOSE (it is the oracle); these
// mirror the kernels' precision instead.
//
// Why the summation order is trustworthy: the build sets no -ffast-math /
// -funsafe-math-optimizations, so under strict IEEE the compiler may not
// reassociate a floating-point reduction.  The sequential order written here
// is the order that executes, which is what makes the three estimators
// comparable to each other and to the kernels.

namespace mini_jit::norm::stability {

// The oracle: variance in double precision, centred two-pass.
// Every error figure below is measured against this.
double variance_ref_f64(const float* x, int64_t n);

// (1) Naive single-pass "textbook" variance: accumulate Sum(x) and Sum(x^2) in
// one sweep, then var = E[x^2] - mean^2.  Cheap, one pass, and catastrophically
// cancelling: for shifted data both terms grow like mean^2 while their
// DIFFERENCE stays at the true variance, so the subtraction annihilates the
// significant bits.  Can return a NEGATIVE variance, which is what makes it
// unusable rather than merely inaccurate: sqrt() of it is NaN.
float variance_naive_f32(const float* x, int64_t n);

// (2) Centred two-pass: form the mean first, then accumulate (x - mean)^2.
// The squared terms are small because the data is already centred, so there is
// no large-minus-large subtraction.  This is what layer_norm_ref and every
// LayerNorm kernel in this project use.
float variance_twopass_f32(const float* x, int64_t n);

// (3) Welford's online recurrence: one pass, updating (mean, M2) per element.
// Included because Sprint 2c benchmarked a Welford KERNEL and Sprint 6 §1.3
// found the accuracy claim about it was wrong; this isolates the arithmetic
// from the kernel so the two can be told apart.
float variance_welford_f32(const float* x, int64_t n);

// The conditioning of the variance problem for this input: 1 + mean^2/var.
// The naive estimator's relative error grows with this quantity, which is why
// a "shift" is the natural stress axis — shifting data by s leaves the variance
// untouched but scales the condition number by ~s^2.
double variance_condition_number(const float* x, int64_t n);

// RMSNorm's own limit, and the honest counterpart to the LayerNorm story.
// RMSNorm is immune to shift (it never forms a mean, so there is nothing to
// cancel), but it must accumulate Sum(x^2) in FP32, which OVERFLOWS to +inf
// once the magnitudes are large enough.  Returns the FP32 sum of squares so a
// caller can find where it stops being finite.
float sumsq_f32(const float* x, int64_t n);

} // namespace mini_jit::norm::stability
