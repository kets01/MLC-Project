#include "norm/stability.hpp"

// Sprint 7a — see stability.hpp for why these exist and why they accumulate in
// FP32.  Each function is written as the plainest possible expression of its
// formulation: the point is to measure what the FORMULATION costs, so nothing
// here is optimized, blocked, or reassociated.

namespace mini_jit::norm::stability {

double variance_ref_f64(const float* x, int64_t n) {
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]);
    const double mean = sum / static_cast<double>(n);

    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        acc += d * d;
    }
    return acc / static_cast<double>(n);
}

// var = E[x^2] - (E[x])^2, both moments accumulated in one sweep.
//
// The failure is structural, not a rounding accident.  For data with mean m and
// variance v, E[x^2] ~ m^2 + v and mean^2 ~ m^2.  Once m^2 is large enough that
// v is below the ULP of m^2, both moments round to the SAME FP32 value and the
// subtraction yields exactly 0 -- or, when they round in opposite directions, a
// negative number.  No amount of care inside this loop fixes it; the
// information is destroyed before the subtraction happens.
float variance_naive_f32(const float* x, int64_t n) {
    if (n <= 0) return 0.0f;
    float sum   = 0.0f;
    float sumsq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum   += x[i];
        sumsq += x[i] * x[i];
    }
    const float inv_n = 1.0f / static_cast<float>(n);
    const float mean  = sum * inv_n;
    return sumsq * inv_n - mean * mean;
}

// Centred two-pass: the formulation every LayerNorm kernel in this project
// uses.  Pass 1 forms the mean; pass 2 accumulates (x - mean)^2, whose terms
// are O(variance) rather than O(mean^2), so there is no large-minus-large
// subtraction anywhere.  The cost is a second read of x -- which is exactly the
// 3R+1W traffic structure the Sprint-2 ablation spends its effort on, and the
// reason LayerNorm is ~2x slower than RMSNorm.  That trade is the point:
// accuracy bought with bandwidth.
float variance_twopass_f32(const float* x, int64_t n) {
    if (n <= 0) return 0.0f;
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += x[i];
    const float mean = sum / static_cast<float>(n);

    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        acc += d * d;
    }
    return acc / static_cast<float>(n);
}

// Welford: one pass, carrying (mean, M2).  The subtraction `x - mean` is done
// against a RUNNING mean, so it is never a large-minus-large cancellation the
// way naive's final subtraction is -- which is why Welford's variance is far
// better than naive's.  Its weakness is elsewhere and is easy to misattribute
// (Sprint 6 §1.3 did): `mean += delta/count` updates at FULL INPUT MAGNITUDE
// every element, so the running mean itself accumulates rounding at the scale
// of the data, and a LayerNorm output of (x - mean)*inv_std amplifies that.
float variance_welford_f32(const float* x, int64_t n) {
    if (n <= 0) return 0.0f;
    float mean = 0.0f;
    float m2   = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float count = static_cast<float>(i + 1);
        const float delta = x[i] - mean;
        mean += delta / count;
        m2   += delta * (x[i] - mean);   // uses the UPDATED mean, per Welford
    }
    return m2 / static_cast<float>(n);
}

// 1 + mean^2/var, evaluated in double so the conditioning measure is itself
// well conditioned.  Shifting data by s leaves var alone and scales mean^2 by
// ~s^2, so this is the knob the stress sweep turns.
double variance_condition_number(const float* x, int64_t n) {
    if (n <= 0) return 1.0;
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]);
    const double mean = sum / static_cast<double>(n);
    const double var  = variance_ref_f64(x, n);
    if (var <= 0.0) return 0.0;          // degenerate: constant input
    return 1.0 + (mean * mean) / var;
}

// RMSNorm's reduction, in the kernels' precision.  No mean is formed, so no
// cancellation is possible at any shift -- but Sum(x^2) grows without bound and
// FP32 tops out at ~3.4e38, so this is where RMSNorm's own limit lives.
float sumsq_f32(const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += x[i] * x[i];
    return acc;
}

} // namespace mini_jit::norm::stability
