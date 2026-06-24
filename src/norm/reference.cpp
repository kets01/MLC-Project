#include "norm/norm.hpp"
#include <cmath>

namespace mini_jit::norm {

// LayerNorm: two-pass per row.
// Pass 1: accumulate mean and variance (Welford-style for stability is not
//         needed here — the two-pass structure itself avoids catastrophic
//         cancellation because variance is computed from (x - mean)^2).
// Pass 2: normalize, scale, shift.
void layer_norm_ref(const float* a,
                    float*       b,
                    const float* gamma,
                    const float* beta,
                    int64_t      m,
                    int64_t      n,
                    int64_t      ld_a,
                    int64_t      ld_b,
                    float        epsilon) {
    for (int64_t row = 0; row < m; ++row) {
        // Pass 1: mean
        double sum = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            sum += static_cast<double>(a[row + col * ld_a]);
        }
        double mean = sum / static_cast<double>(n);

        // Pass 1: variance  E[(x - mean)^2]
        double var = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            double diff = static_cast<double>(a[row + col * ld_a]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(n);

        double inv_std = 1.0 / std::sqrt(var + static_cast<double>(epsilon));

        // Pass 2: normalize + scale + shift
        for (int64_t col = 0; col < n; ++col) {
            double x_hat = (static_cast<double>(a[row + col * ld_a]) - mean) * inv_std;
            b[row + col * ld_b] = static_cast<float>(
                x_hat * static_cast<double>(gamma[col]) + static_cast<double>(beta[col]));
        }
    }
}

// RMSNorm: single-pass per row.
// No mean subtraction and no beta — only the RMS of x is used.
// Uses double accumulation to keep the reference numerically honest.
void rms_norm_ref(const float* a,
                  float*       b,
                  const float* gamma,
                  int64_t      m,
                  int64_t      n,
                  int64_t      ld_a,
                  int64_t      ld_b,
                  float        epsilon) {
    for (int64_t row = 0; row < m; ++row) {
        double sumsq = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            double x = static_cast<double>(a[row + col * ld_a]);
            sumsq += x * x;
        }
        double rms     = std::sqrt(sumsq / static_cast<double>(n) + static_cast<double>(epsilon));
        double inv_rms = 1.0 / rms;

        for (int64_t col = 0; col < n; ++col) {
            b[row + col * ld_b] = static_cast<float>(
                static_cast<double>(a[row + col * ld_a]) * inv_rms
                * static_cast<double>(gamma[col]));
        }
    }
}

} // namespace mini_jit::norm
