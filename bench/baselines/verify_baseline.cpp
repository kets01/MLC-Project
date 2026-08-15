// Sprint 7.5 — cross-verify an external library's output against OUR reference.
//
// Decision B says no throughput number is trusted for a kernel that has not
// been verified against the C++ reference.  That applies to baselines too, and
// for a sharper reason than usual: a vendor comparison can be wrong not because
// the timing is wrong but because the two sides are computing DIFFERENT THINGS.
// That is precisely how the Sprint-6 comparison failed — `vDSP_normalize` was
// called "LayerNorm" when it has no eps, no gamma and no beta.
//
// So this checker exists to answer one question before any GiB/s is quoted:
// does the library compute the same function we do?
//
// Layout bridge: the library writes ROW-major [M rows, N features] with the
// norm over the last (contiguous) axis.  Our reference is COLUMN-major,
// a[row + col*ld], normalized over N.  The transpose happens here, in the
// checker, and nowhere in either timed path — neither implementation is made to
// pay for the other's layout (Sprint 7.5 harness rule 1).

#include "norm/norm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path, size_t expect) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    std::vector<float> v(expect);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(expect * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != expect * sizeof(float)) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        return {};
    }
    return v;
}

// Largest absolute difference, and where it occurred.
double max_abs_diff(const std::vector<float>& got_rowmajor,
                    const std::vector<float>& ref_colmajor,
                    int64_t m, int64_t n, size_t& worst_at) {
    double worst = 0.0;
    worst_at = 0;
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            const size_t i_row = static_cast<size_t>(r * n + c);
            const size_t i_col = static_cast<size_t>(r + c * m);
            const double d = std::fabs(static_cast<double>(got_rowmajor[i_row]) -
                                       static_cast<double>(ref_colmajor[i_col]));
            if (d > worst) { worst = d; worst_at = i_row; }
        }
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: verify_baseline <dumpdir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    int64_t m = 0, n = 0;
    float   eps = 1e-5f;
    {
        std::ifstream f(dir + "/shape.txt");
        if (!f) { std::fprintf(stderr, "missing shape.txt\n"); return 2; }
        f >> m >> n >> eps;
    }
    const size_t count = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::printf("Cross-verification against the C++ float64 reference\n");
    std::printf("  shape M=%lld N=%lld  eps=%g\n",
                static_cast<long long>(m), static_cast<long long>(n),
                static_cast<double>(eps));

    const auto in_row  = read_f32(dir + "/input.f32", count);
    const auto gamma   = read_f32(dir + "/gamma.f32", static_cast<size_t>(n));
    const auto beta    = read_f32(dir + "/beta.f32",  static_cast<size_t>(n));
    const auto lib_ln  = read_f32(dir + "/torch_ln.f32",  count);
    const auto lib_rms = read_f32(dir + "/torch_rms.f32", count);
    if (in_row.empty() || gamma.empty() || beta.empty() ||
        lib_ln.empty() || lib_rms.empty()) return 2;

    // Transpose the library's row-major input into our column-major layout.
    std::vector<float> in_col(count);
    for (int64_t r = 0; r < m; ++r)
        for (int64_t c = 0; c < n; ++c)
            in_col[static_cast<size_t>(r + c * m)] =
                in_row[static_cast<size_t>(r * n + c)];

    std::vector<float> ref_ln(count, 0.0f), ref_rms(count, 0.0f);
    mini_jit::norm::layer_norm_ref(in_col.data(), ref_ln.data(), gamma.data(),
                                   beta.data(), m, n, m, m, eps);
    mini_jit::norm::rms_norm_ref(in_col.data(), ref_rms.data(), gamma.data(),
                                 m, n, m, m, eps);

    size_t at_ln = 0, at_rms = 0;
    const double d_ln  = max_abs_diff(lib_ln,  ref_ln,  m, n, at_ln);
    const double d_rms = max_abs_diff(lib_rms, ref_rms, m, n, at_rms);

    // FP32 tolerance: the library accumulates in FP32 like our kernels, so
    // agreement to ~1e-5 means "same function", while a semantic mismatch
    // (missing eps, unbiased variance, no gamma) shows up orders larger.
    const double kTol = 1e-4;
    std::printf("  LayerNorm : max|diff| = %.3e   %s\n", d_ln,
                d_ln <= kTol ? "SAME FUNCTION" : "*** MISMATCH ***");
    std::printf("  RMSNorm   : max|diff| = %.3e   %s\n", d_rms,
                d_rms <= kTol ? "SAME FUNCTION" : "*** MISMATCH ***");

    if (d_ln > kTol || d_rms > kTol) {
        std::printf("\nA mismatch here invalidates the throughput comparison: the two sides\n"
                    "are not computing the same function, so their speeds are not comparable.\n");
        return 1;
    }
    std::printf("\nBoth verified. The throughput comparison is between implementations of\n"
                "the same function, which is what makes it meaningful.\n");
    return 0;
}
