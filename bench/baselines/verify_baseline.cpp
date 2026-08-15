// Cross-verify an external library's output against OUR reference.
//
// Decision B says no throughput number is trusted for an unverified kernel,
// and that applies to baselines for a sharper reason than usual: a vendor
// comparison can be wrong not because the timing is wrong but because the two
// sides compute DIFFERENT THINGS.  That is exactly how the earlier comparison
// failed — vDSP_normalize was called "LayerNorm" when it has no eps, no
// gamma and no beta.  So this answers one question before any GiB/s is
// quoted: does the library compute the same function we do?
//
// Layout bridge: the library writes ROW-major with the norm over the last
// contiguous axis; our reference is COLUMN-major.  The transpose happens
// HERE and in neither timed path, so neither implementation is made to pay
// for the other's layout.

#include "norm/norm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
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

// Verify one (shape) worth of dumps.  Returns the number of FAILED checks.
static int verify_shape(const std::string& dir, int64_t m, int64_t n, float eps) {
    const std::string tag =
        std::to_string(m) + "x" + std::to_string(n);
    const size_t count = static_cast<size_t>(m) * static_cast<size_t>(n);

    const auto in_row  = read_f32(dir + "/input_" + tag + ".f32", count);
    const auto gamma   = read_f32(dir + "/gamma_" + tag + ".f32", static_cast<size_t>(n));
    const auto beta    = read_f32(dir + "/beta_"  + tag + ".f32", static_cast<size_t>(n));
    const auto lib_ln  = read_f32(dir + "/ln_"    + tag + ".f32", count);
    const auto lib_rms = read_f32(dir + "/rms_"   + tag + ".f32", count);
    if (in_row.empty() || gamma.empty() || beta.empty() ||
        lib_ln.empty() || lib_rms.empty()) {
        std::printf("  %-12s  *** MISSING DUMPS ***\n", tag.c_str());
        return 2;
    }

    // Transpose the library's row-major input into our column-major layout.
    // This happens here, in the checker, and in neither timed path — so no
    // implementation is made to pay for the other's layout.
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

    // The library accumulates in FP32 like our kernels, so agreement to ~1e-4
    // means "same function"; a semantic mismatch (missing eps, unbiased
    // variance, no gamma) shows up orders larger.
    const double kTol = 1e-4;
    const bool ok_ln = d_ln <= kTol, ok_rms = d_rms <= kTol;
    std::printf("  %-12s  LayerNorm %.3e %-6s   RMSNorm %.3e %-6s\n",
                tag.c_str(), d_ln, ok_ln ? "OK" : "FAIL",
                d_rms, ok_rms ? "OK" : "FAIL");
    return (ok_ln ? 0 : 1) + (ok_rms ? 0 : 1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: verify_baseline <dumpdir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    // Verify EVERY shape the benchmark reported: checking one and publishing
    // three is an inference.  The manifest lists exactly what was measured, so
    // the timed set and the gated set cannot drift apart.
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) {
        std::fprintf(stderr,
            "missing manifest.txt in %s — rerun the benchmark driver, which "
            "writes it.\n", dir.c_str());
        return 2;
    }

    std::printf("Cross-verification against the C++ float64 reference\n");
    std::string line;
    int checks = 0, failures = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') { std::printf("  %s\n", line.c_str()); continue; }
        std::istringstream is(line);
        int64_t m = 0, n = 0; float eps = 1e-5f;
        if (!(is >> m >> n >> eps)) continue;
        failures += verify_shape(dir, m, n, eps);
        checks   += 2;                       // LayerNorm + RMSNorm
    }

    if (checks == 0) {
        std::fprintf(stderr, "manifest listed no shapes\n");
        return 2;
    }

    std::printf("\n%d / %d checks passed.\n", checks - failures, checks);
    if (failures > 0) {
        std::printf("A mismatch invalidates the throughput comparison: the two sides are\n"
                    "not computing the same function, so their speeds are not comparable.\n");
        return 1;
    }
    std::printf("Every benchmarked shape computes the same function as our reference,\n"
                "which is what makes the throughput comparison meaningful.\n");
    return 0;
}
