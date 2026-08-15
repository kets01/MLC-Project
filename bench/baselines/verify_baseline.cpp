// Cross-verify each external implementation's output against OUR reference.
//
// Decision B applied to baselines, for a sharper reason than usual: a vendor
// comparison can be wrong not because the timing is wrong but because the two
// sides compute DIFFERENT THINGS.  That is how the earlier comparison failed —
// vDSP_normalize was called "LayerNorm" when it has no eps, no gamma, no beta.
//
// The manifest binds each reported timing to the file it was verified from:
//
//   m n epsilon norm implementation filename
//   1024 2048 1e-05 layer torch_compile ln_torch_compile_1024x2048.f32
//
// One row per (implementation, norm, shape), so torch_eager and torch_compile —
// and ExecuTorch portable and xnnpack — are each checked, rather than one
// standing in for the other.  A row whose file is missing FAILS rather than
// being skipped: a silently skipped check is indistinguishable from a passing
// one in the summary.
//
// Layout bridge: the libraries write ROW-major with the norm over the last
// contiguous axis; our reference is COLUMN-major.  The transpose happens HERE
// and in neither timed path, so neither implementation pays for the other's
// layout.
//
// Build, from the repository root after a normal build:
//   c++ -std=c++17 -O2 -I include bench/baselines/verify_baseline.cpp \
//       build/src/norm/libnorm_lib.a build/src/week6/libweek6_lib.a \
//       -o /tmp/verify_baseline
//   /tmp/verify_baseline <dumpdir>

#include "norm/norm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The libraries accumulate in FP32 like our kernels, so agreement to ~1e-4
// means "same function"; a semantic mismatch (missing eps, unbiased variance,
// no gamma) shows up orders of magnitude larger.
constexpr double kTol = 1e-4;

bool read_f32(const std::string& path, size_t expect, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "  cannot open %s\n", path.c_str());
        return false;
    }
    out.assign(expect, 0.0f);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expect * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != expect * sizeof(float)) {
        std::fprintf(stderr, "  short read on %s (expected %zu floats)\n",
                     path.c_str(), expect);
        return false;
    }
    return true;
}

// Inputs and references for one shape, reused across that shape's rows so the
// float64 reference is not recomputed once per implementation.
struct ShapeCache {
    int64_t m = 0, n = 0;
    float   eps = 0.0f;
    bool    ok = false;
    std::vector<float> gamma, beta, ref_layer, ref_rms;

    bool load(const std::string& dir, int64_t m_, int64_t n_, float eps_) {
        if (ok && m == m_ && n == n_ && eps == eps_) return true;
        m = m_; n = n_; eps = eps_; ok = false;

        const std::string tag = std::to_string(m) + "x" + std::to_string(n);
        const size_t count = static_cast<size_t>(m) * static_cast<size_t>(n);

        std::vector<float> in_row;
        if (!read_f32(dir + "/input_" + tag + ".f32", count, in_row))          return false;
        if (!read_f32(dir + "/gamma_" + tag + ".f32", (size_t)n, gamma))       return false;
        if (!read_f32(dir + "/beta_"  + tag + ".f32", (size_t)n, beta))        return false;

        std::vector<float> in_col(count, 0.0f);
        for (int64_t r = 0; r < m; ++r)
            for (int64_t c = 0; c < n; ++c)
                in_col[static_cast<size_t>(r + c * m)] =
                    in_row[static_cast<size_t>(r * n + c)];

        ref_layer.assign(count, 0.0f);
        ref_rms.assign(count, 0.0f);
        mini_jit::norm::layer_norm_ref(in_col.data(), ref_layer.data(),
                                       gamma.data(), beta.data(),
                                       m, n, m, m, eps);
        mini_jit::norm::rms_norm_ref(in_col.data(), ref_rms.data(),
                                     gamma.data(), m, n, m, m, eps);
        ok = true;
        return true;
    }
};

struct Result {
    bool   finite = true;
    double worst  = 0.0;
};

// Compare a row-major library output against our column-major reference.
//
// Non-finite values are rejected explicitly, and BEFORE the difference is
// taken: every comparison involving a NaN is false, so a running
// `if (d > worst)` maximum would step straight over a NaN and report a small
// error for a broken kernel.
Result compare(const std::vector<float>& got_row,
               const std::vector<float>& ref_col,
               int64_t m, int64_t n) {
    Result res;
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            const float g = got_row[static_cast<size_t>(r * n + c)];
            const float e = ref_col[static_cast<size_t>(r + c * m)];
            if (!std::isfinite(g) || !std::isfinite(e)) {
                res.finite = false;
                return res;
            }
            const double d = std::fabs(static_cast<double>(g) -
                                       static_cast<double>(e));
            if (d > res.worst) res.worst = d;
        }
    }
    return res;
}

void row(const std::string& impl, const std::string& norm,
         const std::string& shape, const char* diff, const char* verdict) {
    std::printf("  %-20s %-6s %-12s %-11s %s\n",
                impl.c_str(), norm.c_str(), shape.c_str(), diff, verdict);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: verify_baseline <dumpdir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) {
        std::fprintf(stderr,
            "missing manifest.txt in %s -- rerun the benchmark driver, which "
            "writes it.\n", dir.c_str());
        return 2;
    }

    std::printf("Cross-verification against the C++ float64 reference\n");

    ShapeCache cache;
    int checks = 0, failures = 0;
    bool header_done = false;
    std::string line;

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') { std::printf("  %s\n", line.c_str()); continue; }

        if (!header_done) {
            std::printf("\n  %-20s %-6s %-12s %-11s %s\n",
                        "implementation", "norm", "shape", "max|diff|", "verdict");
            header_done = true;
        }

        std::istringstream is(line);
        int64_t m = 0, n = 0;
        float   eps = 1e-5f;
        std::string norm, impl, file;
        if (!(is >> m >> n >> eps >> norm >> impl >> file)) {
            std::fprintf(stderr, "  malformed manifest line: %s\n", line.c_str());
            ++checks; ++failures;
            continue;
        }

        const std::string tag = std::to_string(m) + "x" + std::to_string(n);
        ++checks;

        if (!cache.load(dir, m, n, eps)) {
            row(impl, norm, tag, "-", "FAIL (inputs unreadable)");
            ++failures;
            continue;
        }

        const size_t count = static_cast<size_t>(m) * static_cast<size_t>(n);
        std::vector<float> got;
        if (!read_f32(dir + "/" + file, count, got)) {
            row(impl, norm, tag, "-", "FAIL (output unreadable)");
            ++failures;
            continue;
        }

        const std::vector<float>* ref = nullptr;
        if      (norm == "layer") ref = &cache.ref_layer;
        else if (norm == "rms")   ref = &cache.ref_rms;
        else {
            row(impl, norm, tag, "-", "FAIL (unknown norm)");
            ++failures;
            continue;
        }

        const Result r = compare(got, *ref, m, n);
        if (!r.finite) {
            row(impl, norm, tag, "-", "FAIL (non-finite value)");
            ++failures;
            continue;
        }

        char diff[32];
        std::snprintf(diff, sizeof(diff), "%.3e", r.worst);
        const bool pass = r.worst <= kTol;
        row(impl, norm, tag, diff, pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }

    if (checks == 0) {
        std::fprintf(stderr, "manifest listed no implementation rows\n");
        return 2;
    }

    std::printf("\n%d / %d checks passed.\n", checks - failures, checks);
    if (failures > 0) {
        std::printf(
            "A mismatch invalidates the throughput comparison for that row: the\n"
            "two sides are not computing the same function, so their speeds are\n"
            "not comparable.\n");
        return 1;
    }
    std::printf(
        "Every implementation/shape row in the manifest computes the same\n"
        "function as our reference, which is what makes the comparison valid.\n");
    return 0;
}
