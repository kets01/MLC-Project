#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

using namespace mini_jit::norm;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

// Run `fn` `reps` times; return best (minimum) elapsed seconds.
// Best-of-N filters OS jitter while showing the kernel's ceiling.
template<typename Fn>
static double bench(Fn fn, int reps = 50) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }
    return best;
}

static double to_gibs(double bytes, double seconds) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
}

// ---------------------------------------------------------------------------
// STREAM-style peak-bandwidth probe.
// d[i] = s[i] + 1.0f (scale-add) rather than a plain copy so that d != s
// after each pass, preventing macOS VM copy-on-write short-circuiting.
// Counts 1 read + 1 write = 2 * N * sizeof(float) bytes.
//
// __attribute__((noinline)) prevents the compiler from inlining the loop
// into the timing harness and then dead-store-eliminating the stores.
// ---------------------------------------------------------------------------

__attribute__((noinline))
static void bw_scale_add(float* __restrict__ d, const float* __restrict__ s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = s[i] + 1.0f;
}

static double measure_peak_bandwidth() {
    // 128 MiB per array → well beyond any L3 cache on M-series chips
    const size_t N     = 32 * 1024 * 1024; // floats
    const double BYTES = static_cast<double>(N) * sizeof(float) * 2.0;

    std::vector<float> src(N), dst(N);
    for (size_t i = 0; i < N; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
    for (size_t i = 0; i < N; ++i) dst[i] = 0.0f;

    // Warm up: fault all pages so the first timed run doesn't pay page-fault cost.
    bw_scale_add(dst.data(), src.data(), N);

    double best = bench([&]() {
        bw_scale_add(dst.data(), src.data(), N);
    }, 10);

    return to_gibs(BYTES, best);
}

// ---------------------------------------------------------------------------
// Norm bandwidth: bytes = (1 read + 1 write) * M * N * sizeof(float)
// gamma and beta are tiny (N floats) and assumed to stay in L1.
// ---------------------------------------------------------------------------

static double norm_bytes(int64_t m, int64_t n) {
    return static_cast<double>(m) * static_cast<double>(n) * sizeof(float) * 2.0;
}

// ---------------------------------------------------------------------------
// Print a GiB/s row in the same style as the weekly reports
// ---------------------------------------------------------------------------

static void print_row(const char* label, int64_t m, int64_t n,
                      double gibs, double peak) {
    std::cout << std::left  << std::setw(22) << label
              << "  M=" << std::setw(5) << m
              << "  N=" << std::setw(5) << n
              << "  " << std::fixed << std::setprecision(2) << std::setw(7) << gibs
              << " GiB/s"
              << "  (" << std::setprecision(1) << (100.0 * gibs / peak) << "% of peak)\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== MLC-Norm Sprint 2: reference vs SSVE bandwidth ===\n\n";

    // Roofline target
    std::cout << "Measuring peak memory bandwidth (STREAM copy)...\n";
    double peak = measure_peak_bandwidth();
    std::cout << std::fixed << std::setprecision(2)
              << "  Peak bandwidth: " << peak << " GiB/s\n\n";

    // Shapes to benchmark: (M rows, N feature-dim)
    struct Shape { int64_t m, n; };
    const Shape shapes[] = {
        {  128,   64},
        {  128,  512},
        {  128, 2048},
        { 1024,   64},
        { 1024,  512},
        { 1024, 2048},
    };

    const bool have_sme = cpu_supports_sme();
    if (!have_sme)
        std::cout << "Note: SME not detected — rms_norm_ssve rows will be skipped.\n\n";

    std::cout << std::left << std::setw(22) << "kernel"
              << "  rows   feat    GiB/s  (% peak)\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& s : shapes) {
        const int64_t ld = s.m; // packed column-major, no padding
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f), beta(s.n, 0.0f);

        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        double bytes = norm_bytes(s.m, s.n);

        double ln_sec = bench([&]() {
            layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(),
                           s.m, s.n, ld, ld, 1e-5f);
        });
        print_row("layer_norm_ref", s.m, s.n, to_gibs(bytes, ln_sec), peak);

        double rms_sec = bench([&]() {
            rms_norm_ref(a.data(), b.data(), gamma.data(),
                         s.m, s.n, ld, ld, 1e-5f);
        });
        print_row("rms_norm_ref  ", s.m, s.n, to_gibs(bytes, rms_sec), peak);

        if (have_sme) {
            // Warm up: one call outside the timing loop so the SME streaming
            // region is not cold on the first timed iteration.
            rms_norm_ssve(a.data(), b.data(), gamma.data(),
                          s.m, s.n, ld, ld, 1e-5f);
            double ssve_sec = bench([&]() {
                rms_norm_ssve(a.data(), b.data(), gamma.data(),
                              s.m, s.n, ld, ld, 1e-5f);
            });
            print_row("rms_norm_ssve ", s.m, s.n, to_gibs(bytes, ssve_sec), peak);
        }

        std::cout << "\n";
    }

    return 0;
}
