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
    // volatile: prevents compiler from keeping `best` in a caller-saved register
    // that gets clobbered when fn() enters/exits SME streaming mode.
    volatile double best = 1e18;
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
// SSVE benchmark helpers — all noinline for the same reason as bench_ssve:
// SMSTART/SMSTOP leaves caller-saved FP registers (d0-d7) undefined, so a
// proper call boundary is required to force save/restore.
// ---------------------------------------------------------------------------

__attribute__((noinline))
static double bench_ssve(const float* a, float* b, const float* gamma,
                          int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v1(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v1(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v1(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v2(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v2(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v2(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v3(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v3(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v3(a, b, gamma, m, n, ld, ld, eps); });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Ablation row: prints GiB/s and % delta vs V0 baseline.
static void print_ablation_row(const char* label, int64_t m, int64_t n,
                                double gibs, double peak, double v0_gibs) {
    double delta_pct = (v0_gibs > 0.0) ? (gibs / v0_gibs - 1.0) * 100.0 : 0.0;
    std::cout << std::left  << std::setw(24) << label
              << "  M=" << std::setw(5) << m
              << "  N=" << std::setw(5) << n
              << "  " << std::fixed << std::setprecision(2) << std::setw(7) << gibs
              << " GiB/s"
              << "  (" << std::setprecision(1) << (100.0 * gibs / peak) << "% peak)";
    if (v0_gibs > 0.0) {
        std::cout << "  " << (delta_pct >= 0 ? "+" : "") << std::setprecision(1)
                  << delta_pct << "% vs V0";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== MLC-Norm Sprint 2: reference vs SSVE bandwidth ===\n\n";

    std::cout << "Measuring peak memory bandwidth (STREAM copy)...\n";
    double peak = measure_peak_bandwidth();
    std::cout << std::fixed << std::setprecision(2)
              << "  Peak bandwidth: " << peak << " GiB/s\n\n";

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
        std::cout << "Note: SME not detected — SSVE rows will be skipped.\n\n";

    // -----------------------------------------------------------------------
    // Section 1: reference vs V0 baseline (existing table)
    // -----------------------------------------------------------------------
    std::cout << "--- Reference vs V0 baseline ---\n";
    std::cout << std::left << std::setw(22) << "kernel"
              << "  rows   feat    GiB/s  (% peak)\n";
    std::cout << std::string(62, '-') << "\n";

    for (const auto& s : shapes) {
        const int64_t ld = s.m;
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
            rms_norm_ref(a.data(), b.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);
        });
        print_row("rms_norm_ref  ", s.m, s.n, to_gibs(bytes, rms_sec), peak);

        if (have_sme) {
            double ssve_sec = bench_ssve(a.data(), b.data(), gamma.data(),
                                          s.m, s.n, ld, 1e-5f);
            volatile int64_t vm = s.m, vn = s.n;
            volatile double vbytes = norm_bytes(vm, vn);
            volatile double vgibs  = to_gibs(vbytes, ssve_sec);
            print_row("rms_norm_ssve ", (int64_t)vm, (int64_t)vn, (double)vgibs, peak);
        }
        std::cout << "\n";
    }

    if (!have_sme) return 0;

    // -----------------------------------------------------------------------
    // Section 2: V0–V3 ablation table
    // Three representative shapes: small / medium / large N.
    // -----------------------------------------------------------------------
    std::cout << "\n--- RMSNorm SSVE ablation: V0 → V3 ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% peak)  vs V0\n";
    std::cout << std::string(72, '-') << "\n";

    const Shape ablation_shapes[] = {
        { 128,   64},
        { 128, 2048},
        {1024, 2048},
    };

    for (const auto& s : ablation_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // Declare all timing results volatile so the compiler spills them to
        // the stack immediately.  The kernel saves/restores only D8; SMSTART
        // zeroes D9-D15, so any non-volatile double the compiler keeps in
        // those callee-saved registers across successive bench calls would be
        // silently zeroed (producing inf GiB/s).  volatile forces stack
        // storage, which SMSTART does not touch.
        volatile double vsec_v0 = bench_ssve   (a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_v1 = bench_ssve_v1(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_v2 = bench_ssve_v2(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_v3 = bench_ssve_v3(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);
        volatile double vpeak  = peak;   // protect peak across bench calls

        double g0 = to_gibs((double)vbytes, (double)vsec_v0);
        double g1 = to_gibs((double)vbytes, (double)vsec_v1);
        double g2 = to_gibs((double)vbytes, (double)vsec_v2);
        double g3 = to_gibs((double)vbytes, (double)vsec_v3);

        print_ablation_row("V0 (FSQRT+FDIV)",    vm, vn, g0, (double)vpeak, 0.0);
        print_ablation_row("V1 (FRSQRTE+NR)",    vm, vn, g1, (double)vpeak, g0);
        print_ablation_row("V2 (V1 + inv_N)",     vm, vn, g2, (double)vpeak, g0);
        print_ablation_row("V3 (V2 + unroll-2)",  vm, vn, g3, (double)vpeak, g0);
        std::cout << "\n";
    }

    return 0;
}
