#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "norm/norm.hpp"
#include "week3/utility.hpp"  // cpu_supports_sme()

#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

using namespace mini_jit::norm;

// ---------------------------------------------------------------------------
// P-core scheduling bias (Sprint 2a).
// macOS exposes no thread-to-core pinning API; USER_INTERACTIVE QoS is the
// strongest scheduler hint that a thread belongs on a P-core.  Called for the
// main thread and for every chip-wide probe worker.
// ---------------------------------------------------------------------------

static void request_p_core() {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

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
// STREAM-style peak-bandwidth probes (Sprint 2a: three ceilings).
//
// Byte convention (used for probes AND kernels): USEFUL bytes = 1 read +
// 1 write per element — the algorithm's minimum traffic.  The V0–V3 norm
// kernels as implemented physically move 2R+1W (the reduction pass and the
// normalize pass each read x); their moved-bytes figure is 1.5x the printed
// useful-bytes one.  Keeping one convention for probes and kernels makes the
// % -of-peak honest: a two-pass kernel shows up as a lower %, which is
// exactly the gap residency blocking (V6) attacks.
//
// d[i] = s[i] + 1.0f (scale-add) rather than a plain copy so that d != s
// after each pass, preventing macOS VM copy-on-write short-circuiting.
//
// __attribute__((noinline)) prevents the compiler from inlining the loop
// into the timing harness and then dead-store-eliminating the stores.
// ---------------------------------------------------------------------------

// 128 MiB per array → well beyond any last-level cache on M-series chips.
static const size_t PROBE_N = 32 * 1024 * 1024; // floats

__attribute__((noinline))
static void bw_scale_add(float* __restrict__ d, const float* __restrict__ s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = s[i] + 1.0f;
}

// Ceiling 1: single-core NEON — the C++ loop autovectorizes to NEON
// (ldp q / fadd.4s / stp q; verified in the disassembly).  This is the
// number previously (mis)labeled just "peak": single-threaded, non-streaming.
static double measure_peak_neon_1core() {
    const double BYTES = static_cast<double>(PROBE_N) * sizeof(float) * 2.0;

    std::vector<float> src(PROBE_N), dst(PROBE_N);
    for (size_t i = 0; i < PROBE_N; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
    for (size_t i = 0; i < PROBE_N; ++i) dst[i] = 0.0f;

    // Warm up: fault all pages so the first timed run doesn't pay page-fault cost.
    bw_scale_add(dst.data(), src.data(), PROBE_N);

    double best = bench([&]() {
        bw_scale_add(dst.data(), src.data(), PROBE_N);
    }, 10);

    return to_gibs(BYTES, best);
}

// Ceiling 2: single-core SSVE streaming mode — bw_probe_ssve.S runs the same
// scale-add with contiguous LD1W/ST1W inside one streaming region, i.e. the
// execution mode the norm kernels actually use.  THIS is the kernel roofline.
__attribute__((noinline))
static double bench_probe_ssve(float* d, const float* s, size_t n) {
    bw_probe_ssve(d, s, static_cast<int64_t>(n));  // warm-up pass
    return bench([&]() { bw_probe_ssve(d, s, static_cast<int64_t>(n)); }, 10);
}

static double measure_peak_ssve_1core() {
    // volatile: survives the SMSTART register clobber inside the probe.
    volatile double BYTES = static_cast<double>(PROBE_N) * sizeof(float) * 2.0;

    std::vector<float> src(PROBE_N), dst(PROBE_N);
    for (size_t i = 0; i < PROBE_N; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
    for (size_t i = 0; i < PROBE_N; ++i) dst[i] = 0.0f;

    volatile double best = bench_probe_ssve(dst.data(), src.data(), PROBE_N);
    return to_gibs(BYTES, best);
}

// Ceiling 3: chip-wide — T threads, each scale-adding its own slice R times
// back-to-back (reps inside the threads amortize spawn/join overhead to <1%).
// This is the target for TEIR/OpenMP row-parallel scaling in Sprint 5, NOT
// for the single-threaded kernel.
static double measure_peak_chip(unsigned threads) {
    const int R = 10;
    std::vector<float> src(PROBE_N), dst(PROBE_N);
    for (size_t i = 0; i < PROBE_N; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
    for (size_t i = 0; i < PROBE_N; ++i) dst[i] = 0.0f;  // faults dst pages too

    const size_t slice = PROBE_N / threads;

    auto t0 = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            request_p_core();
            float*       d = dst.data() + t * slice;
            const float* s = src.data() + t * slice;
            size_t n = (t == threads - 1) ? PROBE_N - t * slice : slice;
            for (int r = 0; r < R; ++r) bw_scale_add(d, s, n);
        });
    }
    for (auto& w : workers) w.join();
    auto t1 = Clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bytes   = static_cast<double>(PROBE_N) * sizeof(float) * 2.0 * R;
    return to_gibs(bytes, elapsed);
}

// ---------------------------------------------------------------------------
// Norm bandwidth: bytes = (1 read + 1 write) * M * N * sizeof(float)
// — USEFUL bytes (see convention above).  gamma and beta are tiny (N floats)
// and assumed to stay in L1.
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
              << "  (" << std::setprecision(1) << (100.0 * gibs / peak) << "% of 1-core SSVE peak)\n";
}

// ---------------------------------------------------------------------------
// SSVE benchmark helpers — all noinline for the same reason as bench_ssve:
// SMSTART/SMSTOP leaves caller-saved FP registers (d0-d7) undefined, so a
// proper call boundary is required to force save/restore.
// ---------------------------------------------------------------------------

__attribute__((noinline))
static double bench_ln_ssve(const float* a, float* b, const float* gamma,
                              const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_v1(const float* a, float* b, const float* gamma,
                                const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_v1(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_v1(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_v2(const float* a, float* b, const float* gamma,
                                const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_v2(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_v2(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_v4(const float* a, float* b, const float* gamma,
                                const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_v4(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_v4(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_v5(const float* a, float* b, const float* gamma,
                                const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_v5(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_v5(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_v6(const float* a, float* b, const float* gamma,
                                const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_v6(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ln_ssve_welford(const float* a, float* b, const float* gamma,
                                     const float* beta, int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    layer_norm_ssve_welford(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_ssve_welford(a, b, gamma, beta, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve(const float* a, float* b, const float* gamma,
                          int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v1(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps, int reps = 50) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v1(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v1(a, b, gamma, m, n, ld, ld, eps); }, reps);
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

__attribute__((noinline))
static double bench_ssve_v4(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v4(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v4(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v5(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v5(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v5(a, b, gamma, m, n, ld, ld, eps); });
}

__attribute__((noinline))
static double bench_ssve_v6(const float* a, float* b, const float* gamma,
                             int64_t m, int64_t n, int64_t ld, float eps) {
    using namespace mini_jit::norm;
    rms_norm_ssve_v6(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_ssve_v6(a, b, gamma, m, n, ld, ld, eps); });
}

// ---------------------------------------------------------------------------
// Ablation row: prints GiB/s and % delta vs V0 baseline.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Small-N regime characterization (Sprint 2a): sweep N at fixed M with the
// best kernel (V1) and fit  t(N) = t0 + b*N  by least squares.
//   t0            — fixed per-call cost (streaming entry, prologue, plus
//                   (M/VL) per-row-block setup + inv_rms serialization)
//   8*M / b       — asymptotic useful-byte bandwidth once overhead amortizes
//   N_half = t0/b — the N below which fixed overhead exceeds streaming work
// ---------------------------------------------------------------------------

static void small_n_sweep(int64_t m, double peak_ssve) {
    // volatile copy: `peak_ssve` must survive the SMSTART clobbers below.
    volatile double vpeak = peak_ssve;

    const int64_t ns[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int     K    = static_cast<int>(sizeof(ns) / sizeof(ns[0]));
    std::vector<double> secs(K);   // heap storage survives SMSTART clobbers

    std::cout << "M=" << m << ":\n"
              << std::left << std::setw(8) << "  N"
              << std::right << std::setw(12) << "us/call"
              << std::setw(12) << "GiB/s" << std::setw(22) << "(% 1-core SSVE peak)\n";

    for (int k = 0; k < K; ++k) {
        const int64_t n  = ns[k];
        const int64_t ld = m;
        std::vector<float> a(ld * n), b(ld * n, 0.0f), gamma(n, 1.0f);
        for (int64_t i = 0; i < ld * n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // 200 reps: small shapes need more repetitions for a stable minimum.
        secs[k] = bench_ssve_v1(a.data(), b.data(), gamma.data(), m, n, ld, 1e-5f, 200);

        volatile double vgibs = to_gibs(norm_bytes(m, n), secs[k]);
        std::cout << "  " << std::left << std::setw(6) << n
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(3) << secs[k] * 1e6
                  << std::setw(12) << std::setprecision(2) << vgibs
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * vgibs / vpeak) << " %\n";
    }

    // Least-squares fit t = t0 + b*N (no SME calls past this point).
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < K; ++k) {
        double x = static_cast<double>(ns[k]);
        sx += x; sy += secs[k]; sxx += x * x; sxy += x * secs[k];
    }
    double b_slope = (K * sxy - sx * sy) / (K * sxx - sx * sx);
    double t0      = (sy - b_slope * sx) / K;
    double asym    = to_gibs(8.0 * static_cast<double>(m), b_slope);
    double n_half  = (b_slope > 0.0) ? t0 / b_slope : 0.0;

    // The linear model assumes constant per-element throughput.  A negative
    // intercept means throughput FELL as N grew — the sweep crossed into the
    // true-DRAM regime (total footprint > L2, so reps no longer stay
    // cache-assisted; the Sprint-2b diagnostic showed access density, not
    // pass-2 residency, binds there), and the fit is not a valid overhead
    // estimate.
    if (t0 >= 0.0) {
        std::cout << "  fit t(N) = t0 + b*N:  t0 = " << std::setprecision(3) << t0 * 1e6
                  << " us/call fixed overhead,  asymptotic " << std::setprecision(2) << asym
                  << " GiB/s (" << std::setprecision(1) << (100.0 * asym / vpeak)
                  << "% of 1-core SSVE peak),  overhead = streaming work at N ~ "
                  << std::setprecision(0) << n_half << "\n\n";
    } else {
        std::cout << "  fit t(N) = t0 + b*N: INVALID (t0 < 0) — throughput is not\n"
                  << "  constant in N: the sweep enters the true-DRAM regime (total\n"
                  << "  footprint exceeds L2; see the Sprint-2b density diagnostic).\n"
                  << "  Overhead statement only valid for the M=128 sweep.\n\n";
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    request_p_core();

    std::cout << "=== MLC-Norm Sprint 2/2a: roofline validation + SSVE bandwidth ===\n\n";

    std::cout <<
        "Byte convention: all GiB/s count USEFUL bytes = 1 read + 1 write per\n"
        "element (the algorithm's minimum), for kernels AND probes alike.\n"
        "The V0-V3 kernels as implemented move 2R+1W (both passes read x);\n"
        "their moved-bytes figure is 1.5x the printed one.\n\n";

    const bool have_sme = cpu_supports_sme();
    if (!have_sme)
        std::cout << "Note: SME not detected — SSVE ceiling and kernel rows will be skipped.\n\n";

    const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());

    // All ceilings live in volatile doubles: they are read again after many
    // SMSTART/SMSTOP transitions, which zero callee-saved FP registers
    // (d9-d15) behind the compiler's back.
    std::cout << "Measuring ceilings (128 MiB arrays, best-of-10)...\n";
    volatile double peak_neon = measure_peak_neon_1core();
    volatile double peak_chip = measure_peak_chip(nthreads);
    volatile double peak_ssve = have_sme ? measure_peak_ssve_1core()
                                         : static_cast<double>(peak_neon);

    std::cout << std::fixed << std::setprecision(2)
        << "  single-core NEON  (compiler-vectorized scale-add) : "
        << std::setw(7) << peak_neon << " GiB/s\n"
        << "  single-core SSVE  (streaming LD1W/ST1W probe)      : "
        << std::setw(7) << peak_ssve << " GiB/s  <- kernel roofline\n"
        << "  chip-wide         (" << nthreads << " threads, NEON)              : "
        << std::setw(7) << peak_chip << " GiB/s  <- Sprint-5 threading target\n\n";

    // The single-core SSVE streaming ceiling judges kernel quality: it is the
    // same execution mode, instruction mix, and access pattern as the kernels.
    volatile double vpeak = peak_ssve;

    struct Shape { int64_t m, n; };
    const Shape shapes[] = {
        {  128,   64},
        {  128,  512},
        {  128, 2048},
        { 1024,   64},
        { 1024,  512},
        { 1024, 2048},
    };

    // -----------------------------------------------------------------------
    // Section 1: reference vs V0 baseline (existing table)
    // -----------------------------------------------------------------------
    std::cout << "--- Reference vs V0 baseline ---\n";
    std::cout << std::left << std::setw(22) << "kernel"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)\n";
    std::cout << std::string(70, '-') << "\n";

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
        print_row("layer_norm_ref", s.m, s.n, to_gibs(bytes, ln_sec), vpeak);

        if (have_sme) {
            double lnssve_sec = bench_ln_ssve(a.data(), b.data(), gamma.data(), beta.data(),
                                               s.m, s.n, ld, 1e-5f);
            {
                volatile int64_t vm = s.m, vn = s.n;
                volatile double vbytes = norm_bytes(vm, vn);
                volatile double vgibs  = to_gibs(vbytes, lnssve_sec);
                print_row("layer_norm_ssve", (int64_t)vm, (int64_t)vn, (double)vgibs, vpeak);
            }

            double lnv1_sec = bench_ln_ssve_v1(a.data(), b.data(), gamma.data(), beta.data(),
                                                s.m, s.n, ld, 1e-5f);
            {
                volatile int64_t vm = s.m, vn = s.n;
                volatile double vbytes = norm_bytes(vm, vn);
                volatile double vgibs  = to_gibs(vbytes, lnv1_sec);
                print_row("layer_norm_ssve_v1", (int64_t)vm, (int64_t)vn, (double)vgibs, vpeak);
            }
        }

        double rms_sec = bench([&]() {
            rms_norm_ref(a.data(), b.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);
        });
        print_row("rms_norm_ref  ", s.m, s.n, to_gibs(bytes, rms_sec), vpeak);

        if (have_sme) {
            double ssve_sec = bench_ssve(a.data(), b.data(), gamma.data(),
                                          s.m, s.n, ld, 1e-5f);
            volatile int64_t vm = s.m, vn = s.n;
            volatile double vbytes = norm_bytes(vm, vn);
            volatile double vgibs  = to_gibs(vbytes, ssve_sec);
            print_row("rms_norm_ssve ", (int64_t)vm, (int64_t)vn, (double)vgibs, vpeak);
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
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)  vs V0\n";
    std::cout << std::string(78, '-') << "\n";

    // The fourth shape (64 MB footprint) is deliberately beyond the 16 MB L2:
    // the true-DRAM regime where the access-density lever (V6) has its
    // headroom.  The first three stay cache-assisted across bench reps.
    const Shape ablation_shapes[] = {
        { 128,   64},
        { 128, 2048},
        {1024, 2048},
        {4096, 2048},
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
        volatile double vsec_v4 = bench_ssve_v4(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_v5 = bench_ssve_v5(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_v6 = bench_ssve_v6(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);

        double g0 = to_gibs((double)vbytes, (double)vsec_v0);
        double g1 = to_gibs((double)vbytes, (double)vsec_v1);
        double g2 = to_gibs((double)vbytes, (double)vsec_v2);
        double g3 = to_gibs((double)vbytes, (double)vsec_v3);
        double g4 = to_gibs((double)vbytes, (double)vsec_v4);
        double g5 = to_gibs((double)vbytes, (double)vsec_v5);
        double g6 = to_gibs((double)vbytes, (double)vsec_v6);

        print_ablation_row("V0 (FSQRT+FDIV)",    vm, vn, g0, (double)vpeak, 0.0);
        print_ablation_row("V1 (FRSQRTE+NR)",    vm, vn, g1, (double)vpeak, g0);
        print_ablation_row("V2 (V1 + inv_N)",     vm, vn, g2, (double)vpeak, g0);
        print_ablation_row("V3 (V2 + unroll-2)",  vm, vn, g3, (double)vpeak, g0);
        print_ablation_row("V4 (4-acc ILP)",      vm, vn, g4, (double)vpeak, g0);
        print_ablation_row("V5 (V4 + load pipe)", vm, vn, g5, (double)vpeak, g0);
        print_ablation_row("V6 (4-block contig)", vm, vn, g6, (double)vpeak, g0);
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Section 3: LayerNorm SSVE ablation (Sprint 2c)
    // V1 (FRSQRTE+NR baseline) → V2 (inv_N) → V4 (4-acc ILP) →
    // V5 (load-pipe) → V6 (4-block contig) → Welford (2R+1W online)
    // -----------------------------------------------------------------------
    std::cout << "\n--- LayerNorm SSVE ablation: V0 → V6 + Welford ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)  vs V0\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& s : ablation_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f), beta(s.n, 0.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        volatile double ln_v0  = bench_ln_ssve        (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_v1  = bench_ln_ssve_v1     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_v2  = bench_ln_ssve_v2     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_v4  = bench_ln_ssve_v4     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_v5  = bench_ln_ssve_v5     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_v6  = bench_ln_ssve_v6     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double ln_wf  = bench_ln_ssve_welford(a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);

        double lg0 = to_gibs((double)vbytes, (double)ln_v0);
        double lg1 = to_gibs((double)vbytes, (double)ln_v1);
        double lg2 = to_gibs((double)vbytes, (double)ln_v2);
        double lg4 = to_gibs((double)vbytes, (double)ln_v4);
        double lg5 = to_gibs((double)vbytes, (double)ln_v5);
        double lg6 = to_gibs((double)vbytes, (double)ln_v6);
        double lgw = to_gibs((double)vbytes, (double)ln_wf);

        print_ablation_row("LN V0 (FSQRT+FDIV)",  vm, vn, lg0, (double)vpeak, 0.0);
        print_ablation_row("LN V1 (FRSQRTE+NR)",  vm, vn, lg1, (double)vpeak, lg0);
        print_ablation_row("LN V2 (inv_N)",        vm, vn, lg2, (double)vpeak, lg0);
        print_ablation_row("LN V4 (4-acc ILP)",    vm, vn, lg4, (double)vpeak, lg0);
        print_ablation_row("LN V5 (load-pipe)",    vm, vn, lg5, (double)vpeak, lg0);
        print_ablation_row("LN V6 (4-blk contig)", vm, vn, lg6, (double)vpeak, lg0);
        print_ablation_row("LN Welford (2R+1W)",   vm, vn, lgw, (double)vpeak, lg0);
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Section 4: LayerNorm vs RMSNorm comparison (Sprint 2c decision C)
    // Best-variant side-by-side on identical shapes.
    // LayerNorm structural cost: 3R+1W vs RMSNorm 2R+1W.
    // Expected: LN/RMS GiB/s ratio ≈ 2/3 at the DRAM wall.
    // -----------------------------------------------------------------------
    std::cout << "\n--- LayerNorm vs RMSNorm (best variants, same shapes) ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% peak)  LN/RMS ratio\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& s : ablation_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f), beta(s.n, 0.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        volatile double ln_sec  = bench_ln_ssve_v6(a.data(), b.data(), gamma.data(), beta.data(),
                                                    s.m, s.n, ld, 1e-5f);
        volatile double rms_sec = bench_ssve_v6(a.data(), b.data(), gamma.data(),
                                                 s.m, s.n, ld, 1e-5f);

        // Inner scope forces all intermediate values through volatile stack slots —
        // same discipline as Section 1/2 to survive SMSTART-induced D-register zeroing.
        {
            volatile int64_t vm = s.m, vn = s.n;
            volatile double vbytes = norm_bytes(vm, vn);
            volatile double vlg = to_gibs((double)vbytes, (double)ln_sec);
            volatile double vrg = to_gibs((double)vbytes, (double)rms_sec);
            double ratio = ((double)vrg > 0.0) ? (double)vlg / (double)vrg : 0.0;

            std::cout << std::left  << std::setw(24) << "LN V6 (best)"
                      << "  M=" << std::setw(5) << (int64_t)vm
                      << "  N=" << std::setw(5) << (int64_t)vn
                      << "  " << std::fixed << std::setprecision(2) << std::setw(7)
                      << (double)vlg << " GiB/s"
                      << "  (" << std::setprecision(1)
                      << (100.0 * (double)vlg / (double)vpeak) << "% peak)\n";

            std::cout << std::left  << std::setw(24) << "RMS V6 (best)"
                      << "  M=" << std::setw(5) << (int64_t)vm
                      << "  N=" << std::setw(5) << (int64_t)vn
                      << "  " << std::fixed << std::setprecision(2) << std::setw(7)
                      << (double)vrg << " GiB/s"
                      << "  (" << std::setprecision(1)
                      << (100.0 * (double)vrg / (double)vpeak) << "% peak)"
                      << "  LN/RMS = " << std::setprecision(2) << ratio << "\n\n";
        }
    }

    // -----------------------------------------------------------------------
    // Section 5 (Sprint 2a): small-N regime — fixed per-call overhead.
    // -----------------------------------------------------------------------
    std::cout << "\n--- Small-N regime: fixed overhead vs streaming work (V1 kernel) ---\n";
    small_n_sweep( 128, vpeak);
    small_n_sweep(1024, vpeak);

    return 0;
}
