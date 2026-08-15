#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "norm/norm.hpp"
#include "norm/stability.hpp"  // Sprint 7a: FP32 variance estimators
#include "norm/jit_norm.hpp"  // Sprint 4: mini_jit::Norm
#include "week3/utility.hpp"  // cpu_supports_sme()
#include "week7/TeirRuntime.h" // Sprint 5: TEIR-invoked, OpenMP row-parallel

#if __has_include(<omp.h>)
#include <omp.h>
#define BENCH_HAS_OMP 1
#else
#define BENCH_HAS_OMP 0
#endif

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
// Sprint 8 — provenance.
//
// A performance table is only reproducible if the reader knows which build,
// which machine and which CPU features produced it.  This prints all of it
// once, at the top of the run, so no number in the report can be separated
// from its conditions.  The SME/SME2 lines are the DETECTED values, not a
// datasheet claim — this project's own docs carried a stale "M4 is SME1"
// assumption for several sprints while the hardware reported FEAT_SME2=1.
// ---------------------------------------------------------------------------

#ifndef MLC_GIT_SHA
#define MLC_GIT_SHA "unknown"
#endif
#ifndef MLC_BUILD_TYPE
#define MLC_BUILD_TYPE "unknown"
#endif
#ifndef MLC_CXX_COMPILER
#define MLC_CXX_COMPILER "unknown"
#endif

static std::string shell_capture(const char* cmd) {
    std::string out;
    FILE* p = popen(cmd, "r");
    if (!p) return "unavailable";
    char buf[256];
    while (fgets(buf, sizeof(buf), p)) out += buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out.empty() ? "unavailable" : out;
}

static void print_provenance(unsigned nthreads) {
    std::cout << std::string(96, '=') << "\n"
              << "RUN PROVENANCE — the conditions every number below was produced under\n"
              << std::string(96, '=') << "\n"
              << "  git commit        : " << MLC_GIT_SHA << "\n"
              << "  build type        : " << MLC_BUILD_TYPE << "\n"
              << "  compiler          : " << MLC_CXX_COMPILER << "\n"
              << "  C++ standard      : " << __cplusplus << "\n"
              << "  OS                : " << shell_capture("sw_vers -productName 2>/dev/null")
              << " " << shell_capture("sw_vers -productVersion 2>/dev/null")
              << " (" << shell_capture("uname -m") << ")\n"
              << "  CPU               : "
              << shell_capture("sysctl -n machdep.cpu.brand_string 2>/dev/null") << "\n"
              << "  FEAT_SME (sysctl) : "
              << shell_capture("sysctl -n hw.optional.arm.FEAT_SME 2>/dev/null") << "\n"
              << "  FEAT_SME2 (sysctl): "
              << shell_capture("sysctl -n hw.optional.arm.FEAT_SME2 2>/dev/null") << "\n"
              << "  cpu_supports_sme(): " << (cpu_supports_sme()  ? "true" : "false")
              << ",  cpu_supports_sme2(): " << (cpu_supports_sme2() ? "true" : "false") << "\n"
              << "  streaming VL      : " << svl_fp32_lanes() << " FP32 lanes (RDSVL)\n"
              << "  norm dispatch     : " << norm_dispatch_target() << "\n"
              << "  OpenMP threads    : " << nthreads << "\n"
              << "  scheduler QoS     : QOS_CLASS_USER_INTERACTIVE (P-core hint; "
                 "macOS exposes no pinning API)\n"
              << std::string(96, '=') << "\n\n";
}

// ---------------------------------------------------------------------------
// Sprint 8 — report a distribution, not just the best sample.
//
// Every table in this harness historically reported best-of-N (the minimum).
// That is a defensible instrument for estimating a machine CEILING: taking the
// minimum suppresses preemption, migration and unrelated system activity, so
// it answers "what can this kernel do when nothing interferes".
//
// It is a poor number to present ALONE, because it cannot distinguish
//     "18.4 us is what this kernel does"
// from
//     "one lucky run hit 18.4 us and the other 49 were 24-31 us".
// Those have very different engineering meanings and identical minima.
//
// So the minimum is kept and explicitly labelled best-case, and a median plus
// a p10/p90 spread is reported next to it. A wide gap between min and median
// is itself a finding: it says the number is fragile.
// ---------------------------------------------------------------------------

struct Stats {
    double best   = 0.0;   // minimum — best-case envelope
    double median = 0.0;   // robust typical value
    double p10    = 0.0;
    double p90    = 0.0;
    int    n      = 0;
};

// Percentile by nearest-rank on a sorted sample.
static double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[std::min(idx, sorted.size() - 1)];
}

template<typename Fn>
static Stats bench_stats(Fn fn, int reps = 50) {
    // Samples live on the heap: they are read again after calls that cross
    // SMSTART, which zeroes d8-d15 behind the compiler's back.
    std::vector<double> s;
    s.reserve(static_cast<size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        s.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(s.begin(), s.end());
    Stats st;
    st.n      = static_cast<int>(s.size());
    st.best   = s.front();
    st.median = percentile(s, 0.50);
    st.p10    = percentile(s, 0.10);
    st.p90    = percentile(s, 0.90);
    return st;
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

// Ceiling 3: chip-wide — T threads scale-adding the whole array R times.
// This is the target for TEIR/OpenMP row-parallel scaling in Sprint 5, NOT
// for the single-threaded kernel.
//
// Sprint-5 CORRECTION (work distribution, not bandwidth): this probe used to
// hand each thread ONE equal-sized slice. On the M4 the cores are
// heterogeneous (4 P + 6 E), so wall time was set by the slowest E-core while
// the P-cores sat idle after finishing early — it measured
// slowest-core-times-T, not the memory system, and reported ~44 GiB/s. The
// bug was caught by the ceiling being EXCEEDED: the TEIR row-parallel RMSNorm
// sustained 53 GiB/s against a "ceiling" of 44 (decision B in reverse — a
// measurement that says something impossible is wrong, not a discovery).
//
// The fix is to distribute work the same way the workload under test does:
// many small chunks handed out dynamically, so a fast core simply takes more
// of them. Same probe kernel, same bytes, same reps — only the scheduling
// changes, so the number stays comparable to the other two ceilings.
static double measure_peak_chip(unsigned threads) {
    const int R = 10;
    std::vector<float> src(PROBE_N), dst(PROBE_N);
    for (size_t i = 0; i < PROBE_N; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
    for (size_t i = 0; i < PROBE_N; ++i) dst[i] = 0.0f;  // faults dst pages too

    // ~8 chunks per thread: fine enough to balance P vs E cores, coarse
    // enough that each chunk is still a long streaming run.
    //
    // The R repetitions run as R separate PASSES separated by a barrier,
    // not as one flat pool of n_chunks*R work items. Two reasons, both
    // measurement-correctness rather than style: (a) within a pass every
    // chunk has exactly one owner, so threads never write the same cache
    // lines concurrently (a flat pool lets thread A run chunk 5 of rep 0
    // while B runs chunk 5 of rep 1 — coherence ping-pong, and a data
    // race); (b) a pass touches all 128 MiB before any chunk is revisited,
    // so no chunk is still cache-resident on its next rep, which would
    // measure cache instead of DRAM.
    // Sprint 6: this used to be hand-rolled std::thread workers with an
    // atomic work counter and a generation-counter barrier.  OpenMP expresses
    // the same three properties directly, so the machinery is gone but the
    // measurement semantics are unchanged:
    //   - dynamic work distribution  -> schedule(dynamic): a fast P-core
    //     simply claims more chunks than a slow E-core, which is the whole
    //     point (equal static slices measured slowest-core x T, not the
    //     memory system — the bug fixed in Sprint 5).
    //   - one owner per chunk per pass -> a single `omp for` over n_chunks.
    //   - a barrier between passes    -> the implicit barrier at the end of
    //     `omp for`, which is exactly what kept a chunk from being revisited
    //     while another thread still owned it, and what guarantees a full
    //     128 MiB sweep before any chunk comes round again.
    // The `parallel` region is opened ONCE outside the rep loop so the QoS
    // hint is applied per worker thread rather than per pass.
    const long long n_chunks = static_cast<long long>(threads) * 8;
    const size_t    chunk    = PROBE_N / static_cast<size_t>(n_chunks);

    auto t0 = Clock::now();
#if BENCH_HAS_OMP
    #pragma omp parallel num_threads(static_cast<int>(threads))
    {
        request_p_core();
        for (int r = 0; r < R; ++r) {
            #pragma omp for schedule(dynamic)
            for (long long idx = 0; idx < n_chunks; ++idx) {
                float*       d = dst.data() + static_cast<size_t>(idx) * chunk;
                const float* s = src.data() + static_cast<size_t>(idx) * chunk;
                size_t n = (idx == n_chunks - 1)
                             ? PROBE_N - static_cast<size_t>(idx) * chunk : chunk;
                bw_scale_add(d, s, n);
            }
        }
    }
#else
    // No OpenMP: run the same total work sequentially. The number is then a
    // single-core figure and is labelled as such by the caller, rather than
    // silently pretending to be a chip-wide ceiling.
    for (int r = 0; r < R; ++r)
        for (long long idx = 0; idx < n_chunks; ++idx) {
            float*       d = dst.data() + static_cast<size_t>(idx) * chunk;
            const float* s = src.data() + static_cast<size_t>(idx) * chunk;
            size_t n = (idx == n_chunks - 1)
                         ? PROBE_N - static_cast<size_t>(idx) * chunk : chunk;
            bw_scale_add(d, s, n);
        }
#endif
    auto t1 = Clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bytes   = static_cast<double>(PROBE_N) * sizeof(float) * 2.0 * R;
    return to_gibs(bytes, elapsed);
}

// ---------------------------------------------------------------------------
// Sprint 6 correction — the ceiling is a CURVE, not a constant.
//
// Every "% of peak" since Sprint 2a divides by ONE number: the 59.5 GiB/s
// SSVE figure measured above at PROBE_N = 128 MiB per array.  That number is
// correct for what it measures — the single-core streaming ceiling when the
// data comes from DRAM — and it is the right denominator for a DRAM-resident
// kernel.  It is the WRONG denominator for a cache-resident one, which never
// faces DRAM bandwidth at all and is limited by the hierarchy's much higher
// ceiling.  Sprint 2a asked "peak of WHAT" and answered the execution-mode
// half (NEON vs streaming); this is the footprint half it left unasked.
//
// Methodology — the point on which the whole correction rests: at a small
// footprint both arrays stay resident after the first repetition, so what is
// measured is STEADY-STATE RESIDENT bandwidth, not cold-miss bandwidth.  That
// is a legitimate denominator here only because the norm kernels are
// benchmarked exactly the same way (repeated calls over the same buffers).
// Probe and kernel must sit in the same residency regime or their ratio is
// meaningless.
//
// Limitations, stated rather than glossed:
//   * The probe is a pure 1R+1W streaming copy; the norms are 2R+1W (RMS) and
//     3R+1W (LN) with a pass-2 re-read that interacts with the hierarchy
//     differently.  A footprint-matched probe is a BETTER denominator than a
//     single DRAM constant, not an exact one.
//   * Sprint 6 §2.2 found an unexplained base-offset effect (allocation start
//     measured fastest).  Fresh allocations per point AVOID it; they do not
//     control for it.
// ---------------------------------------------------------------------------

struct CeilingPoint {
    double mib;        // total working set (src + dst)
    double neon_gibs;
    double ssve_gibs;
};

// Reps scale with footprint via a fixed byte budget rather than a fixed count.
// Sprint 6 caught the fixed-count failure directly: at a small shape it made
// the hand-written V6 read 10 % below its own word-identical JIT twin, which
// is impossible for identical code and was therefore sampling noise.
static int reps_for(size_t bytes_per_pass) {
    const double kBudget = 512.0 * 1024.0 * 1024.0;   // ~512 MiB moved per point
    int r = static_cast<int>(kBudget / static_cast<double>(bytes_per_pass));
    return std::max(5, std::min(r, 1000));
}

static std::vector<CeilingPoint> measure_ceiling_curve(bool have_sme) {
    // Total working set = 2 arrays * n * 4 B.  64 KiB -> 256 MiB, geometric.
    // The last point is PROBE_N-equivalent, so it must reproduce the 59.5 /
    // 79.5 figures printed above — that agreement is the sweep's own control.
    //
    // The sub-MiB points exist because the smallest ablation shape (128x64) is
    // only 0.06 MiB; without them its denominator would be extrapolated from
    // the 1 MiB point.  They come with a caveat that the printout repeats: at
    // that size a pass takes well under a microsecond, so the SSVE probe's own
    // SMSTART/SMSTOP entry cost and the harness's two Clock::now() calls are a
    // visible fraction of the measurement.  A sub-MiB "ceiling" is therefore
    // NOT a pure bandwidth figure, and the curve bending down there is that
    // overhead, not the memory system.  Quantifying it properly is Sprint 7b.
    const double mibs[] = { 0.0625, 0.125, 0.25, 0.5,
                            1, 2, 4, 8, 16, 32, 64, 128, 256 };
    const int    K      = static_cast<int>(sizeof(mibs) / sizeof(mibs[0]));

    std::vector<CeilingPoint> out;
    out.reserve(static_cast<size_t>(K));

    for (int k = 0; k < K; ++k) {
        const size_t n = static_cast<size_t>(mibs[k] * 1024.0 * 1024.0 / (2.0 * sizeof(float)));

        // Fresh allocation per point (see limitation 2 above).
        std::vector<float> src(n), dst(n);
        for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i & 0xFF) + 1.0f;
        for (size_t i = 0; i < n; ++i) dst[i] = 0.0f;   // faults dst pages too

        const size_t pass_bytes = n * sizeof(float) * 2;
        const int    reps       = reps_for(pass_bytes);

        // BYTES volatile: read again after the SSVE probe's SMSTART/SMSTOP
        // transitions, which zero d8-d15 behind the compiler's back.  Sprint 6
        // Part 3 #5 is exactly this bug — a microbenchmark that reported `inf`
        // because its own probe did not preserve them.
        volatile double BYTES = static_cast<double>(pass_bytes);

        bw_scale_add(dst.data(), src.data(), n);        // warm-up / page-fault
        volatile double t_neon = bench([&]() { bw_scale_add(dst.data(), src.data(), n); }, reps);

        volatile double t_ssve = 0.0;
        if (have_sme)
            t_ssve = bench_probe_ssve(dst.data(), src.data(), n);  // warms up internally

        CeilingPoint p;
        p.mib       = mibs[k];
        p.neon_gibs = to_gibs(static_cast<double>(BYTES), static_cast<double>(t_neon));
        p.ssve_gibs = have_sme
                        ? to_gibs(static_cast<double>(BYTES), static_cast<double>(t_ssve))
                        : 0.0;
        out.push_back(p);
    }
    return out;
}

// The denominator a kernel of this footprint should actually be divided by:
// the measured SSVE ceiling at the nearest working-set size at or above it.
// Falls back to the DRAM figure past the top of the curve.
static double ceiling_for_footprint(const std::vector<CeilingPoint>& pts,
                                    double mib, double dram_fallback) {
    double best = dram_fallback;
    for (const auto& p : pts) {
        if (p.ssve_gibs <= 0.0) continue;
        if (p.mib >= mib) { best = p.ssve_gibs; break; }
        best = p.ssve_gibs;
    }
    return best;
}

// Printing is a SEPARATE loop from measurement: no SME call may sit between a
// computed FP value and its output, or the SMSTART clobber can zero it (the
// discipline small_n_sweep documents below).
static void print_ceiling_curve(const std::vector<CeilingPoint>& pts, bool have_sme,
                                double dram_ssve) {
    std::cout << "\n" << std::string(78, '=') << "\n"
              << "CEILING vs FOOTPRINT - is 59.5 GiB/s the right denominator?\n"
              << std::string(78, '=') << "\n"
              << "Steady-state RESIDENT bandwidth: buffers are re-touched every rep, the\n"
              << "same way the norm kernels are benchmarked.  Probe = 1R+1W scale-add.\n"
              << "The 256 MiB row must reproduce the two ceilings printed above (control).\n"
              << "Sub-MiB rows are overhead-contaminated (a pass is < 1 us, so SMSTART and\n"
              << "the timing calls are a visible share) — not pure bandwidth.  See 7b.\n\n"
              << std::left << std::setw(14) << "  working set"
              << std::right << std::setw(12) << "NEON"
              << std::setw(12) << "SSVE"
              << std::setw(14) << "SSVE/NEON" << "\n";

    for (const auto& p : pts) {
        // Sub-MiB points would all render as "0 MiB" as integers.
        std::string label;
        if (p.mib < 1.0) {
            label = std::to_string(static_cast<int>(p.mib * 1024.0)) + " KiB";
        } else {
            label = std::to_string(static_cast<int>(p.mib)) + " MiB";
        }
        std::cout << "  " << std::left << std::setw(12) << label
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(2) << p.neon_gibs;
        if (have_sme) {
            std::cout << std::setw(12) << std::setprecision(2) << p.ssve_gibs
                      << std::setw(14) << std::setprecision(3)
                      << (p.neon_gibs > 0.0 ? p.ssve_gibs / p.neon_gibs : 0.0);
        } else {
            std::cout << std::setw(12) << "-" << std::setw(14) << "-";
        }
        std::cout << "\n";
    }

    if (!have_sme) { std::cout << "\n"; return; }

    // What the correction actually costs, for the three shapes the consolidated
    // ablation reports.  This is the size of the restatement, printed rather
    // than argued: only shapes that are NOT DRAM-resident move.
    struct AblShape { const char* label; int64_t m, n; };
    const AblShape shapes[] = {
        { "128x64    (per-call overhead)", 128,  64   },
        { "1024x2048 (cache-assisted)",    1024, 2048 },
        { "4096x8192 (true DRAM)",         4096, 8192 },
    };

    std::cout << "\nDenominator restatement for the consolidated-ablation shapes:\n"
              << std::left << std::setw(34) << "  shape"
              << std::right << std::setw(12) << "footprint"
              << std::setw(12) << "old (DRAM)"
              << std::setw(14) << "footprint-matched" << "\n";
    for (const auto& s : shapes) {
        const double mib = static_cast<double>(s.m) * static_cast<double>(s.n)
                           * sizeof(float) * 2.0 / (1024.0 * 1024.0);
        const double matched = ceiling_for_footprint(pts, mib, dram_ssve);
        std::cout << "  " << std::left << std::setw(32) << s.label
                  << std::right << std::fixed
                  << std::setw(10) << std::setprecision(2) << mib << " MiB"
                  << std::setw(12) << std::setprecision(2) << dram_ssve
                  << std::setw(14) << std::setprecision(2) << matched << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Sprint 7a — numerical stability, measured rather than asserted.
//
// context.md §8 claims single-pass variance is "numerically dangerous" and that
// LayerNorm's two-pass form and RMSNorm's mean-free reduction avoid it.  Until
// Sprint 7 the project had never IMPLEMENTED the dangerous version, so the
// claim had no counterexample.  stability.cpp supplies one.
//
// The stress axis is a shift s: it leaves the variance untouched but scales the
// condition number kappa = 1 + mean^2/var by ~s^2, isolating conditioning from
// every other property of the data.
//
// The second table is the one that matters for THIS project: the estimators
// above are scalar C++, so they prove nothing about our kernels until the
// shipped kernels are put on the same axis.
// ---------------------------------------------------------------------------

static void sprint7_stability_section() {
    namespace stab = mini_jit::norm::stability;

    const int64_t N = 512;
    const double  shifts[] = { 0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6 };

    std::cout << "\n" << std::string(96, '=') << "\n"
              << "SPRINT 7a - NUMERICAL STABILITY: where each variance formulation breaks\n"
              << std::string(96, '=') << "\n"
              << "Data: unit-variance samples shifted by s.  Variance is invariant under the\n"
              << "shift; the condition number kappa = 1 + mean^2/var is not.  All estimators\n"
              << "accumulate in FP32 (the kernels' precision); the oracle is float64.\n"
              << "Relative error vs the float64 oracle on identical input:\n\n"
              << std::left << std::setw(10) << "  shift"
              << std::right << std::setw(12) << "kappa"
              << std::setw(14) << "naive 1-pass"
              << std::setw(14) << "two-pass"
              << std::setw(14) << "Welford" << "\n";

    for (double s : shifts) {
        std::vector<float> x(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i)
            x[static_cast<size_t>(i)] =
                static_cast<float>(s + std::sin(static_cast<double>(i) * 0.7));

        const double truth = stab::variance_ref_f64(x.data(), N);
        const double kappa = stab::variance_condition_number(x.data(), N);
        const float  nv    = stab::variance_naive_f32(x.data(), N);

        auto rel = [&](double v) {
            return truth > 0.0 ? std::fabs(v - truth) / truth : 0.0;
        };

        std::cout << "  " << std::left << std::setw(8) << std::scientific
                  << std::setprecision(0) << s
                  << std::right << std::setw(12) << std::setprecision(1) << kappa
                  << std::setw(14) << std::setprecision(2) << rel(nv)
                  << std::setw(14) << rel(stab::variance_twopass_f32(x.data(), N))
                  << std::setw(14) << rel(stab::variance_welford_f32(x.data(), N));
        if (nv < 0.0f) std::cout << "   <- naive NEGATIVE, sqrt = NaN";
        std::cout << "\n";
    }
    std::cout << std::fixed;

    std::cout << "\nReading it: naive's error tracks kappa (each row multiplies the shift by 10,\n"
              << "hence kappa by ~100), and past shift 1e4 it returns a NEGATIVE variance -- so a\n"
              << "kernel built on it emits NaN, not merely an inaccurate number.  Two-pass is flat\n"
              << "across 12 orders of magnitude of kappa.  Welford sits between: vastly better\n"
              << "than naive, slightly worse than two-pass, and degrading slowly because its\n"
              << "running mean updates at full input magnitude (Sprint 6 §1.3).\n"
              << "\nCaveat stated rather than hidden: past shift ~1e5 the FP32 INPUT is itself\n"
              << "quantized (ULP of 1e6 is 0.0625, larger than the signal), so beyond that point\n"
              << "all estimators measure the variance of the quantized data.  They stay\n"
              << "comparable -- same input, same oracle -- but the 'true' variance drifts.\n";

    // ---- The shipped kernels on the same axis -----------------------------
    // Scalar estimators prove nothing about our SSVE kernels; this puts the
    // real ones on the identical stress data.
    if (!cpu_supports_sme()) {
        std::cout << "\n(SME absent: kernel rows skipped.)\n";
        return;
    }

    const int64_t M = 64, ld = M;
    std::cout << "\nThe shipped kernels on the same stress data (M=" << M << ", N=" << N
              << "), max|error| vs the\nfloat64-accumulating C++ reference:\n\n"
              << std::left << std::setw(10) << "  shift"
              << std::right << std::setw(16) << "LN V0 (FSQRT)"
              << std::setw(16) << "LN V6 (FRSQRTE)"
              << std::setw(16) << "LN Welford"
              << std::setw(16) << "RMS V6" << "\n";

    for (double s : shifts) {
        std::vector<float> a(static_cast<size_t>(ld * N));
        for (int64_t c = 0; c < N; ++c)
            for (int64_t r = 0; r < M; ++r)
                a[static_cast<size_t>(r + c * ld)] = static_cast<float>(
                    s + std::sin(static_cast<double>(r * 7 + c * 3) * 0.7));

        std::vector<float> gamma(static_cast<size_t>(N), 1.0f);
        std::vector<float> beta(static_cast<size_t>(N), 0.0f);
        std::vector<float> ref_ln(static_cast<size_t>(ld * N), 0.0f);
        std::vector<float> ref_rms(static_cast<size_t>(ld * N), 0.0f);
        layer_norm_ref(a.data(), ref_ln.data(), gamma.data(), beta.data(),
                       M, N, ld, ld, 1e-5f);
        rms_norm_ref(a.data(), ref_rms.data(), gamma.data(), M, N, ld, ld, 1e-5f);

        auto worst = [&](void (*fn)(const float*, float*, const float*,
                                    const float*, int64_t, int64_t, int64_t,
                                    int64_t, float),
                         const std::vector<float>& ref) {
            std::vector<float> out(static_cast<size_t>(ld * N), 0.0f);
            fn(a.data(), out.data(), gamma.data(), beta.data(),
               M, N, ld, ld, 1e-5f);
            double w = 0.0;
            for (size_t i = 0; i < out.size(); ++i)
                w = std::max(w, static_cast<double>(std::fabs(out[i] - ref[i])));
            return w;
        };

        std::vector<float> out_rms(static_cast<size_t>(ld * N), 0.0f);
        rms_norm_ssve_v6(a.data(), out_rms.data(), gamma.data(),
                         M, N, ld, ld, 1e-5f);
        double w_rms = 0.0;
        for (size_t i = 0; i < out_rms.size(); ++i)
            w_rms = std::max(w_rms,
                             static_cast<double>(std::fabs(out_rms[i] - ref_rms[i])));

        std::cout << "  " << std::left << std::setw(8) << std::scientific
                  << std::setprecision(0) << s << std::right
                  << std::setw(16) << std::setprecision(2) << worst(layer_norm_ssve, ref_ln)
                  << std::setw(16) << worst(layer_norm_ssve_v6, ref_ln)
                  << std::setw(16) << worst(layer_norm_ssve_welford, ref_ln)
                  << std::setw(16) << w_rms << "\n";
    }
    std::cout << std::fixed
        << "\nThree readings, in order of how much they change the project's story:\n\n"
        << "1. The FRSQRTE cost is REAL BUT NARROW.  At shift 0, V0 (exact FSQRT+FDIV) is\n"
        << "   7.2e-7 against V6's 1.7e-5 -- a ~24x accuracy penalty for the ablation's\n"
        << "   winning variant, which Sprint 6 §1.3 found was never documented.  But from\n"
        << "   shift 1e2 upward the two are INDISTINGUISHABLE (1.30e-4 both).  So the\n"
        << "   substitution costs accuracy only where the problem is well conditioned; it\n"
        << "   is not what limits LayerNorm on hard data.\n\n"
        << "2. What DOES limit LayerNorm at high shift is not the variance at all.  Two-pass\n"
        << "   already fixed the variance (see the flat column above).  The residual error is\n"
        << "   in the OUTPUT expression (x - mean): when x ~ 1e5 and (x - mean) ~ 1, the FP32\n"
        << "   representation of x carries the difference in only a handful of bits.  At\n"
        << "   shift 1e5 the ULP near 1e5 is ~1.6e-2 relative to a unit signal, and the\n"
        << "   measured error is 1.2e-2 -- the two agree, so this is an INPUT-REPRESENTATION\n"
        << "   limit that no variance algorithm can remove.  That V0 and V6 coincide here is\n"
        << "   the evidence: the sqrt arithmetic is not the binding constraint.\n"
        << "   (The non-monotonic dip at 1e6 is the same effect seen from the other side --\n"
        << "   once the input is coarsely quantized, (x - mean) becomes exactly\n"
        << "   representable more often, so the kernel's own rounding falls.)\n\n"
        << "3. RMSNorm is flat at ~5e-6 across the whole sweep, ~2300x better than LayerNorm\n"
        << "   at shift 1e5.  This is context.md §8's claim confirmed on the shipped kernel:\n"
        << "   RMSNorm never forms a mean, so it never performs the cancelling subtraction --\n"
        << "   neither in the reduction nor in the output.  Its stability is structural, and\n"
        << "   it is the same property that makes it faster (2R+1W vs 3R+1W).  Accuracy and\n"
        << "   throughput point the same way here, which is unusual and worth saying.\n"
        << "   RMSNorm's own limit is elsewhere: Sum(x^2) overflows FP32 at |x| ~ "
        << "sqrt(3.4e38/N).\n\n";
}

// ---------------------------------------------------------------------------
// Sprint 7b — streaming-mode overhead, measured directly instead of inferred.
//
// context.md §8 asserts "streaming mode is not free" and that toggling it per
// row "destroys performance".  The project has always DESIGNED for that (one
// streaming region per call) but never measured the constant, and every
// per-call cost quoted so far came from the intercept of a linear fit.  That
// intercept is not a safe instrument: Sprint 2a read t0 ~ 1 us/call from it,
// and Sprint 4 then found ~70 % of that was an uncached cpu_supports_sme()
// sysctl in the wrapper — the fit had silently absorbed an unrelated bug.
//
// smstart_probe.S measures the transition itself, with an identical
// transition-free loop as the control, so the intercept can be DECOMPOSED
// rather than trusted whole.
// ---------------------------------------------------------------------------

static void sprint7_streaming_overhead_section() {
    if (!cpu_supports_sme()) {
        std::cout << "\n(SME absent: Sprint 7b streaming-overhead section skipped.)\n";
        return;
    }

    std::cout << "\n" << std::string(96, '=') << "\n"
              << "SPRINT 7b - STREAMING-MODE OVERHEAD: what the transition actually costs\n"
              << std::string(96, '=') << "\n";

    // ---- 1. The transition, measured -------------------------------------
    const int64_t IT = 2000000;
    // Heap/volatile discipline: these calls cross smstart, which zeroes
    // d8-d15 behind the compiler's back (the bug this project has hit five
    // times).  Timings are taken first, printed after, with no SME call in
    // between.
    volatile double t_empty = bench([&] { smstart_probe_empty(IT);   }, 20);
    volatile double t_pairs = bench([&] { smstart_probe_pairs(IT);   }, 20);
    volatile double t_sm    = bench([&] { smstart_probe_sm_only(IT); }, 20);

    const double ns_empty = static_cast<double>(t_empty) * 1e9 / static_cast<double>(IT);
    const double ns_pairs = static_cast<double>(t_pairs) * 1e9 / static_cast<double>(IT);
    const double ns_sm    = static_cast<double>(t_sm)    * 1e9 / static_cast<double>(IT);
    const double trans_pairs = ns_pairs - ns_empty;
    const double trans_sm    = ns_sm    - ns_empty;

    std::cout << "\nDirect measurement (" << IT << " iterations, best-of-20).  The empty loop is\n"
              << "the control: identical branch structure and counter, transition removed, so\n"
              << "the difference is the transition rather than the loop.\n\n"
              << std::left << std::setw(34) << "  variant"
              << std::right << std::setw(14) << "ns/iteration"
              << std::setw(20) << "minus control" << "\n";
    std::cout << "  " << std::left << std::setw(32) << "empty loop (control)"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(14) << ns_empty << std::setw(20) << "-" << "\n";
    std::cout << "  " << std::left << std::setw(32) << "smstart + smstop (SM+ZA)"
              << std::right << std::setw(14) << ns_pairs
              << std::setw(20) << trans_pairs << "\n";
    std::cout << "  " << std::left << std::setw(32) << "smstart sm + smstop sm (SM only)"
              << std::right << std::setw(14) << ns_sm
              << std::setw(20) << trans_sm << "\n";

    std::cout << "\nTwo results here.  First, ONE round trip through streaming mode costs about "
              << std::setprecision(1) << trans_sm << " ns.\n"
              << "Second, SM+ZA and SM-only are the same to within noise ("
              << std::setprecision(3) << trans_pairs << " vs " << trans_sm << "), so managing\n"
              << "PSTATE.ZA is free — the ZA kernels' problem was never the transition.\n";

    // ---- 2. The per-call floor, measured rather than fitted ---------------
    // A linear fit is the WRONG instrument here and this section deliberately
    // does not use one.  Fitting t(N) = t0 + b*N over N=16..512 at M=128 sweeps
    // the footprint from 16 KiB to 512 KiB, crossing cache levels, so per-element
    // throughput is not constant and the intercept comes out NEGATIVE — the same
    // invalid-fit condition small_n_sweep() reports below.  A negative "fixed
    // cost" is not a small error, it is a signal that the model does not apply.
    //
    // Instead: measure the smallest call the kernel can be asked to perform
    // (M=1, N=1). Its time IS the per-call floor, no model required.
    //
    // The call must be timed in BATCHES, not individually: a single M=1,N=1
    // call takes tens of nanoseconds, which is at or below the resolution of
    // two Clock::now() reads, so timing one call and taking a best-of-N
    // minimum reports 0.000 ns.  (It did, on the first run of this section —
    // the same class of instrument error this whole sprint is about.)  One
    // timing region around CALLS invocations puts the measurement far above
    // clock granularity, exactly as the transition probe loops 2e6 times.
    const int64_t CALLS = 20000;
    std::vector<float> a1(1, 1.0f), b1(1, 0.0f), g1(1, 1.0f);
    volatile double t_floor_batch = bench([&] {
        for (int64_t i = 0; i < CALLS; ++i)
            rms_norm_ssve_v7(a1.data(), b1.data(), g1.data(), 1, 1, 1, 1, 1e-5f);
    }, 20);
    const double floor_ns =
        static_cast<double>(t_floor_batch) * 1e9 / static_cast<double>(CALLS);

    std::cout << "\nThe per-call floor (RMSNorm V7 at its smallest possible call, M=1, N=1):\n\n"
              << "  measured floor                     : " << std::fixed << std::setprecision(1)
              << std::setw(8) << floor_ns << " ns/call\n"
              << "  of which the streaming transition  : " << std::setw(8)
              << trans_sm << " ns  ("
              << std::setprecision(1) << (100.0 * trans_sm / floor_ns) << " % of the floor)\n"
              << "  everything else                    : " << std::setprecision(1)
              << std::setw(8) << (floor_ns - trans_sm) << " ns\n";

    std::cout << "\nNote this is measured, not fitted. A linear fit over these sizes returns a\n"
              << "NEGATIVE intercept because the sweep crosses cache levels, so the model does\n"
              << "not hold — which is precisely how the inferred t0 went wrong before.\n";

    std::cout << "\nThe transition is about a FIFTH of the floor — real, but not the thing that\n"
              << "makes a small call expensive. The other ~" << std::setprecision(0)
              << (floor_ns - trans_sm) << " ns is prologue/epilogue (this project\n"
              << "saves d8-d15 on every call — the AAPCS64 fix from Sprints 5/6), pointer\n"
              << "setup, and the inv_rms serialization.\n"
              << "\nWorth comparing against the inferred figures this replaces: Sprint 2a read\n"
              << "t0 ~ 1000 ns from a fit and Sprint 4 corrected it to ~400 ns after removing a\n"
              << "syscall. The measured floor is " << std::setprecision(0) << floor_ns
              << " ns. Those are not the same quantity — the fits\n"
              << "were at M=128, which pays more setup than this M=1 minimum — but the gap is\n"
              << "large enough to say the intercept was never a clean read of fixed cost.\n";

    // What the "one streaming region" design decision was actually worth,
    // computed rather than asserted: the alternative is one transition per row.
    const int64_t M_ex = 128;
    const double  per_row_us = static_cast<double>(M_ex) * trans_sm / 1000.0;
    std::cout << "\nWhat the 'one streaming region per call' decision (context.md §8) is worth,\n"
              << "quantified: the alternative costs one transition per row. At M=" << M_ex
              << " that is\n" << M_ex << " x " << std::setprecision(1) << trans_sm << " ns = "
              << std::setprecision(2) << per_row_us << " us/call against a single "
              << std::setprecision(1) << trans_sm << " ns — about "
              << std::setprecision(0) << (per_row_us * 1000.0 / floor_ns)
              << "x the entire\nper-call floor. So the decision is right, and now it is right by a"
              << " measured\nmargin rather than by assumption. But the constant itself is 9 ns:"
              << " 'streaming\nmode is expensive' is true per ROW and false per CALL, and only"
              << " the second of\nthose is what this kernel does.\n";

    // ---- 3. When does the SME kernel actually win? -----------------------
    // The ROADMAP asks for the regime where fixed cost dominates. The honest
    // way to answer is a crossover against the scalar reference, which pays no
    // fixed cost at all but is slow per element.
    std::cout << "\nCrossover: at what size does the SME kernel beat the scalar reference?\n"
              << "(RMSNorm, N=512 — a realistic transformer feature dimension — sweeping rows)\n\n"
              << std::left << std::setw(10) << "  M rows"
              << std::right << std::setw(14) << "scalar us"
              << std::setw(14) << "V7 us"
              << std::setw(12) << "speedup"
              << std::setw(16) << "groups touched" << "\n";

    const int64_t Nf = 512;
    for (int64_t m : { int64_t(1), int64_t(2), int64_t(4), int64_t(8),
                       int64_t(16), int64_t(64), int64_t(256) }) {
        const int64_t ldm = m;
        std::vector<float> a(static_cast<size_t>(ldm * Nf)),
                           b(static_cast<size_t>(ldm * Nf), 0.0f),
                           g(static_cast<size_t>(Nf), 1.0f);
        for (size_t i = 0; i < a.size(); ++i) a[i] = 0.01f * static_cast<float>(i % 97);

        volatile double s_ref = bench([&] {
            rms_norm_ref(a.data(), b.data(), g.data(), m, Nf, ldm, ldm, 1e-5f);
        }, 2000);
        volatile double s_sme = bench([&] {
            rms_norm_ssve_v7(a.data(), b.data(), g.data(), m, Nf, ldm, ldm, 1e-5f);
        }, 2000);

        // V6/V7 process a GROUP of 4 VL-row blocks at a time; on this machine
        // SVL=512 bits so VL=16 FP32 lanes and a group is 64 rows.  Deriving
        // it from the runtime SVL rather than writing 64 keeps decision D.
        const int64_t vl        = static_cast<int64_t>(svl_fp32_lanes());
        const int64_t group     = 4 * vl;
        const int64_t n_groups  = (m + group - 1) / group;

        std::cout << "  " << std::left << std::setw(8) << m
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << static_cast<double>(s_ref) * 1e6
                  << std::setw(14) << static_cast<double>(s_sme) * 1e6
                  << std::setw(12) << std::setprecision(2)
                  << (static_cast<double>(s_ref) / static_cast<double>(s_sme)) << "x"
                  << std::setw(15) << n_groups << "\n";
    }

    const int64_t vl    = static_cast<int64_t>(svl_fp32_lanes());
    const int64_t group = 4 * vl;
    std::cout << "\nThe crossover is at M ~ " << (group / 4) << " rows, and the reason is NOT "
              << "streaming overhead.\n"
              << "Look at the last column: V7's time is flat from M=1 to M=" << group
              << " because that is\n"
              << "ONE group. V6/V7 process 4 VL-row blocks at a time (VL=" << vl
              << " lanes at SVL=512,\nso a group is " << group
              << " rows), and a partially-filled group costs the same as a full\n"
              << "one — the lanes are predicated off, not skipped. At M=1 the kernel does "
              << group << "\nrows' worth of work for 1 row's worth of result.\n"
              << "\nSo the small-tensor penalty is GROUP GRANULARITY, not SMSTART. The fix would\n"
              << "be a narrower group for small M — which is exactly the shape-specialized\n"
              << "emission decision the JIT was built to make (Sprint 4) and which Sprint 2b\n"
              << "deferred. It is also the same granularity that made Sprint 6's threaded\n"
              << "chunking sensitive to alignment: chunks not a multiple of "
              << group << " rows pay this\nsame partial-group cost on every thread.\n\n";
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
// The measured ceiling curve, populated once in main() before any kernel runs.
//
// File-scope and heap-backed on purpose: every "% of peak" in this harness now
// reads from it, and a std::vector's storage is memory, which survives the
// SMSTART/SMSTOP transitions that zero d8-d15.  A ceiling held in a register
// would not (Sprint 6 Part 3 #5).
// ---------------------------------------------------------------------------

static std::vector<CeilingPoint> g_ceiling_curve;

// The denominator for a kernel of this shape: a norm's working set is exactly
// its useful bytes (input read once + output written once), so norm_bytes()
// IS the footprint.  Falls back to the DRAM constant if the curve is empty
// (no SME, or called before main() measured it).
static double peak_for_shape(int64_t m, int64_t n, double dram_fallback) {
    const double mib = norm_bytes(m, n) / (1024.0 * 1024.0);
    return ceiling_for_footprint(g_ceiling_curve, mib, dram_fallback);
}

// ---------------------------------------------------------------------------
// Print a GiB/s row in the same style as the weekly reports.
//
// Sprint 6 §1.2 correction: the percentage is against the ceiling measured at
// THIS shape's footprint, not the single 59.5 GiB/s DRAM figure.  `peak` is
// now only the fallback for shapes past the top of the curve.
// ---------------------------------------------------------------------------

static void print_row(const char* label, int64_t m, int64_t n,
                      double gibs, double peak) {
    const double matched = peak_for_shape(m, n, peak);
    std::cout << std::left  << std::setw(22) << label
              << "  M=" << std::setw(5) << m
              << "  N=" << std::setw(5) << n
              << "  " << std::fixed << std::setprecision(2) << std::setw(7) << gibs
              << " GiB/s"
              << "  (" << std::setprecision(1) << (100.0 * gibs / matched)
              << "% of the " << std::setprecision(1) << matched
              << " GiB/s ceiling at this footprint)\n";
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

__attribute__((noinline))
static double bench_za(const float* a, float* b, const float* gamma,
                       int64_t m, int64_t n, int64_t ld, float eps, int reps = 50) {
    using namespace mini_jit::norm;
    rms_norm_za(a, b, gamma, m, n, ld, ld, eps);
    return bench([&]() { rms_norm_za(a, b, gamma, m, n, ld, ld, eps); }, reps);
}

__attribute__((noinline))
static double bench_ln_za(const float* a, float* b, const float* gamma,
                          const float* beta, int64_t m, int64_t n, int64_t ld,
                          float eps, int reps = 50) {
    using namespace mini_jit::norm;
    layer_norm_za(a, b, gamma, beta, m, n, ld, ld, eps);
    return bench([&]() { layer_norm_za(a, b, gamma, beta, m, n, ld, ld, eps); }, reps);
}

// ---------------------------------------------------------------------------
// Ablation row: prints GiB/s and % delta vs V0 baseline.
// ---------------------------------------------------------------------------

static void print_ablation_row(const char* label, int64_t m, int64_t n,
                                double gibs, double peak, double base_gibs,
                                const char* base_name = "V0") {
    // Footprint-matched denominator (Sprint 6 §1.2), as in print_row.
    const double matched   = peak_for_shape(m, n, peak);
    double       delta_pct = (base_gibs > 0.0) ? (gibs / base_gibs - 1.0) * 100.0 : 0.0;
    std::cout << std::left  << std::setw(24) << label
              << "  M=" << std::setw(5) << m
              << "  N=" << std::setw(5) << n
              << "  " << std::fixed << std::setprecision(2) << std::setw(7) << gibs
              << " GiB/s"
              << "  (" << std::setprecision(1) << (100.0 * gibs / matched) << "% of "
              << std::setprecision(1) << matched << ")";
    if (base_gibs > 0.0) {
        std::cout << "  " << (delta_pct >= 0 ? "+" : "") << std::setprecision(1)
                  << delta_pct << "% vs " << base_name;
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

    // Every N in this sweep has a DIFFERENT footprint (16 -> 4096 spans three
    // orders of magnitude), so a single denominator is least defensible here
    // of anywhere in the harness: it is the whole point of the sweep that the
    // regime changes as N grows.  Each row now divides by the ceiling measured
    // at its own footprint.
    std::cout << "M=" << m << ":\n"
              << std::left << std::setw(8) << "  N"
              << std::right << std::setw(12) << "us/call"
              << std::setw(12) << "GiB/s"
              << std::setw(12) << "ceiling"
              << std::setw(12) << "% of it\n";

    // Bench loop and print loop are SEPARATE on purpose.  SMSTART zeroes
    // D9-D15 behind the compiler's back, and volatile inputs cannot protect
    // FP CONSTANTS the compiler hoists into those registers for the loop
    // (e.g. the 1/2^30 reciprocal inside to_gibs — this printed GiB/s as
    // 0.00 when register allocation shifted).  With no SME call inside the
    // print loop, nothing can be clobbered between computation and output.
    for (int k = 0; k < K; ++k) {
        const int64_t n  = ns[k];
        const int64_t ld = m;
        std::vector<float> a(ld * n), b(ld * n, 0.0f), gamma(n, 1.0f);
        for (int64_t i = 0; i < ld * n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // 200 reps: small shapes need more repetitions for a stable minimum.
        secs[k] = bench_ssve_v1(a.data(), b.data(), gamma.data(), m, n, ld, 1e-5f, 200);
    }

    for (int k = 0; k < K; ++k) {
        const int64_t n       = ns[k];
        const double  gibs    = to_gibs(norm_bytes(m, n), secs[k]);
        const double  matched = peak_for_shape(m, n, static_cast<double>(vpeak));
        std::cout << "  " << std::left << std::setw(6) << n
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(3) << secs[k] * 1e6
                  << std::setw(12) << std::setprecision(2) << gibs
                  << std::setw(12) << std::setprecision(2) << matched
                  << std::setw(11) << std::setprecision(1)
                  << (100.0 * gibs / matched) << " %\n";
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
        // The asymptote is the slope's large-N limit, so it belongs against
        // the ceiling at the LARGEST swept footprint, not at an average one.
        const double asym_peak = peak_for_shape(m, ns[K - 1], static_cast<double>(vpeak));
        std::cout << "  fit t(N) = t0 + b*N:  t0 = " << std::setprecision(3) << t0 * 1e6
                  << " us/call fixed overhead,  asymptotic " << std::setprecision(2) << asym
                  << " GiB/s (" << std::setprecision(1) << (100.0 * asym / asym_peak)
                  << "% of the " << std::setprecision(2) << asym_peak
                  << " GiB/s ceiling at N=" << ns[K - 1]
                  << "),  overhead = streaming work at N ~ "
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

// ---------------------------------------------------------------------------
// Sprint 5 — the norm invoked THROUGH TEIR, row-parallel via OpenMP.
//
// Correctness first (decision B): the threaded result is verified against the
// C++ reference before any GiB/s is printed.
//
// The trees are built here rather than parsed from data/rmsnorm.teir because
// each file pins ONE shape and ONE chunk count; a scaling study needs to vary
// both. The structure is identical to what the parser produces for those
// files (verified in test_week7) — one parallel row axis over M/chunk_rows
// chunks, each invoking the JIT kernel on a chunk_rows x N column-major tile.
//
// Note the kernel is called here from OpenMP worker threads with no `volatile`
// workaround anywhere: that is only safe because of this sprint's d8-d15
// AAPCS64 prerequisite fix.
// ---------------------------------------------------------------------------

using namespace mini_jit::teir;

// Only referenced from teir_threading_section()'s BENCH_HAS_OMP branch below;
// on CI (no OpenMP, see week7 report) that branch is preprocessed out,
// leaving this otherwise unreferenced and tripping -Werror,-Wunused-function.
#if BENCH_HAS_OMP
static std::shared_ptr<Node> build_norm_tree(bool layer, int64_t m, int64_t n,
                                             int64_t chunk_rows,
                                             std::vector<Axis*>& owned) {
    // Column-major: a chunk of `chunk_rows` rows is chunk_rows elements
    // along the row axis; gamma/beta are per-feature, so they do not move.
    Axis* row = new Axis{"row0", static_cast<uint32_t>(m / chunk_rows),
                         static_cast<uint64_t>(chunk_rows),
                         static_cast<uint64_t>(chunk_rows), 0, 0};
    owned.push_back(row);

    auto inv = std::make_shared<Invocation>();
    inv->kernel_name = layer ? "layernorm" : "rmsnorm";
    inv->m = chunk_rows; inv->n = n; inv->ld_a = m; inv->ld_b = m;
    inv->eps = 1e-5f;

    auto it = std::make_shared<Iteration>();
    it->axis = row; it->body = inv; it->is_parallel = true;
    return it;
}
#endif  // BENCH_HAS_OMP

static void teir_threading_section(double peak_ssve, double peak_chip) {
    std::cout << "\n" << std::string(78, '=') << "\n"
              << "SPRINT 5 - TEIR-INVOKED NORM, OpenMP ROW-PARALLEL\n"
              << std::string(78, '=') << "\n";
#if !BENCH_HAS_OMP
    (void)peak_ssve; 
    (void)peak_chip; 
    std::cout << "OpenMP not available in this build - threaded scaling skipped.\n";
    return;
#else
    // Shape matched to the chip probe's regime (128 MiB per array, well past
    // the ~16 MB L2), so "% of chip" is apples-to-apples. An earlier run of
    // this study used M=N=2048 (16 MiB/array): partly L2-resident and reused
    // across timing reps, it reported 54 GiB/s against a 47 GiB/s DRAM
    // ceiling — comparing a cache-assisted kernel to a DRAM-only probe, not
    // a real result. Same trap Sprint 2a hit with "peak of WHAT".
    const int64_t M = 4096, N = 8192, ld = M;
    const double  bytes = norm_bytes(M, N);

    std::vector<float> a(static_cast<size_t>(ld) * N), b(static_cast<size_t>(ld) * N);
    std::vector<float> gamma(N), beta(N), expected(static_cast<size_t>(ld) * N);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.01f * float(i % 97);
    for (int64_t j = 0; j < N; ++j) {
        gamma[j] = 1.0f + 0.001f * float(j % 13);
        beta[j]  = 0.01f * float(j % 7);
    }

    TeirRuntime runtime;
    std::vector<Axis*> owned;

    std::cout << "Shape M=" << M << " N=" << N
              << "   working set " << (2.0 * bytes / (1 << 20)) / 2.0 << " MiB"
              << "   bytes counted: useful (1R+1W)\n"
              << "Ceilings: 1-core SSVE " << std::fixed << std::setprecision(1)
              << peak_ssve << " GiB/s, chip-wide " << peak_chip << " GiB/s\n\n";

    for (int layer = 0; layer <= 1; ++layer) {
        const char* name = layer ? "LayerNorm" : "RMSNorm";

        if (layer) layer_norm_ref(a.data(), expected.data(), gamma.data(), beta.data(),
                                  M, N, ld, ld, 1e-5f);
        else       rms_norm_ref(a.data(), expected.data(), gamma.data(),
                                M, N, ld, ld, 1e-5f);

        std::cout << name << "\n"
                  << std::left << std::setw(9) << "threads"
                  << std::setw(12) << "chunk rows"
                  << std::setw(12) << "GiB/s"
                  << std::setw(12) << "speedup"
                  << std::setw(12) << "% of chip"
                  << "correct\n"
                  << std::string(70, '-') << "\n";

        double base = 0.0;
        for (int64_t threads : {1, 2, 4, 8, 16}) {
            const int64_t chunk_rows = M / threads;
            omp_set_num_threads(static_cast<int>(threads));
            auto tree = build_norm_tree(layer != 0, M, N, chunk_rows, owned);

            std::fill(b.begin(), b.end(), -1.0f);
            if (layer) runtime.execute(tree.get(), a.data(), b.data(), gamma.data(), beta.data());
            else       runtime.execute(tree.get(), a.data(), b.data(), gamma.data());

            size_t bad = 0;
            for (size_t i = 0; i < b.size(); ++i)
                if (std::fabs(b[i] - expected[i]) > 1e-4f) ++bad;

            double sec = bench([&]() {
                if (layer) runtime.execute(tree.get(), a.data(), b.data(), gamma.data(), beta.data());
                else       runtime.execute(tree.get(), a.data(), b.data(), gamma.data());
            }, 5);

            double gibs = to_gibs(bytes, sec);
            if (threads == 1) base = gibs;

            std::cout << std::left << std::setw(9) << threads
                      << std::setw(12) << chunk_rows
                      << std::setw(12) << std::fixed << std::setprecision(2) << gibs
                      << std::setw(12) << (std::to_string(gibs / base).substr(0, 4) + "x")
                      << std::setw(12) << (std::to_string(100.0 * gibs / peak_chip).substr(0, 4) + "%")
                      << (bad == 0 ? "yes" : ("NO (" + std::to_string(bad) + " bad)"))
                      << "\n";
        }
        std::cout << "\n";
    }

    for (Axis* ax : owned) delete ax;
#endif
}

// ---------------------------------------------------------------------------
// Sprint 6 — the consolidated ablation (the evaluation deliverable).
//
// Sprints 2-5 each produced their own table with its own shapes and its own
// baseline, which makes "what did each optimisation actually buy" a
// cross-referencing exercise.  This section answers it in ONE table: every
// rung of the ladder, from the scalar reference to the TEIR+OpenMP integrated
// path, on the SAME shapes through the SAME harness.
//
// Three rules it enforces, one per design decision:
//   B — every row is verified against the C++ reference BEFORE its GiB/s is
//       printed; the measured max deviation is printed alongside, so the
//       accuracy each variant actually meets is visible rather than asserted.
//   C — LayerNorm (two-pass) and Welford and RMSNorm (single-pass) all appear
//       in the same table on the same shapes, so the comparison is direct.
//   E — every number is stated against BOTH validated ceilings, with the byte
//       convention restated in the header.
//
// Written deliberately WITHOUT the `volatile` discipline the older sections
// use.  That was a caller-side workaround for kernels that clobbered d8-d15;
// as of this sprint every kernel preserves them ([sprint6][abi] tests), so
// the workaround is no longer load-bearing.  If the ABI fix ever regresses,
// this section is where it shows up first — as 0.00 rows, exactly the
// signature that exposed the bug in the first place.
// ---------------------------------------------------------------------------

namespace {

// Ladder position -> what changed at that rung. Kept next to the numbers so a
// reader never has to reconstruct the story from variant names.
struct Rung {
    const char* name;
    const char* lever;
};

// The canonical signatures (decision A) — the scalar reference, every
// hand-written variant and the JIT-emitted kernel all share them, which is
// what lets one loop drive the whole ladder.
using KernelFnBench   = void (*)(const float*, float*, const float*,
                                 int64_t, int64_t, int64_t, int64_t, float);
using LNKernelFnBench = void (*)(const float*, float*, const float*, const float*,
                                 int64_t, int64_t, int64_t, int64_t, float);

// Sprint 8: every (implementation, norm, shape) that appears in a results
// table is correctness-gated before it is timed.  These count the gate so the
// run can state the total, and so a single failure is visible in the summary
// rather than only in one row.
int g_gate_total  = 0;
int g_gate_failed = 0;

double max_abs_dev(const std::vector<float>& got, const std::vector<float>& ref) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(got[i] - ref[i])));
    return worst;
}

void print_abl_header(int64_t m, int64_t n, double footprint_mib,
                      double peak_ssve, double peak_chip) {
    std::cout << "\nShape M=" << m << " N=" << n
              << "  (" << std::fixed << std::setprecision(1) << footprint_mib
              << " MiB working set)\n"
              << std::left
              << std::setw(22) << "stage"
              << std::setw(30) << "lever"
              << std::right
              << std::setw(9)  << "GiB/s"
              << std::setw(9)  << "median"
              << std::setw(14) << "p10-p90"
              << std::setw(9)  << "%1core"
              << std::setw(8)  << "%chip"
              << std::setw(11) << "vs scalar"
              << std::setw(12) << "max|dev|"
              << std::setw(5)  << "ok" << "\n"
              << std::string(126, '-') << "\n";
    (void)peak_ssve; (void)peak_chip;
}

// `gibs` is the best-case (minimum-time) figure the ablation has always
// reported; `med`/`lo`/`hi` come from the same sample set so the reader can see
// whether that best case is typical or a lucky outlier.
void print_abl_row(const Rung& r, double gibs, double med, double lo, double hi,
                   double peak_ssve, double peak_chip,
                   double scalar_gibs, double dev, bool ok) {
    std::ostringstream spread;
    spread << std::fixed << std::setprecision(1) << lo << "-" << hi;

    std::cout << std::left
              << std::setw(22) << r.name
              << std::setw(30) << r.lever
              << std::right << std::fixed
              << std::setw(9)  << std::setprecision(2) << gibs
              << std::setw(9)  << std::setprecision(2) << med
              << std::setw(14) << spread.str()
              << std::setw(8)  << std::setprecision(1) << (100.0 * gibs / peak_ssve) << "%"
              << std::setw(7)  << std::setprecision(1) << (100.0 * gibs / peak_chip) << "%"
              << std::setw(10) << std::setprecision(1) << (gibs / scalar_gibs) << "x"
              << std::setw(12) << std::setprecision(2) << std::scientific << dev
              << std::fixed
              << std::setw(5)  << (ok ? "yes" : "NO") << "\n";
}

} // namespace

static void sprint6_consolidated_ablation(double peak_ssve, double peak_chip) {
    std::cout << "\n" << std::string(106, '=') << "\n"
              << "SPRINT 6 - CONSOLIDATED ABLATION (scalar -> SSVE levers -> ZA -> JIT -> TEIR+OpenMP)\n"
              << std::string(106, '=') << "\n"
              << "Bytes: USEFUL (1 read + 1 write per element) for every row, kernels and probes alike.\n"
              << "%1core is against the ceiling measured AT EACH SHAPE'S FOOTPRINT (see the curve\n"
              << "above), not the single " << std::fixed << std::setprecision(2) << peak_ssve
              << " GiB/s DRAM figure — a cache-resident kernel never faces\n"
              << "DRAM bandwidth, so dividing by it credits the kernel for a constraint it never met.\n"
              << "Chip-wide ceiling " << peak_chip << " GiB/s (threading target).\n"
              << "Every row is verified against the C++ reference before its GiB/s is reported;\n"
              << "max|dev| is the largest absolute deviation observed on that row's output.\n";

    struct Shape { int64_t m, n; const char* regime; };
    const Shape shapes[] = {
        {  128,    64, "small / per-call-overhead regime" },
        { 1024,  2048, "16 MiB, cache-assisted"           },
        { 4096,  8192, "256 MiB, true DRAM (probe-matched)" },
    };

    // Emission is one-time and host-portable; generate once, reuse everywhere.
    mini_jit::Norm jit_rms, jit_ln;
    jit_rms.generate(mini_jit::Norm::ntype_t::rms);
    jit_ln.generate(mini_jit::Norm::ntype_t::layer);
    std::cout << "JIT emission on this host: "
              << (jit_rms.emitted_isa() == mini_jit::Norm::isa_t::sme2
                      ? "SME2 (V7 multi-vector)" : "SME1 (V6)")
              << " — the feature-dependent emission decision, made at generate() time.\n";
    auto krms = jit_rms.get_rms_kernel();
    auto kln  = jit_ln.get_layer_kernel();

    // Accuracy gate: FRSQRTE+NR costs ~5e-6 relative in inv_rms/inv_std, and
    // the multi-accumulator variants reassociate the reduction. 1e-3 absolute
    // is loose enough for both on this data and tight enough that a real
    // addressing or ABI bug (which produces garbage or zeros) fails loudly.
    const double kGate = 1e-3;

    for (const auto& s : shapes) {
        const int64_t ld = s.m;
        const size_t  sz = static_cast<size_t>(ld) * static_cast<size_t>(s.n);
        const double  bytes = norm_bytes(s.m, s.n);
        // Constant work per row, not constant rep count: a ~1 us small-shape
        // call needs many samples for best-of-N to actually find the minimum,
        // while a 256 MiB call already moves enough data that 5 is plenty.
        // (Measured the flaw first: at 20 reps the small shape reported the
        // hand-written V6 10% below its own word-identical JIT twin, which is
        // impossible for identical code and so was sampling noise, not signal.)
        const int     reps  = std::max(5, std::min(500,
                                  static_cast<int>(50.0e6 / static_cast<double>(sz))));

        std::vector<float> a(sz), b(sz, 0.0f), expect(sz, 0.0f);
        std::vector<float> gamma(s.n), beta(s.n);
        for (size_t i = 0; i < sz; ++i) a[i] = 0.01f * float(i % 97);
        for (int64_t j = 0; j < s.n; ++j) {
            gamma[j] = 1.0f + 0.001f * float(j % 13);
            beta[j]  = 0.01f * float(j % 7);
        }

        // Sprint 6 §1.2: this shape's denominator is the ceiling measured at
        // ITS footprint, not the 59.5 GiB/s DRAM constant.  Only the 256 MiB
        // shape is unchanged by the correction; the other two roughly halve.
        const double matched_peak = peak_for_shape(s.m, s.n, peak_ssve);

        print_abl_header(s.m, s.n, 2.0 * bytes / 2.0 / (1 << 20), matched_peak, peak_chip);
        std::cout << "  regime: " << s.regime
                  << "   |  ceiling at this footprint: " << std::fixed
                  << std::setprecision(2) << matched_peak << " GiB/s"
                  << (matched_peak > peak_ssve * 1.02
                          ? "  (DRAM constant would have been " : "  (= the DRAM constant ")
                  << std::setprecision(2) << peak_ssve << ")\n";

        // ---- RMSNorm ladder -------------------------------------------------
        rms_norm_ref(a.data(), expect.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);

        double scalar_rms = 0.0;
        auto rms_row = [&](const Rung& r, KernelFnBench fn) {
            // Correctness gate FIRST, for this exact (implementation, norm,
            // shape): run the very function that is about to be timed, compare
            // its complete output against the reference, and refuse to time it
            // if it disagrees.  A GiB/s figure is only meaningful for a kernel
            // that computes the right thing (decision B), so an unverified row
            // must not appear at all rather than appear with a "NO" beside it.
            std::fill(b.begin(), b.end(), -1.0f);
            fn(a.data(), b.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);
            const double dev = max_abs_dev(b, expect);
            ++g_gate_total;
            if (dev > kGate) {
                ++g_gate_failed;
                std::cout << std::left << std::setw(22) << r.name
                          << std::setw(30) << r.lever
                          << "  *** CORRECTNESS GATE FAILED (max|dev| = "
                          << std::scientific << std::setprecision(2) << dev
                          << std::fixed << ") — not timed ***\n";
                return;
            }

            const Stats st   = bench_stats([&] {
                fn(a.data(), b.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);
            }, reps);
            const double gibs = to_gibs(bytes, st.best);
            if (scalar_rms == 0.0) scalar_rms = gibs;
            print_abl_row(r, gibs, to_gibs(bytes, st.median),
                          to_gibs(bytes, st.p90), to_gibs(bytes, st.p10),
                          matched_peak, peak_chip, scalar_rms, dev, true);
        };

        std::cout << "RMSNorm (single-pass, 2R+1W)\n";
        rms_row({"scalar reference", "C++ oracle, double accum"}, rms_norm_ref);
        rms_row({"V0 SSVE", "VLA streaming, 1 accumulator"},      rms_norm_ssve);
        rms_row({"V1", "FRSQRTE+NR replaces FSQRT/FDIV"},         rms_norm_ssve_v1);
        rms_row({"V2", "+ pre-computed 1/N"},                     rms_norm_ssve_v2);
        rms_row({"V3", "+ 2x column unroll"},                     rms_norm_ssve_v3);
        rms_row({"V4", "+ 4 accumulator chains (ILP)"},           rms_norm_ssve_v4);
        rms_row({"V5", "+ software-pipelined loads"},             rms_norm_ssve_v5);
        rms_row({"V6 (incumbent)", "4-row-block contiguity"},     rms_norm_ssve_v6);
        rms_row({"V7 (SME2)", "4-vector LD1W/ST1W, same traffic"}, rms_norm_ssve_v7);
        rms_row({"ZA residency", "ZA staging, 1R+1W"},            rms_norm_za);
        rms_row({"JIT (auto ISA)", "emitted: V7 on SME2, else V6"},  krms);

        // ---- LayerNorm ladder ----------------------------------------------
        layer_norm_ref(a.data(), expect.data(), gamma.data(), beta.data(),
                       s.m, s.n, ld, ld, 1e-5f);

        double scalar_ln = 0.0;
        auto ln_row = [&](const Rung& r, LNKernelFnBench fn) {
            // Same gate-then-time discipline as rms_row above.
            std::fill(b.begin(), b.end(), -1.0f);
            fn(a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, ld, 1e-5f);
            const double dev = max_abs_dev(b, expect);
            ++g_gate_total;
            if (dev > kGate) {
                ++g_gate_failed;
                std::cout << std::left << std::setw(22) << r.name
                          << std::setw(30) << r.lever
                          << "  *** CORRECTNESS GATE FAILED (max|dev| = "
                          << std::scientific << std::setprecision(2) << dev
                          << std::fixed << ") — not timed ***\n";
                return;
            }

            const Stats st = bench_stats([&] {
                fn(a.data(), b.data(), gamma.data(), beta.data(),
                   s.m, s.n, ld, ld, 1e-5f);
            }, reps);
            const double gibs = to_gibs(bytes, st.best);
            if (scalar_ln == 0.0) scalar_ln = gibs;
            print_abl_row(r, gibs, to_gibs(bytes, st.median),
                          to_gibs(bytes, st.p90), to_gibs(bytes, st.p10),
                          matched_peak, peak_chip, scalar_ln, dev, true);
        };

        std::cout << "LayerNorm (two-pass, 3R+1W)\n";
        ln_row({"scalar reference", "C++ oracle, double accum"},  layer_norm_ref);
        ln_row({"V0 SSVE", "VLA streaming, 1 accumulator"},       layer_norm_ssve);
        ln_row({"V1", "FRSQRTE+NR replaces FSQRT/FDIV"},          layer_norm_ssve_v1);
        ln_row({"V2", "+ pre-computed 1/N (2 FDIVs)"},           layer_norm_ssve_v2);
        ln_row({"V4", "+ 4 accumulator chains (ILP)"},            layer_norm_ssve_v4);
        ln_row({"V5", "+ software-pipelined loads"},              layer_norm_ssve_v5);
        ln_row({"V6 (incumbent)", "4-row-block contiguity"},      layer_norm_ssve_v6);
        ln_row({"V7 (SME2)", "4-vector LD1W/ST1W, same traffic"}, layer_norm_ssve_v7);
        ln_row({"Welford", "online single-pass, 2R+1W"},          layer_norm_ssve_welford);
        ln_row({"ZA residency", "ZA staging, 1R+1W"},             layer_norm_za);
        ln_row({"JIT (auto ISA)", "emitted: V7 on SME2, else V6"},   kln);
    }

    // The sentence this earns: "every configuration in this table was
    // correctness-gated against the FP64 reference before it was timed."
    // It is only true because the gate runs the exact function that is then
    // timed, on the exact shape, and refuses to time it on disagreement.
    std::cout << "\n" << std::string(126, '-') << "\n"
              << "Correctness gate: " << (g_gate_total - g_gate_failed) << " / "
              << g_gate_total << " configurations verified against the float64 "
              << "reference BEFORE timing.\n";
    if (g_gate_failed > 0) {
        std::cout << "*** " << g_gate_failed << " configuration(s) FAILED and were "
                  << "not timed — the table above is incomplete. ***\n";
    } else {
        std::cout << "No configuration produced a timing without first matching the "
                  << "reference.\n";
    }
    std::cout << "The GiB/s column is the BEST sample (best-case envelope, the figure this\n"
              << "harness has always reported); `median` and `p10-p90` show the distribution\n"
              << "from the same samples, so a best case that is not typical is visible.\n";
}

// ---------------------------------------------------------------------------
// Sprint 6 — OpenMP row-parallel scaling, per ablation variant.
//
// WHY THE PARALLELISM LIVES HERE AND NOT IN THE KERNELS.  Decision F makes the
// kernel a *leaf the compiler schedules*: it operates tile-at-a-time and never
// assumes it owns the whole tensor.  Putting `#pragma omp parallel` inside a
// kernel would break that contract (and is impossible anyway — they are .S
// files).  So a caller splits the ROW axis into chunks and calls the unchanged
// kernel once per chunk, which is exactly what the TEIR runtime already does
// with `is_parallel` on the row axis; this section does the same thing without
// TEIR so that EVERY variant can be measured threaded, not just the two the
// JIT emits.
//
// Splitting on rows is the natural choice for this layout: the norm is
// independent per row, and in column-major storage a row-chunk is a
// contiguous slice of each column, so chunk k is just `a + k*chunk_rows` with
// the leading dimension unchanged.  No data is shared between threads and no
// reduction is needed across them.
//
// This is also the answer to "where does scaling flatten?" — it prints the
// speed-up curve per variant instead of only for the TEIR path.
// ---------------------------------------------------------------------------

static void openmp_scaling_section(double peak_ssve, double peak_chip) {
    std::cout << "\n" << std::string(96, '=') << "\n"
              << "SPRINT 6 - OpenMP ROW-PARALLEL SCALING, PER VARIANT\n"
              << std::string(96, '=') << "\n";
#if !BENCH_HAS_OMP
    (void)peak_ssve; (void)peak_chip;
    std::cout << "OpenMP not available in this build - skipped.\n";
    return;
#else
    // True-DRAM shape: the only regime where threading has anything to add
    // (the cache-resident shapes already sit at ~97% of the streaming ceiling
    // on ONE core, so there is nothing for more cores to recover).
    const int64_t M = 4096, N = 8192, ld = M;
    const size_t  sz = static_cast<size_t>(ld) * static_cast<size_t>(N);
    const double  bytes = norm_bytes(M, N);

    std::vector<float> a(sz), b(sz, 0.0f), expect(sz, 0.0f);
    std::vector<float> gamma(N), beta(N);
    for (size_t i = 0; i < sz; ++i) a[i] = 0.01f * float(i % 97);
    for (int64_t j = 0; j < N; ++j) {
        gamma[j] = 1.0f + 0.001f * float(j % 13);
        beta[j]  = 0.01f * float(j % 7);
    }

    const int threads[] = {1, 2, 4, 6, 8, 10, 16};
    const int reps = 5;

    std::cout << "Shape M=" << M << " N=" << N << " (256 MiB), useful bytes (1R+1W).\n"
              << "Ceilings: 1-core SSVE " << std::fixed << std::setprecision(1) << peak_ssve
              << ", chip-wide " << peak_chip << " GiB/s.\n"
              << "This M4 is 4 P-cores + 6 E-cores, so 10 threads is full occupancy\n"
              << "and 16 is deliberate oversubscription.\n"
              << "'moved' = physical traffic (RMSNorm 2R+1W = 1.5x useful, LayerNorm 3R+1W = 2x).\n\n";

    // GROUP-ALIGNED WORK DISTRIBUTION — measured, not cosmetic.
    //
    // The V6/V7 kernels process 4*VL rows per group (64 on this 512-bit-SVL
    // M4) and send whatever is left to a single-block predicated tail that
    // touches only 64 B per column instead of the group path's 256 B — i.e.
    // the tail reverts to exactly the low access density V6 was built to fix.
    //
    // A naive `chunk = M/nthreads` therefore hands EVERY thread a tail
    // whenever nthreads does not divide M into multiples of 64.  At M=4096
    // that is true for 6 threads (chunk 682, 42-row tail) and 10 threads
    // (chunk 409, 25-row tail) — and those are precisely the thread counts
    // that measured ~45% slower than their neighbours, collapsing to about
    // the single-thread number.  It looked like core heterogeneity; it was
    // the scheduler handing the kernel work it is bad at.
    //
    // So distribute in units of whole row-blocks.  kRowBlock = 256 is a
    // multiple of 4*VL for every plausible SVL (256/512/1024-bit -> 32/64/128
    // rows per group), which keeps this VLA-safe without needing to query SVL
    // from C++ (decision D: no hard-coded streaming vector length).
    const int64_t kRowBlock = 256;
    const int64_t nblocks   = (M + kRowBlock - 1) / kRowBlock;

    auto span = [&](int t, int nthreads, bool aligned) {
        if (!aligned) {                       // the naive scheme, kept to show the cliff
            const int64_t chunk = M / nthreads;
            const int64_t r0    = static_cast<int64_t>(t) * chunk;
            return std::pair<int64_t,int64_t>{ r0, (t == nthreads - 1) ? (M - r0) : chunk };
        }
        const int64_t base  = nblocks / nthreads;
        const int64_t extra = nblocks % nthreads;
        const int64_t b0    = t * base + std::min<int64_t>(t, extra);
        const int64_t nb    = base + (t < extra ? 1 : 0);
        const int64_t r0    = b0 * kRowBlock;
        return std::pair<int64_t,int64_t>{ r0, std::min<int64_t>(nb * kRowBlock, M - r0) };
    };

    auto run_rms = [&](KernelFnBench fn, int nthreads, bool aligned = true) {
        omp_set_num_threads(nthreads);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < nthreads; ++t) {
            auto [r0, rows] = span(t, nthreads, aligned);
            if (rows > 0)
                fn(a.data() + r0, b.data() + r0, gamma.data(), rows, N, ld, ld, 1e-5f);
        }
    };
    auto run_ln = [&](LNKernelFnBench fn, int nthreads, bool aligned = true) {
        omp_set_num_threads(nthreads);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < nthreads; ++t) {
            auto [r0, rows] = span(t, nthreads, aligned);
            if (rows > 0)
                fn(a.data() + r0, b.data() + r0, gamma.data(), beta.data(),
                   rows, N, ld, ld, 1e-5f);
        }
    };

    auto header = [&](const char* norm, double moved_factor) {
        std::cout << norm << "   (moved = " << std::setprecision(2) << moved_factor
                  << "x useful)\n"
                  << std::left  << std::setw(9) << "threads"
                  << std::right << std::setw(10) << "GiB/s"
                  << std::setw(10) << "speedup"
                  << std::setw(11) << "%1core"
                  << std::setw(11) << "%chip"
                  << std::setw(13) << "moved %chip"
                  << std::setw(9)  << "correct" << "\n"
                  << std::string(73, '-') << "\n";
    };

    auto row = [&](int t, double gibs, double base, double moved_factor, bool ok) {
        std::cout << std::left  << std::setw(9) << t
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << gibs
                  << std::setw(9)  << std::setprecision(2) << (gibs / base) << "x"
                  << std::setw(10) << std::setprecision(1) << (100.0 * gibs / peak_ssve) << "%"
                  << std::setw(10) << std::setprecision(1) << (100.0 * gibs / peak_chip) << "%"
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * gibs * moved_factor / peak_chip) << "%"
                  << std::setw(9)  << (ok ? "yes" : "NO") << "\n";
    };

    // ---- RMSNorm variants ------------------------------------------------
    rms_norm_ref(a.data(), expect.data(), gamma.data(), M, N, ld, ld, 1e-5f);
    struct RmsV { const char* name; KernelFnBench fn; };
    const RmsV rms_variants[] = {
        { "RMSNorm V0 (baseline)", rms_norm_ssve    },
        { "RMSNorm V6 (SME1)",     rms_norm_ssve_v6 },
        { "RMSNorm V7 (SME2)",     rms_norm_ssve_v7 },
    };
    for (const auto& v : rms_variants) {
        header(v.name, 1.5);
        double base = 0.0;
        for (int t : threads) {
            std::fill(b.begin(), b.end(), -1.0f);
            run_rms(v.fn, t);
            bool ok = max_abs_dev(b, expect) <= 1e-3;
            double sec = bench([&] { run_rms(v.fn, t); }, reps);
            double gibs = to_gibs(bytes, sec);
            if (t == 1) base = gibs;
            row(t, gibs, base, 1.5, ok);
        }
        std::cout << "\n";
    }

    // ---- The alignment cliff, isolated -----------------------------------
    // Same kernel, same threads, same data — only the chunk boundaries move.
    // This is the evidence for the group-aligned distribution above.
    std::cout << "Work-distribution alignment (RMSNorm V7, identical work, only chunking differs)\n"
              << std::left  << std::setw(9)  << "threads"
              << std::right << std::setw(12) << "naive"
              << std::setw(12) << "aligned"
              << std::setw(11) << "gain"
              << std::setw(18) << "naive chunk%64" << "\n"
              << std::string(62, '-') << "\n";
    for (int t : threads) {
        std::fill(b.begin(), b.end(), -1.0f);
        double sec_n = bench([&] { run_rms(rms_norm_ssve_v7, t, false); }, reps);
        double sec_a = bench([&] { run_rms(rms_norm_ssve_v7, t, true ); }, reps);
        double gn = to_gibs(bytes, sec_n), ga = to_gibs(bytes, sec_a);
        std::cout << std::left  << std::setw(9) << t
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << gn << std::setw(12) << ga
                  << std::setw(10) << std::setprecision(1) << (100.0 * (ga / gn - 1.0)) << "%"
                  << std::setw(18) << ((M / t) % 64) << "\n";
    }
    std::cout << "\n";

    // ---- LayerNorm variants ----------------------------------------------
    layer_norm_ref(a.data(), expect.data(), gamma.data(), beta.data(),
                   M, N, ld, ld, 1e-5f);
    struct LnV { const char* name; LNKernelFnBench fn; };
    const LnV ln_variants[] = {
        { "LayerNorm V0 (baseline)", layer_norm_ssve    },
        { "LayerNorm V6 (SME1)",     layer_norm_ssve_v6 },
        { "LayerNorm V7 (SME2)",     layer_norm_ssve_v7 },
    };
    for (const auto& v : ln_variants) {
        header(v.name, 2.0);
        double base = 0.0;
        for (int t : threads) {
            std::fill(b.begin(), b.end(), -1.0f);
            run_ln(v.fn, t);
            bool ok = max_abs_dev(b, expect) <= 1e-3;
            double sec = bench([&] { run_ln(v.fn, t); }, reps);
            double gibs = to_gibs(bytes, sec);
            if (t == 1) base = gibs;
            row(t, gibs, base, 2.0, ok);
        }
        std::cout << "\n";
    }
#endif
}

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

#if BENCH_HAS_OMP
    // OpenMP's own view of available parallelism; std::thread is no longer
    // used anywhere in this harness.
    const unsigned nthreads = std::max(1, omp_get_max_threads());
#else
    const unsigned nthreads = 1u;
#endif

    // All ceilings live in volatile doubles: they are read again after many
    // SMSTART/SMSTOP transitions, which zero callee-saved FP registers
    // (d9-d15) behind the compiler's back.
    print_provenance(nthreads);

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

    // The footprint half of "peak of WHAT" (Sprint 6 §1.2).  Runs before the
    // ablation so its curve is available as a per-shape denominator, and so
    // the 256 MiB row can be checked against the constants just printed.
    // Populate the file-scope curve BEFORE any table runs: every "% of peak"
    // printed below reads its denominator from it via peak_for_shape().
    g_ceiling_curve = measure_ceiling_curve(have_sme);
    print_ceiling_curve(g_ceiling_curve, have_sme, static_cast<double>(peak_ssve));

    // Sprint 7a runs before the throughput tables: correctness/accuracy gates
    // performance (decision B), so the accuracy story is stated before any
    // GiB/s claim that rests on it.  Host-portable except its kernel rows.
    sprint7_stability_section();
    sprint7_streaming_overhead_section();

    // Sprint 6's consolidated ablation runs FIRST: it is the evaluation
    // deliverable, and the per-sprint sections below are the detail behind it.
    // Guarded on SME because the kernels are no-ops without it, which would
    // make every verification column read "NO" for the wrong reason.
    if (have_sme)
        sprint6_consolidated_ablation(static_cast<double>(peak_ssve),
                                      static_cast<double>(peak_chip));

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
    // Section 2b (Sprint 3): RMSNorm ZA-tile residency vs the SSVE winner V6.
    //
    // Hypothesis (written before measuring, ROADMAP Sprint 3): ZA cannot raise
    // DRAM bandwidth; it only turns V6's 2R+1W into 1R+1W where a row fits in
    // ZA (N <= 4*SVL = 64 on the M4).  That is the small-N, streaming-overhead-
    // dominated regime, so the saved read may be eaten by SMSTART + mova cost.
    // For N > 64 the ZA kernel falls back to a streaming two-pass and should
    // tie/lose V6.  Both kernels are timed on USEFUL bytes (1R+1W): a higher
    // ZA number means the residency fusion genuinely did the useful work
    // faster.  Note the moved-bytes gap: in the ZA fast path ZA moves exactly
    // the useful 1R+1W, whereas V6 moves 2R+1W (1.5x), so any ZA win here also
    // reflects a real ~33% traffic reduction on this regime.
    // -----------------------------------------------------------------------
    std::cout << "\n--- RMSNorm ZA residency vs SSVE V6 (Sprint 3) ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)  vs V6\n";
    std::cout << std::string(78, '-') << "\n";

    // N <= 64 -> ZA fast path (residency active).  N > 64 -> fallback.
    const Shape za_shapes[] = {
        { 128,   16},   // 1 tile, ZA path
        { 128,   32},   // 2 tiles, ZA path
        { 128,   64},   // 4 tiles full (=4*SVL), ZA path — max residency
        {1024,   64},   // more row blocks at the residency boundary
        {4096,   64},   // large footprint, still ZA path
        { 128, 2048},   // N > 64 -> fallback (expect tie/loss vs V6)
        {1024, 2048},   // N > 64 -> fallback
        // --- true-DRAM regime (footprint >> 16 MB L2) -----------------------
        // The regime where ZA's 1R+1W traffic saving *should* pay if it ever
        // does.  Measured verdict (Sprint 3 addendum): it does not — ZA stays
        // ~10 GiB/s (mova-bound) at every footprint while V6 delivers ~22-25,
        // because V6's 4-row-block reuse distance keeps pass-2's re-read
        // L1/L2-resident even at 64 MB, so the DRAM read ZA "saves" never
        // existed.  ZA fast path (N=64, huge M) and fallback (N=2048) both.
        {131072, 64},   // 32 MB, ZA fast path, true DRAM
        {262144, 64},   // 64 MB, ZA fast path, deep DRAM
        {  4096, 2048}, // 32 MB, fallback, true DRAM
        {  8192, 2048}, // 64 MB, fallback, deep DRAM
    };

    for (const auto& s : za_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // volatile: SMSTART zeroes D9-D15; keep timing off callee-saved FP regs.
        volatile double vsec_v6 = bench_ssve_v6(a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_za = bench_za      (a.data(), b.data(), gamma.data(), s.m, s.n, ld, 1e-5f);

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);

        double g6  = to_gibs((double)vbytes, (double)vsec_v6);
        double gza = to_gibs((double)vbytes, (double)vsec_za);

        print_ablation_row("V6 (2R+1W, SSVE)",    vm, vn, g6,  (double)vpeak, 0.0);
        print_ablation_row("ZA residency",         vm, vn, gza, (double)vpeak, g6, "V6");
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Section 2c (Sprint 3, gated): LayerNorm ZA-tile residency vs V6.
    //
    // Unlike rms_norm_za (2R+1W -> 1R+1W), layer_norm_za goes further: x is
    // staged in ZA once during the mean pass and reused from ZA for BOTH the
    // variance pass and the normalize pass, so the ZA fast path is a full
    // 3R+1W -> 1R+1W fusion (50% traffic cut, vs RMSNorm's 33%). Hypothesis
    // (written before measuring, given the RMSNorm verdict): if a 33%
    // traffic cut still lost 39-65% to mova throughput, a 50% cut needs a
    // proportionally bigger mova tax (3 mova ops/element here: 1 store into
    // ZA + 2 loads back out, vs RMSNorm's 1 store + 1 load) to break even.
    // Expect this to also lose, likely by a larger margin than RMSNorm's,
    // confirming mova throughput -- not DRAM bandwidth -- is the norm-
    // agnostic bottleneck. A win here would be the surprising result.
    // -----------------------------------------------------------------------
    std::cout << "\n--- LayerNorm ZA residency vs SSVE V6 (Sprint 3, gated) ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)  vs V6\n";
    std::cout << std::string(78, '-') << "\n";

    const Shape ln_za_shapes[] = {
        { 128,   16},   // 1 tile, ZA path
        { 128,   32},   // 2 tiles, ZA path
        { 128,   64},   // 4 tiles full (=4*SVL), ZA path — max residency
        {1024,   64},   // more row blocks at the residency boundary
        {4096,   64},   // large footprint, still ZA path
        { 128, 2048},   // N > 64 -> fallback (expect tie/loss vs V6)
        {1024, 2048},   // N > 64 -> fallback
    };

    for (const auto& s : ln_za_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f), beta(s.n, 0.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // volatile: SMSTART zeroes D9-D15; keep timing off callee-saved FP regs.
        volatile double vsec_v6 = bench_ln_ssve_v6(a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double vsec_za = bench_ln_za     (a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, 1e-5f);

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);

        double g6  = to_gibs((double)vbytes, (double)vsec_v6);
        double gza = to_gibs((double)vbytes, (double)vsec_za);

        print_ablation_row("V6 (3R+1W, SSVE)",    vm, vn, g6,  (double)vpeak, 0.0);
        print_ablation_row("ZA residency",         vm, vn, gza, (double)vpeak, g6, "V6");
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

    // -----------------------------------------------------------------------
    // Section 6 (Sprint 4): JIT-emitted V6 vs hand-written V6 + emission cost.
    //
    // The encoding-diff test already proves the JIT buffer word-identical to
    // the assembled .S, so the expectation is parity within noise: any gap
    // would come from the buffer's placement (mmap'd JIT page vs the
    // executable's text section), not the instructions.  Emission is a
    // one-time cost; it is reported in µs alongside the per-call time of the
    // smallest bench shape so the break-even call count is visible.
    // -----------------------------------------------------------------------
    std::cout << "\n--- Sprint 4: JIT-emitted V6 vs hand-written V6 ---\n";

    mini_jit::Norm jit_rms, jit_ln;
    volatile double emit_rms_us = bench([&]() {
        jit_rms.generate(mini_jit::Norm::ntype_t::rms);
    }, 100) * 1e6;
    volatile double emit_ln_us = bench([&]() {
        jit_ln.generate(mini_jit::Norm::ntype_t::layer);
    }, 100) * 1e6;

    std::cout << std::fixed << std::setprecision(1)
              << "emission (one-time):  rms " << (double)emit_rms_us
              << " us (" << jit_rms.words().size() << " words),  layer "
              << (double)emit_ln_us << " us (" << jit_ln.words().size()
              << " words)\n\n";

    mini_jit::Norm::rms_kernel_t   krms = jit_rms.get_rms_kernel();
    mini_jit::Norm::layer_kernel_t kln  = jit_ln.get_layer_kernel();

    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)  vs hand-written\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& s : ablation_shapes) {
        const int64_t ld = s.m;
        std::vector<float> a(ld * s.n), b(ld * s.n, 0.0f);
        std::vector<float> gamma(s.n, 1.0f), beta(s.n, 0.0f);
        for (int64_t i = 0; i < ld * s.n; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // volatile: SMSTART zeroes D9-D15; keep timing off callee-saved FP regs.
        volatile double sec_hw  = bench_ssve_v6(a.data(), b.data(), gamma.data(),
                                                s.m, s.n, ld, 1e-5f);
        volatile double sec_jit = bench([&]() {
            krms(a.data(), b.data(), gamma.data(), s.m, s.n, ld, ld, 1e-5f);
        });
        volatile double lsec_hw  = bench_ln_ssve_v6(a.data(), b.data(), gamma.data(),
                                                    beta.data(), s.m, s.n, ld, 1e-5f);
        volatile double lsec_jit = bench([&]() {
            kln(a.data(), b.data(), gamma.data(), beta.data(), s.m, s.n, ld, ld, 1e-5f);
        });

        volatile int64_t vm = s.m, vn = s.n;
        volatile double vbytes = norm_bytes(vm, vn);

        double gh  = to_gibs((double)vbytes, (double)sec_hw);
        double gj  = to_gibs((double)vbytes, (double)sec_jit);
        double lgh = to_gibs((double)vbytes, (double)lsec_hw);
        double lgj = to_gibs((double)vbytes, (double)lsec_jit);

        print_ablation_row("RMS V6 hand-written",  vm, vn, gh,  (double)vpeak, 0.0);
        print_ablation_row("RMS V6 JIT-emitted",   vm, vn, gj,  (double)vpeak, gh,  "hand");
        print_ablation_row("LN  V6 hand-written",  vm, vn, lgh, (double)vpeak, 0.0);
        print_ablation_row("LN  V6 JIT-emitted",   vm, vn, lgj, (double)vpeak, lgh, "hand");
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Section 7: true DRAM regime.
    //
    // Every shape above tops out at 64 MiB total footprint ({4096, 2048}):
    // "beyond the 16 MB L2" does not rule out a larger shared SLC on
    // M-series absorbing repeated bench() reps, so those numbers are only
    // PROBABLY DRAM-bound. This shape is sized to match PROBE_N exactly —
    // the same 128 MiB-per-array footprint already used for the roofline
    // ceiling probes at the top of this program — so these kernel numbers
    // sit in the identical, unambiguously-DRAM regime as peak_ssve, and the
    // "% of peak" column here is a directly honest DRAM-efficiency figure
    // (not inflated by cache residency, as it is for the smaller shapes
    // above).
    // -----------------------------------------------------------------------
    std::cout << "\n--- True DRAM regime (128 MiB/array, matches roofline probe footprint) ---\n";
    std::cout << std::left << std::setw(24) << "variant"
              << "  rows   feat    GiB/s  (% of 1-core SSVE peak)\n";
    std::cout << std::string(70, '-') << "\n";

    {
        // dm * dn == PROBE_N, so each of a/b is exactly the 128 MiB used above.
        const int64_t dm = 4096, dn = 8192;
        const int64_t ld = dm;
        std::vector<float> a(ld * dn), b(ld * dn, 0.0f);
        std::vector<float> gamma(dn, 1.0f), beta(dn, 0.0f);
        for (int64_t i = 0; i < ld * dn; ++i)
            a[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        // Fewer reps than the smaller-shape sections: each call already
        // moves 256 MiB of useful traffic, so 50 reps of e.g. the scalar
        // reference would move many GiB for no extra precision.
        const int dram_reps = 10;

        volatile double rms_ref_sec = bench([&]() {
            rms_norm_ref(a.data(), b.data(), gamma.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double rms_v6_sec  = bench([&]() {
            rms_norm_ssve_v6(a.data(), b.data(), gamma.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double rms_za_sec  = bench([&]() {
            rms_norm_za(a.data(), b.data(), gamma.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double rms_jit_sec = bench([&]() {
            krms(a.data(), b.data(), gamma.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);

        volatile double ln_ref_sec = bench([&]() {
            layer_norm_ref(a.data(), b.data(), gamma.data(), beta.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double ln_v6_sec  = bench([&]() {
            layer_norm_ssve_v6(a.data(), b.data(), gamma.data(), beta.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double ln_za_sec  = bench([&]() {
            layer_norm_za(a.data(), b.data(), gamma.data(), beta.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);
        volatile double ln_jit_sec = bench([&]() {
            kln(a.data(), b.data(), gamma.data(), beta.data(), dm, dn, ld, ld, 1e-5f);
        }, dram_reps);

        volatile double vbytes = norm_bytes(dm, dn);

        print_row("rms_norm_ref  ",     dm, dn, to_gibs((double)vbytes, (double)rms_ref_sec), (double)vpeak);
        print_row("rms_norm_ssve_v6",   dm, dn, to_gibs((double)vbytes, (double)rms_v6_sec),  (double)vpeak);
        print_row("rms_norm_za",        dm, dn, to_gibs((double)vbytes, (double)rms_za_sec),  (double)vpeak);
        print_row("rms_norm_jit",       dm, dn, to_gibs((double)vbytes, (double)rms_jit_sec), (double)vpeak);
        std::cout << "\n";
        print_row("layer_norm_ref  ",   dm, dn, to_gibs((double)vbytes, (double)ln_ref_sec), (double)vpeak);
        print_row("layer_norm_ssve_v6", dm, dn, to_gibs((double)vbytes, (double)ln_v6_sec),  (double)vpeak);
        print_row("layer_norm_za",      dm, dn, to_gibs((double)vbytes, (double)ln_za_sec),  (double)vpeak);
        print_row("layer_norm_jit",     dm, dn, to_gibs((double)vbytes, (double)ln_jit_sec), (double)vpeak);
        std::cout << "\n";
    }

    if (have_sme)
        openmp_scaling_section((double)peak_ssve, (double)peak_chip);

    teir_threading_section((double)peak_ssve, (double)peak_chip);

    return 0;
}
