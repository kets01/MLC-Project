

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <cstdint>
#include <string>

/**
 * Checks if the current Apple Silicon CPU supports SME (Scalable Matrix Extension).
 *
 * Sprint-4 fix: the sysctlbyname result is cached in a function-local static.
 * The previous version performed the syscall on EVERY call; the mini_jit::norm
 * guard wrappers call this per kernel invocation, so small-shape benchmarks
 * were paying ~1 us of syscall per call (75% overhead at M=128/N=64 — found
 * when the word-identical JIT kernel, called without the wrapper, benchmarked
 * faster than the hand-written one).  CPU features cannot change at runtime,
 * so caching is safe; static-local init is thread-safe in C++11+.
 */
inline bool cpu_supports_sme() {
#ifdef __APPLE__
    static const bool supported = []() {
        int val = 0;
        size_t len = sizeof(val);
        // FEAT_SME is available on M4 and later.
        if (sysctlbyname("hw.optional.arm.FEAT_SME", &val, &len, NULL, 0) == 0) {
            return val != 0;
        }
        return false;
    }();
    return supported;
#else
    // Non-Apple platforms: no probe wired up.
    return false;
#endif
}

/**
 * Checks for SME2 (multi-vector loads/stores, multi-vector ZA ops).
 *
 * Sprint 6: the project docs conservatively assumed SME1 only, but sysctl on
 * the target M4 reports FEAT_SME2 = 1 — verified, and the multi-vector
 * LD1W/ST1W forms were confirmed to execute inside a streaming region on the
 * hardware before any kernel was written on top of them.
 *
 * SME2 implies SME, so this is strictly narrower than cpu_supports_sme():
 * an SME2 kernel is always a valid choice only where BOTH are true. Callers
 * use it to dispatch to an SME2 variant and fall back to the SME1 kernel
 * otherwise, so the same binary runs on M1/M2 (no SME at all) and on M4.
 *
 * Cached for the same reason as cpu_supports_sme(): the norm guard wrappers
 * call it per invocation, and an uncached sysctl there cost ~1 us per call.
 */
inline bool cpu_supports_sme2() {
#ifdef __APPLE__
    static const bool supported = []() {
        int val = 0;
        size_t len = sizeof(val);
        if (sysctlbyname("hw.optional.arm.FEAT_SME2", &val, &len, NULL, 0) == 0) {
            return val != 0;
        }
        return false;
    }();
    return supported;
#else
    return false;
#endif
}