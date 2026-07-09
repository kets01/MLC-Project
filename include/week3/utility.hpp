

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