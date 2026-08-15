

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <cstdint>
#include <string>

/**
 * Checks if the current Apple Silicon CPU supports SME (Scalable Matrix Extension).
 *
 * The result is cached in a function-local static.  The norm guard wrappers
 * call this per kernel invocation, and an uncached syscall there cost ~1 us
 * per call — 75% overhead at M=128/N=64, found when the word-identical JIT
 * kernel, called without the wrapper, benchmarked faster than the hand-written
 * one.  CPU features cannot change at runtime, so caching is safe, and
 * static-local init is thread-safe in C++11+.
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
 * The feature set is DETECTED, not assumed: these docs asserted "the M4 is
 * SME1" for several sprints while sysctl reported FEAT_SME2 = 1, and the
 * multi-vector forms were confirmed to execute in a streaming region on the
 * hardware before any kernel was built on them.
 *
 * SME2 implies SME, so this is strictly narrower than cpu_supports_sme().
 * Callers dispatch to an SME2 variant and fall back to the SME1 kernel, so one
 * binary runs on M1/M2 (no SME at all) and on M4.  Cached for the same reason.
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