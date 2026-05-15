

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <cstdint>
#include <string>

/**
 * Checks if the current Apple Silicon CPU supports SME (Scalable Matrix Extension).
 
 */
inline bool cpu_supports_sme() {
#ifdef __APPLE__
    int val = 0;
    size_t len = sizeof(val);
    // FEAT_SME is available on M4 and later.
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &val, &len, NULL, 0) == 0) {
        return val != 0;
    }
#endif
    // Return false for non-Apple platforms or if the check fails.
    return false;
}