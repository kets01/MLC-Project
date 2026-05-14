#include <sys/sysctl.h>

bool cpu_supports_sme() {
    int val = 0;
    size_t len = sizeof(val);
    // Check for the "FEAT_SME" capability on Apple Silicon
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &val, &len, NULL, 0) == 0) {
        return val != 0;
    }
    return false;
}