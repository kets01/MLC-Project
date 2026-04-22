#ifndef PERM_H
#define PERM_H
#include <cstdint>

extern "C" {
   
    void perm_neon_abc_cba(int64_t size_c, float const * abc, float * cba);

    // Reference for your local testing
    void perm_cpp_abc_cba(int64_t size_c, float const * abc, float * cba);
}
#endif