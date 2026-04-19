#ifndef FMADD_H
#define FMADD_H
#include <cstdint>

extern "C" {
    
    float fmadd_asm(uint64_t iterations);
}
#endif