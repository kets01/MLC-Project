#pragma once
#include <sys/mman.h>
#include <vector>
#include <cstring>
#include <iostream>

#ifdef __APPLE__
#include <pthread.h>
#include <libkern/OSCacheControl.h>
#endif

class JitEngine {
public:
    template<typename FuncPtr>
    static FuncPtr generate(const std::vector<uint32_t>& opcodes) {
        size_t size = opcodes.size() * sizeof(uint32_t);
        size = (size + 4095) & ~4095;

#ifdef __APPLE__
        void* ptr = mmap(
            nullptr,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON | MAP_JIT,
            -1,
            0
        );
#else
        void* ptr = mmap(
            nullptr,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0
        );
#endif

        if (ptr == MAP_FAILED) {
            perror("mmap failed");
            return nullptr;
        }

#ifdef __APPLE__
        pthread_jit_write_protect_np(false);
#endif

        memcpy(ptr, opcodes.data(), opcodes.size()*sizeof(uint32_t));

#ifdef __APPLE__
        sys_icache_invalidate(ptr, size);
        pthread_jit_write_protect_np(true);
#endif

        if (mprotect(ptr, size, PROT_READ | PROT_EXEC) != 0) {
            perror("mprotect failed");
            munmap(ptr, size);
            return nullptr;
        }
        return reinterpret_cast<FuncPtr>(ptr);
    }
};