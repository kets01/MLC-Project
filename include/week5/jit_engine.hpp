
#include <sys/mman.h>
#include <pthread.h>
#include <vector>
#include <cstring>
#include <libkern/OSCacheControl.h> 

class JitEngine {
public:
    template<typename FuncPtr>
    static FuncPtr generate(const std::vector<uint32_t>& opcodes) {
        size_t size = opcodes.size() * sizeof(uint32_t);
        size = (size + 4095) & ~4095; // Page alignment

        void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, 
                         MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0);
        
        if (ptr == MAP_FAILED) return nullptr;

        pthread_jit_write_protect_np(0); // Enable Writing
        std::memcpy(ptr, opcodes.data(), opcodes.size() * sizeof(uint32_t));
        pthread_jit_write_protect_np(1); // Enable Executing
        
        sys_icache_invalidate(ptr, size);
        return reinterpret_cast<FuncPtr>(ptr);
    }
};