#include <cstdint>

extern "C" {
    // The ASM Microkernel: Computes 32x32 block for K=1
    void gemm_32_32_1(const float* a, const float* b, float* c, 
                      int64_t lda, int64_t ldb, int64_t ldc);
}

// Level 1: Loop over K (Inner dimension)
void gemm_32_32_512(const float* a, const float* b, float* c, 
                    int64_t lda, int64_t ldb, int64_t ldc);

// Level 2: Loop over M (Rows of C/A)
void gemm_512_32_512(const float* a, const float* b, float* c, 
                     int64_t lda, int64_t ldb, int64_t ldc);

// Level 3: Final GEMM (M=512, N=512, K=512)
void gemm_512_512_512(const float* a, const float* b, float* c, 
                       int64_t lda, int64_t ldb, int64_t ldc);