#include "week3/gemm_sme.hpp"

void gemm_32_32_512(const float* a, const float* b, float* c, 
                    int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t k = 0; k < 512; ++k) {
        // A (col-major): next k-column is at offset lda
        // B (row-major): next k-row is at offset ldb
        gemm_32_32_1(a + (k * lda), b + (k * ldb), c, lda, ldb, ldc);
    }
}

void gemm_512_32_512(const float* a, const float* b, float* c, 
                     int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t m = 0; m < 512; m += 32) {
        // Shift A and C down by 32 rows
        gemm_32_32_512(a + m, b, c + m, lda, ldb, ldc);
    }
}

void gemm_512_512_512(const float* a, const float* b, float* c, 
                       int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t n = 0; n < 512; n += 32) {
        // Shift B and C right by 32 columns
        // B (row-major): next 32 cols start 32 elements forward in the row
        // C (col-major): next 32 cols start 32 * ldc forward
        gemm_512_32_512(a, b + n, c + (n * ldc), lda, ldb, ldc);
    }
}