#include "week3/gemm_sme.hpp"

// A is Col-Major: A[i][k] = a[k*lda + i]
// B is Row-Major: B[k][j] = b[k*ldb + j]
// C is Col-Major: C[i][j] = c[j*ldc + i]

void ref_gemm_32_32_1(const float* a, const float* b, float* c, int64_t lda, int64_t ldb, int64_t ldc) {
    (void)lda;
    (void)ldb;
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            // k=0 only
            c[j * ldc + i] += a[i] * b[j];
        }
    }
}

void ref_gemm_32_32_512(const float* a, const float* b, float* c, int64_t lda, int64_t ldb, int64_t ldc) {
    for (int k = 0; k < 512; ++k) {
        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 32; ++i) {
                c[j * ldc + i] += a[k * lda + i] * b[k * ldb + j];
            }
        }
    }
}

void ref_gemm_512_32_512(const float* a, const float* b, float* c, int64_t lda, int64_t ldb, int64_t ldc) {
    for (int k = 0; k < 512; ++k) {
        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 512; ++i) {
                c[j * ldc + i] += a[k * lda + i] * b[k * ldb + j];
            }
        }
    }
}

void ref_gemm_512_512_512(const float* a, const float* b, float* c, int64_t lda, int64_t ldb, int64_t ldc) {
    for (int k = 0; k < 512; ++k) {
        for (int j = 0; j < 512; ++j) {
            for (int i = 0; i < 512; ++i) {
                c[j * ldc + i] += a[k * lda + i] * b[k * ldb + j];
            }
        }
    }
}