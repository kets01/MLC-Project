#ifndef MINI_JIT_GEMM_H
#define MINI_JIT_GEMM_H

#include <cstdint>

namespace mini_jit {
  class Gemm;
}

class mini_jit::Gemm {
  public:
    /// data type
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// error codes
    enum class error_t : int32_t {
      success               = 0,
      err_unsupported_dtype = 1,   // fp64 emission not implemented (no callers)
      err_shape             = 2    // m, n, k must be positive multiples of 16
    };

    /**
     * @brief Generate a kernel computing C += A * B (accumulating — the
     *        TEIR schedules zero C once via the guarded Zero primitive and
     *        then invoke GEMM repeatedly over outer K chunks, so the kernel
     *        must accumulate; this also matches the week3 hand-written
     *        kernels, which load C into ZA before the FMOPA loop).
     *
     * Layout semantics (ld_* are element counts, passed at CALL time):
     *   trans_a=0: A column-major (M contiguous), ld_a = distance between columns.
     *   trans_a=1: A row-major (K contiguous),    ld_a = distance between rows.
     *   trans_b=0: B column-major (K contiguous), ld_b = distance between columns.
     *   trans_b=1: B row-major (N contiguous),    ld_b = distance between rows.
     *   trans_c likewise for C.
     *
     * FMOPA needs A's M-direction and B's N-direction as vectors; layouts
     * where that direction is not contiguous (trans_a=1, trans_b=0) are
     * handled by staging 16x16 chunks through a ZA tile (contiguous loads
     * into horizontal slices, transposed vectors read back from vertical
     * slices) — streaming mode has no gather loads on this target.
     *
     * @param m       Number of rows in A and C (multiple of 16).
     * @param n       Number of columns in B and C (multiple of 16).
     * @param k       Number of columns in A and rows in B (multiple of 16).
     * @param trans_a 0 if A is stored in column-major order, 1 if row-major.
     * @param trans_b 0 if B is stored in column-major order, 1 if row-major.
     * @param trans_c 0 if C is stored in column-major order, 1 if row-major.
     * @param dtype   Data type of the matrices (fp32 only).
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t generate( uint32_t m,
                      uint32_t n,
                      uint32_t k,
                      uint32_t trans_a,
                      uint32_t trans_b,
                      uint32_t trans_c,
                      dtype_t  dtype );

    /*
     * A kernel is a function that takes the following parameters:
     * - a:           Pointer to matrix A.
     * - b:           Pointer to matrix B.
     * - c:           Pointer to C matrix.
     * - ld_a:        Leading dimension of A.
     * - ld_b:        Leading dimension of B.
     * - ld_c:        Leading dimension of C.
     */
    using kernel_t = void (*)( void    const * a,
                               void    const * b,
                               void          * c,
                               int64_t         ld_a,
                               int64_t         ld_b,
                               int64_t         ld_c);

    /**
     * @brief Get the generated kernel: C += A * B.
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;

  private:
    kernel_t m_kernel = nullptr;
};

#endif