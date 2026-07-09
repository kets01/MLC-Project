#pragma once
#include <cstdint>
#include <vector>

// Sprint 4 — mini_jit::Norm: JIT generator for the norm primitives.
//
// Emits, at runtime, the instruction words of the measured Sprint-2 winners
// (rms_norm_ssve_v6 / layer_norm_ssve_v6 — the ZA variants lost in Sprint 3
// and are deliberately not emitted).  Verified by the encoding-diff test:
// the generated buffer must match, word for word, the toolchain-assembled
// hand-written kernel it reproduces (tests/test_norm.cpp, [sprint4]).
//
// Follows the week6 mini_jit::Unary pattern: generate() emits via InstGen,
// get_*_kernel() returns a function pointer into a week5 JitEngine buffer.
// Emission is a one-time cost; the pointer is reused across calls.
//
// File is jit_norm.hpp, not Norm.hpp: APFS is case-insensitive, so Norm.hpp
// would collide with the existing norm.hpp.

namespace mini_jit {
  class Norm;
}

class mini_jit::Norm {
  public:
    /// norm primitive to generate
    enum class ntype_t : uint32_t {
      rms   = 0,
      layer = 1
    };

    /// error codes
    enum class error_t : int32_t {
      success   = 0,
      err_alloc = 1
    };

    // Canonical kernel signatures (decision A) — identical to the C++
    // reference and the hand-written kernels in norm.hpp.
    using rms_kernel_t = void (*)( const float* a,
                                   float*       b,
                                   const float* gamma,
                                   int64_t      m,
                                   int64_t      n,
                                   int64_t      ld_a,
                                   int64_t      ld_b,
                                   float        epsilon );

    using layer_kernel_t = void (*)( const float* a,
                                     float*       b,
                                     const float* gamma,
                                     const float* beta,
                                     int64_t      m,
                                     int64_t      n,
                                     int64_t      ld_a,
                                     int64_t      ld_b,
                                     float        epsilon );

    /**
     * @brief Emit the SSVE V6 kernel for the requested norm.
     * Emission is host-portable; EXECUTING the kernel requires SME.
     * @return error_t::success on success.
     */
    error_t generate( ntype_t ntype );

    /// Kernel pointer after generate(ntype_t::rms); nullptr otherwise.
    rms_kernel_t get_rms_kernel() const;

    /// Kernel pointer after generate(ntype_t::layer); nullptr otherwise.
    layer_kernel_t get_layer_kernel() const;

    /// Instruction words of the last generate() — the encoding-diff hook.
    const std::vector<uint32_t>& words() const { return m_ops; }

  private:
    void emit_rms_v6();
    void emit_layer_v6();

    std::vector<uint32_t> m_ops;
    void*                 m_kernel = nullptr;
    ntype_t               m_ntype  = ntype_t::rms;
};
