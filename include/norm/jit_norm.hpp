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

    /// which instruction set the emitted kernel targets
    enum class isa_t : uint32_t {
      sme1      = 0,   ///< SSVE V6 — runs on any SME machine
      sme2      = 1,   ///< V7: V6 with multi-vector LD1W/ST1W (needs FEAT_SME2)
      automatic = 2    ///< pick sme2 when the host reports FEAT_SME2, else sme1
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
     * @brief Emit a kernel for the requested norm.
     *
     * Emission is host-portable (it only writes instruction words), but
     * EXECUTING the result requires SME — and, for isa_t::sme2, FEAT_SME2.
     * With isa_t::automatic the generator makes the *feature*-dependent
     * emission decision Sprint 4 left open: Sprint 6 measured V7 (SME2
     * multi-vector) at +17% over V6 for RMSNorm in the DRAM regime, so it is
     * emitted wherever the hardware supports it and V6 is emitted otherwise.
     *
     * Passing sme1/sme2 explicitly forces the choice, which is what lets the
     * encoding-diff tests compare each emitted variant against the
     * corresponding hand-written kernel regardless of the build host.
     *
     * @return error_t::success on success.
     */
    error_t generate( ntype_t ntype, isa_t isa = isa_t::automatic );

    /// ISA actually emitted by the last generate() (never isa_t::automatic).
    isa_t emitted_isa() const { return m_isa; }

    /// Kernel pointer after generate(ntype_t::rms); nullptr otherwise.
    rms_kernel_t get_rms_kernel() const;

    /// Kernel pointer after generate(ntype_t::layer); nullptr otherwise.
    layer_kernel_t get_layer_kernel() const;

    /// Instruction words of the last generate() — the encoding-diff hook.
    const std::vector<uint32_t>& words() const { return m_ops; }

  private:
    void emit_rms_v6();
    void emit_layer_v6();
    void emit_rms_v7();      ///< V6 + SME2 multi-vector accesses
    void emit_layer_v7();

    std::vector<uint32_t> m_ops;
    void*                 m_kernel = nullptr;
    ntype_t               m_ntype  = ntype_t::rms;
    isa_t                 m_isa    = isa_t::sme1;
};
