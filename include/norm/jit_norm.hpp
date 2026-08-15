#pragma once
#include <cstdint>
#include <vector>

// mini_jit::Norm — runtime code generator for the norm primitives.
//
// Emits the instruction words of the measured winners (V6, or V7 where
// FEAT_SME2 is present) into a week5 JitEngine buffer via InstGen, following
// the week6 mini_jit::Unary pattern.  Emission is a one-time cost and the
// returned function pointer is reused.
//
// Verified by the encoding-diff test: the generated buffer must match, word
// for word, the toolchain-assembled hand-written kernel it reproduces, so the
// generator inherits that kernel's verification instead of re-earning it.
// The ZA variants lost their ablation and are deliberately not emitted.
//
// Named jit_norm.hpp, not Norm.hpp: APFS is case-insensitive, so Norm.hpp and
// norm.hpp are the same directory entry.

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
     * Emission is host-portable — it only writes instruction words — but
     * EXECUTING the result requires SME, and FEAT_SME2 for isa_t::sme2.
     * isa_t::automatic makes the feature-dependent choice; passing sme1 or
     * sme2 explicitly is what lets the encoding-diff tests compare each
     * variant against its hand-written counterpart on any build host.
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
