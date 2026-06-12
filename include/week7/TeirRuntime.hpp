
#ifndef MINI_JIT_TEIR_RUNTIME_HPP
#define MINI_JIT_TEIR_RUNTIME_HPP

#include "Teir.hpp"
#include "TeirParser.hpp"
#include "week6/unary.hpp"
#include "week6/gemm.hpp"

#include <memory>
#include <string>
#include <map>
#include <stdexcept>

namespace mini_jit::teir {

// ─────────────────────────────────────────────────────────────────────────────
// Execution context threaded through the recursive traversal.
// ─────────────────────────────────────────────────────────────────────────────
struct ExecContext {
    // Live tensor data pointers, keyed by tensor name ("in0", "in1", "out", "in", …)
    std::map<std::string, float*> ptrs;
    // Current iteration index for each axis (axis name → index); used by guards.
    std::map<std::string, uint32_t> iter_idx;
};

// ─────────────────────────────────────────────────────────────────────────────
// TeirRuntime
//
// Compiles a TeirObject into a reusable KernelFn and executes it.
//
// Lowering rules (spec §10.6):
//   Zero        → Unary kernel  (ptype_t::zero)
//   Copy        → Unary kernel  (ptype_t::identity)
//   Contraction → Gemm kernel   (exactly one M, N, K axis → GEMM)
//
// Leading dimensions are derived from TEIR axis byte strides by detecting
// which role axis has unit stride on each tensor (spec §10.6 / Table 10.6.2).
// Strides in the TeirObject come from the .teir file and are in BYTES; the
// runtime divides by sizeof(float) before doing float* pointer arithmetic.
// ─────────────────────────────────────────────────────────────────────────────
class TeirRuntime {
public:
    TeirRuntime();

    // Compile a TeirObject into a reusable kernel.
    // On success, result.error is empty and result.kernel is callable.
    // On failure (unsupported configuration), result.error describes the problem.
    struct CompileResult {
        KernelFn    kernel;
        std::string error;
        explicit operator bool() const { return error.empty(); }
    };

    CompileResult compile(const TeirObject& obj);

    // Convenience: parse a .teir file and compile in one call.
    CompileResult compile_file(const std::string& path);

private:
    // ── Kernel caches ────────────────────────────────────────────────────

    // Key for the Unary cache (Zero and Copy share layout)
    struct KernelKey {
        uint32_t M, N, K;
        bool operator<(const KernelKey& o) const {
            if (M != o.M) return M < o.M;
            if (N != o.N) return N < o.N;
            return K < o.K;
        }
    };

    // Key for the Gemm cache — must include trans flags because the same (M,N,K)
    // with different trans settings produces a different kernel.
    struct GemmKey {
        uint32_t M, N, K, trans_a, trans_b, trans_c;
        bool operator<(const GemmKey& o) const {
            if (M       != o.M)       return M       < o.M;
            if (N       != o.N)       return N       < o.N;
            if (K       != o.K)       return K       < o.K;
            if (trans_a != o.trans_a) return trans_a < o.trans_a;
            if (trans_b != o.trans_b) return trans_b < o.trans_b;
            return trans_c < o.trans_c;
        }
    };

    mini_jit::Unary unary_gen_;
    mini_jit::Gemm  gemm_gen_;

    std::map<KernelKey, mini_jit::Unary::kernel_t> zero_cache_;
    std::map<KernelKey, mini_jit::Unary::kernel_t> copy_cache_;
    std::map<GemmKey,   mini_jit::Gemm::kernel_t>  gemm_cache_;

    mini_jit::Unary::kernel_t get_zero_kernel(uint32_t M, uint32_t N);
    mini_jit::Unary::kernel_t get_copy_kernel(uint32_t M, uint32_t N);
    mini_jit::Gemm::kernel_t  get_gemm_kernel(uint32_t M, uint32_t N, uint32_t K,
                                               uint32_t trans_a, uint32_t trans_b,
                                               uint32_t trans_c);

    // ── Traversal ────────────────────────────────────────────────────────

    void traverse(const Node*       node,
                  ExecContext&      ctx,
                  const TeirObject& obj) const;

    void invoke_primitive(const Invocation* inv,
                          ExecContext&      ctx,
                          const TeirObject& obj) const;

    // ── Validation ───────────────────────────────────────────────────────

    // Returns an empty string on success, or an error description if the
    // primitive cannot be lowered by the current code generators.
    std::string validate_primitive(const Primitive& prim) const;
};

} // namespace mini_jit::teir
#endif