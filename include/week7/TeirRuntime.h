#ifndef MINI_JIT_TEIR_RUNTIME_H
#define MINI_JIT_TEIR_RUNTIME_H

#include "Teir.h"
#include "week6/unary.hpp"
#include "week6/gemm.hpp"
#include "norm/jit_norm.hpp"
#include <map>
#include <memory>
#include <string>
#include <tuple>

namespace mini_jit::teir {

class TeirRuntime {
public:
    TeirRuntime();

    // Compile TEIR into a reusable function pointer
    mini_jit::teir::CompiledKernel compile(std::shared_ptr<Node> root);

    // Load TEIR object by filename mapping
    std::shared_ptr<Node> load_teir(const std::string& filename);

    // Execute the loop nest. d is only used by trees whose leaf kernel
    // needs a 4th data stream (LayerNorm's beta, see build_layernorm_tree);
    // every other tree ignores it, so existing 3-pointer call sites don't
    // need to change.
    void execute(Node* root, float* a, float* b, float* c, float* d = nullptr);

    // Task builders with correct strides
    std::shared_ptr<Node> build_transposition_tree();
    std::shared_ptr<Node> build_matmul_tree();
    std::shared_ptr<Node> build_contraction_tree();
    std::shared_ptr<Node> build_rmsnorm_tree();
    std::shared_ptr<Node> build_layernorm_tree();

private:
    void traverse(Node* node, RuntimeContext& ctx);

    // Lazily generate & cache kernels per shape (and, for GEMM, per layout
    // combination) — the parsed .teir primitives need whatever shape and
    // storage order their own axes/strides declare (e.g. a row-major
    // 32x64x512 contraction, a 48x32 transposing copy), so kernels are
    // generated on first use and reused for any later invocation asking
    // for the same configuration.
    mini_jit::Unary::kernel_t get_unary_kernel(uint32_t m, uint32_t n,
                                               mini_jit::Unary::ptype_t ptype,
                                               uint32_t trans_b);
    mini_jit::Gemm::kernel_t  get_gemm_kernel(uint32_t m, uint32_t n, uint32_t k,
                                              uint32_t trans_a, uint32_t trans_b, uint32_t trans_c);

    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, mini_jit::Unary::kernel_t> m_unary_cache;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, mini_jit::Gemm::kernel_t> m_gemm_cache;

    mini_jit::Norm norm_rms_gen;
    mini_jit::Norm norm_layer_gen;
    mini_jit::Norm::rms_kernel_t   rmsnorm_kernel   = nullptr;
    mini_jit::Norm::layer_kernel_t layernorm_kernel = nullptr;
};

} 
#endif