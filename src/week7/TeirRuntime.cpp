#include "week7/TeirRuntime.h"
#include "week7/TeirParser.h"
#include <iostream>

#if __has_include(<omp.h>)
#include <omp.h>
#define HAS_OMP 1
#else
#define HAS_OMP 0
#endif

#ifndef MLC_PROJECT_DATA_DIR
#define MLC_PROJECT_DATA_DIR "."
#endif

namespace mini_jit::teir {

TeirRuntime::TeirRuntime() {
    // Emission is host-portable (only EXECUTING the emitted kernel needs
    // SME), so no SME guard is needed here. Unary/Gemm kernels are no
    // longer pre-generated at a fixed 16x16(x16) shape — the real .teir
    // files each need their own shape (48x32, 32x64x512, ...), so those
    // are generated lazily and cached by get_unary_kernel/get_gemm_kernel
    // the first time a parsed Invocation asks for them.
    if (norm_rms_gen.generate(mini_jit::Norm::ntype_t::rms) == mini_jit::Norm::error_t::success)
        rmsnorm_kernel = norm_rms_gen.get_rms_kernel();
    if (norm_layer_gen.generate(mini_jit::Norm::ntype_t::layer) == mini_jit::Norm::error_t::success)
        layernorm_kernel = norm_layer_gen.get_layer_kernel();
}

mini_jit::teir::CompiledKernel TeirRuntime::compile(std::shared_ptr<Node> root) {
    return [this, root](float* a, float* b, float* c, float* d) {
        this->execute(root.get(), a, b, c, d);
    };
}

mini_jit::Unary::kernel_t TeirRuntime::get_unary_kernel(uint32_t m, uint32_t n,
                                                        mini_jit::Unary::ptype_t ptype,
                                                        uint32_t trans_b) {
    auto key = std::make_tuple(m, n, static_cast<uint32_t>(ptype), trans_b);
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    auto it = m_unary_cache.find(key);
    if (it != m_unary_cache.end()) return it->second;

    mini_jit::Unary gen;
    gen.generate(m, n, trans_b, mini_jit::Unary::dtype_t::fp32, ptype);
    mini_jit::Unary::kernel_t kernel = gen.get_kernel();
    m_unary_cache[key] = kernel;
    return kernel;
}

mini_jit::Gemm::kernel_t TeirRuntime::get_gemm_kernel(uint32_t m, uint32_t n, uint32_t k,
                                                      uint32_t trans_a, uint32_t trans_b,
                                                      uint32_t trans_c) {
    auto key = std::make_tuple(m, n, k, trans_a | (trans_b << 1) | (trans_c << 2));
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    auto it = m_gemm_cache.find(key);
    if (it != m_gemm_cache.end()) return it->second;

    mini_jit::Gemm gen;
    if (gen.generate(m, n, k, trans_a, trans_b, trans_c,
                     mini_jit::Gemm::dtype_t::fp32) != mini_jit::Gemm::error_t::success)
        throw std::runtime_error("teir: Gemm generation failed (shape/layout unsupported)");
    mini_jit::Gemm::kernel_t kernel = gen.get_kernel();
    m_gemm_cache[key] = kernel;
    return kernel;
}

std::shared_ptr<Node> TeirRuntime::load_teir(const std::string& filename) {
    // A bare filename (no path separator) resolves against the project's
    // data/ directory, so callers don't need to know where the build
    // places the test/bench binary's working directory.
    std::string path = filename;
    if (path.find('/') == std::string::npos)
        path = std::string(MLC_PROJECT_DATA_DIR) + "/" + path;
    return parse_teir_file(path);
}

void TeirRuntime::execute(Node* root, float* a, float* b, float* c, float* d) {
    RuntimeContext ctx;
    ctx.data_a = a; ctx.data_b = b; ctx.data_c = c; ctx.data_d = d;
    traverse(root, ctx);
}

void TeirRuntime::traverse(Node* node, RuntimeContext& ctx) {
    if (auto* iter = dynamic_cast<Iteration*>(node)) {
        Axis* ax = iter->axis;

        if (iter->is_parallel && HAS_OMP) {
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (uint32_t i = 0; i < ax->range; ++i) {
                RuntimeContext local_ctx = ctx;
                // Move pointers for this specific iteration
                if (local_ctx.data_a) local_ctx.data_a += i * ax->stride_a;
                if (local_ctx.data_b) local_ctx.data_b += i * ax->stride_b;
                if (local_ctx.data_c) local_ctx.data_c += i * ax->stride_c;
                if (local_ctx.data_d) local_ctx.data_d += i * ax->stride_d;
                local_ctx.axis_index[ax] = i;
                traverse(iter->body.get(), local_ctx);
            }
        } else {
            for (uint32_t i = 0; i < ax->range; ++i) {
                float* old_a = ctx.data_a; float* old_b = ctx.data_b;
                float* old_c = ctx.data_c; float* old_d = ctx.data_d;

                if (ctx.data_a) ctx.data_a += i * ax->stride_a;
                if (ctx.data_b) ctx.data_b += i * ax->stride_b;
                if (ctx.data_c) ctx.data_c += i * ax->stride_c;
                if (ctx.data_d) ctx.data_d += i * ax->stride_d;
                ctx.axis_index[ax] = i;

                traverse(iter->body.get(), ctx);

                // Restore pointers for next iteration
                ctx.data_a = old_a; ctx.data_b = old_b;
                ctx.data_c = old_c; ctx.data_d = old_d;
            }
        }
    }
    else if (auto* seq = dynamic_cast<Sequence*>(node)) {
        for (auto& child : seq->children) traverse(child.get(), ctx);
    }
    else if (auto* call = dynamic_cast<Invocation*>(node)) {
        // `guard first(@axis)`: skip unless this is that axis's first step
        // (e.g. contraction.teir zeros the accumulator only once, before
        // the K-reduction loop accumulates into it).
        if (call->guard_axis) {
            auto idx_it = ctx.axis_index.find(call->guard_axis);
            if (idx_it != ctx.axis_index.end() && idx_it->second != 0) return;
        }

        if (call->kernel_name == "identity") {
            // trans_b=1 is the course-spec transposing copy (A col-major in,
            // B row-major out) — the Copy primitive's layout flags are
            // derived from the .teir strides by the parser.
            auto kernel = get_unary_kernel(static_cast<uint32_t>(call->m),
                                           static_cast<uint32_t>(call->n),
                                           mini_jit::Unary::ptype_t::identity,
                                           call->trans_b);
            if (kernel) kernel(ctx.data_a, ctx.data_b, call->ld_a, call->ld_b);
        } else if (call->kernel_name == "zero") {
            // Zero writes the output tile ("out" -> pointer slot c). The
            // parser normalized the tile to m runs of n contiguous floats,
            // ld_b apart. Dense tiles (ld == run length, matmul.teir) are
            // one kernel call; interleaved tiles (contraction.teir's q x s
            // tile, where other axes sit between consecutive runs) are
            // zeroed run by run.
            if (call->ld_b == call->n) {
                auto kernel = get_unary_kernel(static_cast<uint32_t>(call->m),
                                               static_cast<uint32_t>(call->n),
                                               mini_jit::Unary::ptype_t::zero, 0u);
                if (kernel) kernel(nullptr, ctx.data_c, call->ld_a, call->ld_b);
            } else {
                auto kernel = get_unary_kernel(1u, static_cast<uint32_t>(call->n),
                                               mini_jit::Unary::ptype_t::zero, 0u);
                if (kernel)
                    for (int64_t r = 0; r < call->m; ++r)
                        kernel(nullptr, ctx.data_c + r * call->ld_b, call->ld_a, call->ld_b);
            }
        } else if (call->kernel_name == "gemm") {
            auto kernel = get_gemm_kernel(static_cast<uint32_t>(call->m),
                                          static_cast<uint32_t>(call->n),
                                          static_cast<uint32_t>(call->k),
                                          call->trans_a, call->trans_b, call->trans_c);
            if (kernel) kernel(ctx.data_a, ctx.data_b, ctx.data_c,
                               call->ld_a, call->ld_b, call->ld_c);
        } else if (call->kernel_name == "rmsnorm" && rmsnorm_kernel) {
            rmsnorm_kernel(ctx.data_a, ctx.data_b, ctx.data_c,
                           call->m, call->n, call->ld_a, call->ld_b, call->eps);
        } else if (call->kernel_name == "layernorm" && layernorm_kernel) {
            layernorm_kernel(ctx.data_a, ctx.data_b, ctx.data_c, ctx.data_d,
                             call->m, call->n, call->ld_a, call->ld_b, call->eps);
        }
    }
}

// The five task builders are now thin wrappers around the real .teir
// files in data/ — load_teir() parses each file's own tensors, axes,
// primitives and schedule (including contraction.teir's `guard first`)
// into the Node tree, rather than reproducing it by hand here.
std::shared_ptr<Node> TeirRuntime::build_transposition_tree() {
    return load_teir("transposition.teir");
}

std::shared_ptr<Node> TeirRuntime::build_matmul_tree() {
    return load_teir("matmul.teir");
}

std::shared_ptr<Node> TeirRuntime::build_contraction_tree() {
    return load_teir("contraction.teir");
}

std::shared_ptr<Node> TeirRuntime::build_rmsnorm_tree() {
    return load_teir("rmsnorm.teir");
}

std::shared_ptr<Node> TeirRuntime::build_layernorm_tree() {
    return load_teir("layernorm.teir");
}

}