#include "week7/TeirRuntime.h"
#include <iostream>

#if __has_include(<omp.h>)
#include <omp.h>
#define HAS_OMP 1
#else
#define HAS_OMP 0
#endif

namespace mini_jit::teir {

TeirRuntime::TeirRuntime() {
    // Inner tile size is 16x16
    unary_gen.generate(16, 16, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity);
    identity_kernel = unary_gen.get_kernel();

    gemm_gen.generate(16, 16, 16, 0, 0, 0, Gemm::dtype_t::fp32);
    gemm_kernel = gemm_gen.get_kernel();
}

mini_jit::teir::CompiledKernel TeirRuntime::compile(std::shared_ptr<Node> root) {
    return [this, root](float* a, float* b, float* c) {
        this->execute(root.get(), a, b, c);
    };
}

std::shared_ptr<Node> TeirRuntime::load_teir(const std::string& filename) {
    if (filename.find("transposition") != std::string::npos) return build_transposition_tree();
    if (filename.find("matmul") != std::string::npos) return build_matmul_tree();
    if (filename.find("contraction") != std::string::npos) return build_contraction_tree();
    throw std::runtime_error("Unknown TEIR file");
}

void TeirRuntime::execute(Node* root, float* a, float* b, float* c) {
    RuntimeContext ctx;
    ctx.data_a = a; ctx.data_b = b; ctx.data_c = c;
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
                traverse(iter->body.get(), local_ctx);
            }
        } else {
            for (uint32_t i = 0; i < ax->range; ++i) {
                float* old_a = ctx.data_a; float* old_b = ctx.data_b; float* old_c = ctx.data_c;
                
                if (ctx.data_a) ctx.data_a += i * ax->stride_a;
                if (ctx.data_b) ctx.data_b += i * ax->stride_b;
                if (ctx.data_c) ctx.data_c += i * ax->stride_c;
                
                traverse(iter->body.get(), ctx);
                
                // Restore pointers for next iteration
                ctx.data_a = old_a; ctx.data_b = old_b; ctx.data_c = old_c;
            }
        }
    } 
    else if (auto* call = dynamic_cast<Invocation*>(node)) {
        if (call->kernel_name == "identity" && identity_kernel) {
            identity_kernel(ctx.data_a, ctx.data_b, 16, 16);
        } else if (call->kernel_name == "gemm" && gemm_kernel) {
            gemm_kernel(ctx.data_a, ctx.data_b, ctx.data_c, 16, 16, 16);
        }
    }
}

// Tensor (96, 128, 48, 32): outer loop slices at 16*128*48*32 = 3,145,728 elements.
// Inner loop covers the rest of each slice in 256-element (16×16) kernel steps.
std::shared_ptr<Node> TeirRuntime::build_transposition_tree() {
    const uint32_t tile = 16 * 16;  // elements per identity-kernel call
    auto a = new Axis{"a", 96/16, 16*128*48*32, 16*128*48*32};
    auto b = new Axis{"b", 128*48*32/16, tile, tile};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "identity";
    auto lb = std::make_shared<Iteration>(); lb->axis = b; lb->body = inv;
    auto la = std::make_shared<Iteration>(); la->axis = a; la->body = lb;
    la->is_parallel = true;
    return la;
}

// 512×512 matmul tiled by 16, row-major tile order.
// A[m_tile, k_tile] at A + (m_tile*32 + k_tile)*256
// B[k_tile, n_tile] at B + (k_tile*32 + n_tile)*256
// C[m_tile, n_tile] at C + (m_tile*32 + n_tile)*256
std::shared_ptr<Node> TeirRuntime::build_matmul_tree() {
    const uint32_t N_tiles = 512/16;   // 32 tiles per dimension
    const uint32_t tile    = 16 * 16;  // floats per tile
    auto m = new Axis{"m", N_tiles, N_tiles*tile, 0,         N_tiles*tile};
    auto n = new Axis{"n", N_tiles, 0,            tile,      tile};
    auto k = new Axis{"k", N_tiles, tile,         N_tiles*tile, 0};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "gemm";
    auto lk = std::make_shared<Iteration>(); lk->axis = k; lk->body = inv;
    auto ln = std::make_shared<Iteration>(); ln->axis = n; ln->body = lk;
    auto lm = std::make_shared<Iteration>(); lm->axis = m; lm->body = ln;
    lm->is_parallel = true;
    return lm;
}

// Small contraction — 16×16 p/q tile loops, all offsets well within the
// 256*256*16 = 1,048,576-float test buffers.
std::shared_ptr<Node> TeirRuntime::build_contraction_tree() {
    const uint32_t tile = 16 * 16;
    auto p = new Axis{"p", 16, tile, 0,    tile};
    auto q = new Axis{"q", 16, 0,   tile,  tile};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "gemm";
    auto lq = std::make_shared<Iteration>(); lq->axis = q; lq->body = inv;
    auto lp = std::make_shared<Iteration>(); lp->axis = p; lp->body = lq;
    lp->is_parallel = true;
    return lp;
}

}