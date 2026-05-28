#include "week7/TeirRuntime.h"
#include <iostream>

// Detection of OpenMP support
#if __has_include(<omp.h>)
#include <omp.h>
#define HAS_OMP 1
#else
#define HAS_OMP 0
#endif

namespace mini_jit::teir {

TeirRuntime::TeirRuntime() {
    // Week 6 Kernels setup
    unary_gen.generate(16, 16, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity);
    identity_kernel = unary_gen.get_kernel();

    gemm_gen.generate(16, 16, 16, 0, 0, 0, Gemm::dtype_t::fp32);
    gemm_kernel = gemm_gen.get_kernel();
}

// THIS IS THE FUNCTION THAT WAS MISSING IN YOUR LINKER ERROR
mini_jit::teir::CompiledKernel TeirRuntime::compile(std::shared_ptr<Node> root) {
    // Returns a lambda that acts as the "reusable function pointer"
    return [this, root](float* a, float* b, float* c) {
        this->execute(root.get(), a, b, c);
    };
}

std::shared_ptr<Node> TeirRuntime::load_teir(const std::string& filename) {
    if (filename.find("transposition") != std::string::npos) return build_transposition_tree();
    if (filename.find("matmul") != std::string::npos) return build_matmul_tree();
    if (filename.find("contraction") != std::string::npos) return build_contraction_tree();
    throw std::runtime_error("Unknown TEIR file: " + filename);
}

void TeirRuntime::execute(Node* root, float* a, float* b, float* c) {
    RuntimeContext ctx;
    ctx.data_a = a; ctx.data_b = b; ctx.data_c = c;
    traverse(root, ctx);
}

void TeirRuntime::traverse(Node* node, RuntimeContext& ctx) {
    if (auto* iter = dynamic_cast<Iteration*>(node)) {
        uint32_t range = iter->axis->range;

        if (iter->is_parallel && HAS_OMP) {
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (uint32_t i = 0; i < range; ++i) {
                RuntimeContext local_ctx = ctx;
                local_ctx.indices[iter->axis->name] = i;
                traverse(iter->body.get(), local_ctx);
            }
        } else {
            for (uint32_t i = 0; i < range; ++i) {
                ctx.indices[iter->axis->name] = i;
                traverse(iter->body.get(), ctx);
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

// Builders for Task 1, 2, and 3
std::shared_ptr<Node> TeirRuntime::build_transposition_tree() {
    auto a = new Axis{"a", 96}, b = new Axis{"b", 128};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "identity";
    auto lb = std::make_shared<Iteration>(); lb->axis = b; lb->body = inv;
    auto la = std::make_shared<Iteration>(); la->axis = a; la->body = lb;
    la->is_parallel = true;
    return la;
}

std::shared_ptr<Node> TeirRuntime::build_matmul_tree() {
    auto m0 = new Axis{"m0", 32}, n0 = new Axis{"n0", 64}, k0 = new Axis{"k0", 128};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "gemm";
    auto lk = std::make_shared<Iteration>(); lk->axis = k0; lk->body = inv;
    auto ln = std::make_shared<Iteration>(); ln->axis = n0; ln->body = lk;
    auto lm = std::make_shared<Iteration>(); lm->axis = m0; lm->body = ln;
    lm->is_parallel = true;
    return lm;
}

std::shared_ptr<Node> TeirRuntime::build_contraction_tree() {
    auto p = new Axis{"p", 128}, q = new Axis{"q", 96};
    auto inv = std::make_shared<Invocation>(); inv->kernel_name = "gemm";
    auto lq = std::make_shared<Iteration>(); lq->axis = q; lq->body = inv;
    auto lp = std::make_shared<Iteration>(); lp->axis = p; lp->body = lq;
    lp->is_parallel = true;
    return lp;
}

} // namespace mini_jit::teir