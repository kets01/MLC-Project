#ifndef MINI_JIT_TEIR_RUNTIME_H
#define MINI_JIT_TEIR_RUNTIME_H

#include "Teir.h"
#include "week6/Unary.h"
#include "week6/gemm.hpp"
#include <memory>
#include <string>

namespace mini_jit::teir {

class TeirRuntime {
public:
    TeirRuntime();
    
    // Requirement: Compile TEIR into a reusable function pointer
    mini_jit::teir::CompiledKernel compile(std::shared_ptr<Node> root);

    // Requirement: Take TEIR object (from file)
    std::shared_ptr<Node> load_teir(const std::string& filename);

    // Helper for direct execution
    void execute(Node* root, float* a, float* b, float* c);

    // Tree builders for the 3 tasks
    std::shared_ptr<Node> build_transposition_tree();
    std::shared_ptr<Node> build_matmul_tree();
    std::shared_ptr<Node> build_contraction_tree();

private:
    void traverse(Node* node, RuntimeContext& ctx);
    
    mini_jit::Unary unary_gen;
    mini_jit::Gemm gemm_gen;
    mini_jit::Unary::kernel_t identity_kernel;
    mini_jit::Gemm::kernel_t gemm_kernel;
};

} 
#endif