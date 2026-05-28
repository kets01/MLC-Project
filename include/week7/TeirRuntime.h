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
    
    // Compiles a TEIR tree into a reusable function pointer
    mini_jit::teir::CompiledKernel compile(std::shared_ptr<Node> root);

    // Loads a TEIR object based on filename (Internal mapping)
    std::shared_ptr<Node> load_teir(const std::string& filename);

    // Direct execution of the TEIR tree
    void execute(Node* root, float* a, float* b, float* c);

    // Task-specific tree builders
    std::shared_ptr<Node> build_transposition_tree();
    std::shared_ptr<Node> build_matmul_tree();
    std::shared_ptr<Node> build_contraction_tree();

private:
    // Recursive function to generate the loop nest
    void traverse(Node* node, RuntimeContext& ctx);
    
    // Week 6 generators and kernels
    mini_jit::Unary unary_gen;
    mini_jit::Gemm gemm_gen;
    mini_jit::Unary::kernel_t identity_kernel;
    mini_jit::Gemm::kernel_t gemm_kernel;
};

} 
#endif