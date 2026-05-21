#include "week6/Unary.h"
#include "week5/jit_engine.hpp"
#include <vector>
#include <unordered_map>

namespace mini_jit {

/**
 * Internal registry to store kernel pointers.
 * Since Unary.h cannot be modified to include a data member,
 * we use the address of the 'Unary' instance as a key.
 */
static std::unordered_map<const Unary*, Unary::kernel_t> kernel_registry;

Unary::error_t Unary::generate(uint32_t m, uint32_t n, 
                               [[maybe_unused]] uint32_t trans_b, 
                               [[maybe_unused]] dtype_t dtype, 
                               ptype_t ptype) {
    
    std::vector<uint32_t> opcodes;

    // Lambda to emit instructions into the buffer
    auto emit = [&](uint32_t inst) { opcodes.push_back(inst); };

    // smstart sm: Enter Streaming SVE mode
    emit(0xd503437f); 
    // ptrue p0.s, vl16: Set predicate for 16-lane (512-bit) operation
    emit(0x2598e120); 

    // Setup registers for specific primitives
    if (ptype == ptype_t::zero) {
        emit(0x25b8c000); // mov z0.s, #0
    } else if (ptype == ptype_t::relu) {
        emit(0x25b8c01f); // mov z31.s, #0 (Zero constant for comparison)
    }

    // Process data in blocks of 16 floats (64 bytes)
    uint32_t total_elements = m * n;
    uint32_t num_vectors = total_elements / 16;

    for (uint32_t i = 0; i < num_vectors; ++i) {
        // Load block from matrix A (skip for zero primitive)
        if (ptype != ptype_t::zero) {
            emit(0xa540a000); // ld1w {z0.s}, p0/z, [x0]
            emit(0x91010000); // add x0, x0, #64 (Increment A pointer)
        }

        // Apply computation (ReLU)
        if (ptype == ptype_t::relu) {
            emit(0x658683e0); // fmax z0.s, p0/m, z0.s, z31.s
        }

        // Store block to matrix B
        emit(0xe540e020); // st1w {z0.s}, p0, [x1]
        emit(0x91010021); // add x1, x1, #64 (Increment B pointer)
    }

    emit(0xd503427f); // smstop sm: Exit Streaming mode
    emit(0xd65f03c0); // ret: Return to caller

    // Generate the machine code and store it in the registry using 'this' as the key
    kernel_registry[this] = JitEngine::generate<kernel_t>(opcodes);
    
    return error_t::success;
}

Unary::kernel_t Unary::get_kernel() const {
    // Look up the kernel associated with this specific instance
    auto it = kernel_registry.find(this);
    if (it != kernel_registry.end()) {
        return it->second;
    }
    return nullptr; // Return nullptr if generate() hasn't been called yet
}

} // namespace mini_jit