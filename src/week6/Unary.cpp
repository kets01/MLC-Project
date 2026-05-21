#include "week6/Unary.h"
#include "week5/jit_engine.hpp"
#include <vector>

namespace mini_jit {

// Internal implementation of the Unary class
class UnaryImpl : public Unary {
private:
    kernel_t kernel_ptr = nullptr;
    std::vector<uint32_t> opcodes;

public:
    error_t generate(uint32_t m, uint32_t n, uint32_t trans_b, dtype_t dtype, ptype_t ptype) override {
        opcodes.clear();

        // Helper to push instructions into the buffer
        auto emit = [&](uint32_t inst) { opcodes.push_back(inst); };

        
        // Enable Streaming Mode (SM) for SVE instructions
        emit(0xd503437f); // smstart sm
        // ptrue p0.s, vl16: sets the first 16 lanes to true (for 512-bit vector)
        emit(0x2598e120); 

        // Initial setup based on the primitive type
        if (ptype == ptype_t::zero) {
            emit(0x25b8c000); // mov z0.s, #0 (Initialize zero vector)
        } else if (ptype == ptype_t::relu) {
            emit(0x25b8c01f); // mov z31.s, #0 (Constant zero for fmax/ReLU)
        }

        // We assume M and N are multiples of 16.
        // Total iterations: (M * N) / 16 vectors
        uint32_t total_elements = m * n;
        uint32_t num_vectors = total_elements / 16;

        for (uint32_t i = 0; i < num_vectors; ++i) {
            // LOAD: Only if not Zero primitive
            if (ptype != ptype_t::zero) {
                // ld1w {z0.s}, p0/z, [x0]: load 16 floats from address in x0
                emit(0xa540a000); 
                // add x0, x0, #64: move A pointer (16 floats * 4 bytes = 64)
                emit(0x91010000); 
            }

            // COMPUTE: Apply ReLU if requested
            if (ptype == ptype_t::relu) {
                // fmax z0.s, p0/m, z0.s, z31.s: z0 = max(z0, 0)
                emit(0x658683e0); 
            }

            // STORE: Write result to matrix B
            // st1w {z0.s}, p0, [x1]: store 16 floats to address in x1
            emit(0xe540e020); 
            // add x1, x1, #64: move B pointer
            emit(0x91010021); 
        }

        // Disable Streaming Mode
        emit(0xd503427f); // smstop sm
        // Return from function
        emit(0xd65f03c0); // ret

        // Convert the opcode vector into an executable function pointer
        kernel_ptr = JitEngine::generate<kernel_t>(opcodes);
        
        return error_t::success;
    }

    kernel_t get_kernel() const override {
        return kernel_ptr;
    }
};

} // namespace mini_jit