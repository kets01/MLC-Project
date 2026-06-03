#include "week6/unary.hpp"
#include "week5/jit_engine.hpp"
#include "week6/Instgen.hpp"
#include <vector>
#include <unordered_map>

namespace mini_jit {

using I = mini_jit::InstGen;

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

    std::vector<uint32_t> ops;

    // ------------------------------------------------------------------
    // Prologue: enter streaming SVE mode, set all-true predicate
    // ------------------------------------------------------------------
    ops.push_back(I::sme_smstart_sm());
    ops.push_back(I::sve_ptrue_all(I::p0, I::dtype_t::fp32));

    // For zero we pre-load a zero register; for relu we load a zero
    // constant into Z31 for the fmax comparison.
    if (ptype == ptype_t::zero) {
        // MOV Z0.S, #0  — SVE broadcast immediate
        // FDUP Z0.S, #0.0 = 0x2598C000 | imm8=0 | Zd=0
        // Use the hardcoded 0 broadcast (same as original)
        ops.push_back(0x25b8c000u); // mov z0.s, #0
    } else if (ptype == ptype_t::relu) {
        // Load zero constant into Z31 for ReLU comparison
        ops.push_back(0x25b8c01fu); // mov z31.s, #0
    }

    // ------------------------------------------------------------------
    // Process data in blocks of 16 fp32 values (512-bit SVL)
    // ------------------------------------------------------------------
    const uint32_t block_bytes = 64u;   // 16 * 4
    uint32_t total_elements = m * n;
    uint32_t num_vectors    = total_elements / 16u;

    for (uint32_t i = 0; i < num_vectors; ++i) {
        if (ptype != ptype_t::zero) {
            // Load from A (X0)
            ops.push_back(I::sve_ld1w_scalar(I::z0, I::p0, I::x0));
            // ADD X0, X0, #64
            ops.push_back(I::base_add_imm_x(I::x0, I::x0, block_bytes));
        }

        if (ptype == ptype_t::relu) {
            // FMAX Z0.S, P0/M, Z0.S, Z31.S  — clamp to zero
            ops.push_back(I::sve_fmax_s(I::z0, I::p0, I::z31));
        }

        // Store to B (X1)
        ops.push_back(I::sve_st1w_scalar(I::z0, I::p0, I::x1));
        // ADD X1, X1, #64
        ops.push_back(I::base_add_imm_x(I::x1, I::x1, block_bytes));
    }

    // ------------------------------------------------------------------
    // Epilogue
    // ------------------------------------------------------------------
    ops.push_back(I::sme_smstop_sm());
    ops.push_back(I::base_br_ret());

    kernel_registry[this] = JitEngine::generate<kernel_t>(ops);
    return error_t::success;
}

Unary::kernel_t Unary::get_kernel() const {
    auto it = kernel_registry.find(this);
    if (it != kernel_registry.end())
        return it->second;
    return nullptr;
}

} // namespace mini_jit