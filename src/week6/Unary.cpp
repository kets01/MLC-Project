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

// trans_b == 1 (B row-major while A is column-major — the transposing
// copy the course's Unary.h always specified but this implementation
// previously ignored): B[j*ld_b + i] = op(A[i + j*ld_a]). The transpose
// is staged through ZA tile 0 per 16x16 sub-tile — contiguous column
// loads of A into horizontal slices, contiguous row stores of B from
// vertical slices (context.md §5's sanctioned ZA use: 2D movement;
// streaming mode has no gather/scatter on this target). This path is
// ld-aware; m and n must be multiples of 16 (course spec).
//
// Register map (trans_b=1): x0=A x1=B x2=ld_a x3=ld_b -> x6/x7 byte
// strides, x9/x10 walking pointers, x12=W12 slice base, x15/x17 scratch.
static void emit_transposed(std::vector<uint32_t>& ops, uint32_t m, uint32_t n,
                            Unary::ptype_t ptype) {
    ops.push_back(I::base_lsl_x(I::x6, I::x2, 2));   // ld_a bytes
    ops.push_back(I::base_lsl_x(I::x7, I::x3, 2));   // ld_b bytes
    if (ptype == Unary::ptype_t::relu)
        ops.push_back(0x25b8c01fu);                  // mov z31.s, #0

    auto mov_imm32 = [&](int reg, uint32_t val) {
        I::gpr_t rd = (I::gpr_t)(I::x0 + reg);
        ops.push_back(I::base_movz_x(rd, val & 0xFFFFu));
        if (val > 0xFFFFu)
            ops.push_back(I::base_movk_x(rd, (val >> 16) & 0xFFFFu, 1));
    };

    for (uint32_t mi = 0; mi < m; mi += 16u) {
        for (uint32_t nj = 0; nj < n; nj += 16u) {
            // x9 = A + (nj*ld_a + mi)*4 — A col-major: column nj+c starts
            // ld_a apart, rows mi..mi+15 contiguous within it.
            mov_imm32(15, nj);
            ops.push_back(I::base_mul_x(I::x17, I::x15, I::x2));
            mov_imm32(15, mi);
            ops.push_back(I::base_add_reg_x(I::x17, I::x17, I::x15));
            ops.push_back(I::base_add_lsl_x(I::x9, I::x0, I::x17, 2));

            // stage: columns of A into za0's horizontal slices
            for (uint32_t c = 0; c < 16u; ++c) {
                if (c % 4u == 0u) ops.push_back(I::base_movz_w(I::w12, c & ~3u));
                ops.push_back(I::sve_ld1w_imm(I::z0, I::p0, I::x9, 0));
                ops.push_back(I::sme_mova_vec_to_tile_h_s(0, 12, c & 3u, I::p0, I::z0));
                ops.push_back(I::base_add_reg_x(I::x9, I::x9, I::x6));
            }

            // x10 = B + (mi*ld_b + nj)*4 — B row-major: row mi+r starts
            // ld_b apart, columns nj..nj+15 contiguous within it.
            mov_imm32(15, mi);
            ops.push_back(I::base_mul_x(I::x17, I::x15, I::x3));
            mov_imm32(15, nj);
            ops.push_back(I::base_add_reg_x(I::x17, I::x17, I::x15));
            ops.push_back(I::base_add_lsl_x(I::x10, I::x1, I::x17, 2));

            // drain: vertical slice r = A[mi+r, nj..nj+15] = row mi+r of B
            for (uint32_t r = 0; r < 16u; ++r) {
                if (r % 4u == 0u) ops.push_back(I::base_movz_w(I::w12, r & ~3u));
                ops.push_back(I::sme_mova_tile_to_vec_v_s(I::z0, I::p0, 0, 12, r & 3u));
                if (ptype == Unary::ptype_t::relu)
                    ops.push_back(I::sve_fmax_s(I::z0, I::p0, I::z31));
                ops.push_back(I::sve_st1w_imm(I::z0, I::p0, I::x10, 0));
                ops.push_back(I::base_add_reg_x(I::x10, I::x10, I::x7));
            }
        }
    }
}

Unary::error_t Unary::generate(uint32_t m, uint32_t n,
                               uint32_t trans_b,
                               [[maybe_unused]] dtype_t dtype,
                               ptype_t ptype) {

    std::vector<uint32_t> ops;

    // ------------------------------------------------------------------
    // Prologue: enter streaming SVE mode, set all-true predicate
    // ------------------------------------------------------------------
    ops.push_back(I::sme_smstart_sm());
    ops.push_back(I::sve_ptrue_all(I::p0, I::dtype_t::fp32));

    // trans_b=1: transposing identity/relu (B row-major). Zero fills B
    // without reading A, so its trans_b=1 form is the same dense fill.
    if (trans_b == 1 && ptype != ptype_t::zero) {
        emit_transposed(ops, m, n, ptype);
        ops.push_back(I::sme_smstop_sm());
        ops.push_back(I::base_br_ret());
        kernel_registry[this] = JitEngine::generate<kernel_t>(ops);
        return error_t::success;
    }

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