#include "week6/gemm.hpp"
#include "week5/jit_engine.hpp"
#include "week6/Instgen.hpp"
#include <vector>

using namespace mini_jit;
using I = mini_jit::InstGen;

// ---------------------------------------------------------------------------
// Register allocation (AArch64 calling convention + SME constraints)
//
//  X0  = A        X1  = B        X2  = C
//  X3  = ld_a     X4  = ld_b     X5  = ld_c
//
//  X9  = ptr_a0   X10 = ptr_b0
//  X11 = ptr_b1 / ptr_c (store phase)
//  X12 = stride_a_bytes  (k-loop) / W12 = ZA slice index (store phase)
//  X13 = stride_b_bytes  X14 = stride_c_bytes
//  X15 = scratch    X16 = ptr_a1     X17 = scratch
//  X8  = k loop counter
// ---------------------------------------------------------------------------

// Emit MOVZ + optional MOVK to load a 32-bit immediate into X<reg>.
static void emit_mov_imm32(std::vector<uint32_t>& ops, int reg, uint32_t val)
{
    I::gpr_t rd = (I::gpr_t)(I::x0 + reg);
    ops.push_back(I::base_movz_x(rd, val & 0xFFFFu));
    if (val > 0xFFFFu)
        ops.push_back(I::base_movk_x(rd, (val >> 16) & 0xFFFFu, 1));
}

// ADD X<dst>, X<src>, X<idx>, LSL #shift
static void emit_add_lsl(std::vector<uint32_t>& ops, int dst, int src, int idx, int shift)
{
    I::gpr_t rd = (I::gpr_t)(I::x0 + dst);
    I::gpr_t rn = (I::gpr_t)(I::x0 + src);
    I::gpr_t rm = (I::gpr_t)(I::x0 + idx);
    ops.push_back(I::base_add_lsl_x(rd, rn, rm, (uint32_t)shift));
}

// ADD X<dst>, X<src>, #imm12
static void emit_add_imm(std::vector<uint32_t>& ops, int dst, int src, uint32_t imm12)
{
    I::gpr_t rd = (I::gpr_t)(I::x0 + dst);
    I::gpr_t rn = (I::gpr_t)(I::x0 + src);
    ops.push_back(I::base_add_imm_x(rd, rn, imm12));
}

// MUL X<dst>, X<a>, X<b>
static void emit_mul(std::vector<uint32_t>& ops, int dst, int a, int b)
{
    I::gpr_t rd = (I::gpr_t)(I::x0 + dst);
    I::gpr_t rn = (I::gpr_t)(I::x0 + a);
    I::gpr_t rm = (I::gpr_t)(I::x0 + b);
    ops.push_back(I::base_mul_x(rd, rn, rm));
}

// ADD X<dst>, X<a>, X<b>  (no shift)
static void emit_add_reg(std::vector<uint32_t>& ops, int dst, int a, int b)
{
    I::gpr_t rd = (I::gpr_t)(I::x0 + dst);
    I::gpr_t rn = (I::gpr_t)(I::x0 + a);
    I::gpr_t rm = (I::gpr_t)(I::x0 + b);
    ops.push_back(I::base_add_reg_x(rd, rn, rm));
}

Gemm::error_t Gemm::generate(uint32_t m, uint32_t n, uint32_t k,
                              uint32_t trans_a, uint32_t trans_b, uint32_t trans_c,
                              dtype_t dtype)
{
    std::vector<uint32_t> ops;
    ops.reserve(512);

    const uint32_t esize = (dtype == dtype_t::fp32) ? 4u : 8u;
    const uint32_t step  = (dtype == dtype_t::fp32) ? 16u : 8u;
    const int      lsl   = (dtype == dtype_t::fp32) ? 2 : 3;

    I::dtype_t idt = (dtype == dtype_t::fp32) ? I::dtype_t::fp32 : I::dtype_t::fp64;

    // -----------------------------------------------------------------------
    // Prologue
    // -----------------------------------------------------------------------
    ops.push_back(I::sme_smstart_sm());
    ops.push_back(I::sve_ptrue_all(I::p0, idt));

    // LSL X12, X3, #lsl  ;  LSL X13, X4, #lsl  ;  LSL X14, X5, #lsl
    ops.push_back(I::base_lsl_x(I::x12, I::x3, (uint32_t)lsl));
    ops.push_back(I::base_lsl_x(I::x13, I::x4, (uint32_t)lsl));
    ops.push_back(I::base_lsl_x(I::x14, I::x5, (uint32_t)lsl));

    // -----------------------------------------------------------------------
    // Tile loops — 2×2 blocking
    // -----------------------------------------------------------------------
    const uint32_t tile2 = 2 * step;

    for (uint32_t j = 0; j < n; j += tile2) {
        for (uint32_t i = 0; i < m; i += tile2) {

            // Zero all four ZA tiles
            ops.push_back(I::sme_zero_za());

            // X9 = X0 + i*esize  (ptr_a0)
            emit_mov_imm32(ops, 15, i * esize);
            emit_add_reg(ops, 9, 0, 15);

            // X10 = X1 + j*ld_b*esize  (ptr_b0)
            emit_mov_imm32(ops, 15, j);
            emit_mul(ops, 17, 15, 4);              // X17 = j * ld_b
            emit_add_lsl(ops, 10, 1, 17, lsl);

            // X16 = X9 + step*esize  (ptr_a1)
            emit_add_imm(ops, 16, 9, step * esize);

            // X11 = X10 + step*ld_b*esize  (ptr_b1)
            emit_mov_imm32(ops, 15, step);
            emit_mul(ops, 17, 15, 4);              // X17 = step * ld_b
            emit_add_lsl(ops, 11, 10, 17, lsl);

            // MOV X8, #k
            emit_mov_imm32(ops, 8, k);

            uint32_t loop_start = (uint32_t)ops.size();

            // Load A vectors
            if (dtype == dtype_t::fp32) {
                ops.push_back(I::sve_ld1w_scalar(I::z0, I::p0, I::x9));
                ops.push_back(I::sve_ld1w_scalar(I::z1, I::p0, I::x16));
            } else {
                ops.push_back(I::sve_ld1d_scalar(I::z0, I::p0, I::x9));
                ops.push_back(I::sve_ld1d_scalar(I::z1, I::p0, I::x16));
            }

            // Load B vectors
            if (dtype == dtype_t::fp32) {
                ops.push_back(I::sve_ld1w_scalar(I::z2, I::p0, I::x10));
                ops.push_back(I::sve_ld1w_scalar(I::z3, I::p0, I::x11));
            } else {
                ops.push_back(I::sve_ld1d_scalar(I::z2, I::p0, I::x10));
                ops.push_back(I::sve_ld1d_scalar(I::z3, I::p0, I::x11));
            }

            // Outer products
            if (dtype == dtype_t::fp32) {
                ops.push_back(I::sme_fmopa_s(0, I::p0, I::p0, I::z0, I::z2));
                ops.push_back(I::sme_fmopa_s(1, I::p0, I::p0, I::z0, I::z3));
                ops.push_back(I::sme_fmopa_s(2, I::p0, I::p0, I::z1, I::z2));
                ops.push_back(I::sme_fmopa_s(3, I::p0, I::p0, I::z1, I::z3));
            } else {
                ops.push_back(I::sme_fmopa_d(0, I::p0, I::p0, I::z0, I::z2));
                ops.push_back(I::sme_fmopa_d(1, I::p0, I::p0, I::z0, I::z3));
                ops.push_back(I::sme_fmopa_d(2, I::p0, I::p0, I::z1, I::z2));
                ops.push_back(I::sme_fmopa_d(3, I::p0, I::p0, I::z1, I::z3));
            }

            // Advance A pointers by one K step
            if (trans_a == 0) {
                emit_add_reg(ops, 9,  9,  12);   // ADD X9,  X9,  X12
                emit_add_reg(ops, 16, 16, 12);   // ADD X16, X16, X12
            } else {
                emit_add_imm(ops, 9,  9,  esize);
                emit_add_imm(ops, 16, 16, esize);
            }

            // Advance B pointers by one K step
            if (trans_b == 0) {
                emit_add_imm(ops, 10, 10, esize);
                emit_add_imm(ops, 11, 11, esize);
            } else {
                emit_add_reg(ops, 10, 10, 13);   // ADD X10, X10, X13
                emit_add_reg(ops, 11, 11, 13);   // ADD X11, X11, X13
            }

            // SUBS X8, X8, #1
            ops.push_back(I::base_subs_imm_x(I::x8, I::x8, 1));

            // B.NE loop_start
            {
                int32_t offset = (int32_t)loop_start - (int32_t)ops.size();
                ops.push_back(I::base_b_ne(offset));
            }

            // ----------------------------------------------------------------
            // Store phase
            // ----------------------------------------------------------------
            for (int ti = 0; ti < 2; ++ti) {
                for (int tj = 0; tj < 2; ++tj) {
                    int za_idx = tj + ti * 2;

                    uint32_t row_off = i + (uint32_t)ti * step;
                    uint32_t col_off = j + (uint32_t)tj * step;

                    // X11 = X2 + (col_off * ld_c + row_off) * esize
                    emit_mov_imm32(ops, 15, col_off);
                    emit_mul(ops, 17, 15, 5);            // X17 = col_off * ld_c
                    emit_add_imm(ops, 17, 17, row_off);
                    emit_add_lsl(ops, 11, 2, 17, lsl);

                    for (uint32_t s = 0; s < step; ++s) {
                        // MOVZ W12, #s
                        ops.push_back(I::base_movz_w(I::w12, s));

                        if (trans_c == 0) {
                            // MOV Z0.S, P0/M, ZA<n>V.S[W12, 0]
                            ops.push_back(I::sme_mova_tile_to_vec_v_s(I::z0, I::p0, (uint32_t)za_idx, 12));
                            // ST1W {Z0.S}, P0, [X11]
                            ops.push_back(I::sve_st1w_scalar(I::z0, I::p0, I::x11));
                            // ADD X11, X11, X14
                            emit_add_reg(ops, 11, 11, 14);
                        } else {
                            // MOV Z0.S, P0/M, ZA<n>H.S[W12, 0]
                            ops.push_back(I::sme_mova_tile_to_vec_h_s(I::z0, I::p0, (uint32_t)za_idx, 12));
                            // ST1W {Z0.S}, P0, [X11]
                            ops.push_back(I::sve_st1w_scalar(I::z0, I::p0, I::x11));
                            // ADD X11, X11, #esize
                            emit_add_imm(ops, 11, 11, esize);
                        }
                    }
                }
            }

        } // i
    } // j

    // -----------------------------------------------------------------------
    // Epilogue
    // -----------------------------------------------------------------------
    ops.push_back(I::sme_smstop_sm());
    ops.push_back(I::base_br_ret());

    m_kernel = JitEngine::generate<kernel_t>(ops);
    return error_t::success;
}

Gemm::kernel_t Gemm::get_kernel() const {
    return m_kernel;
}