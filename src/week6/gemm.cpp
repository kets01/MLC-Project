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

// ---------------------------------------------------------------------------
// Redesigned generator (Sprint 5). What the old one got wrong, found via
// non-uniform-data tests (the original 27-setting all-ones test is blind to
// every addressing bug — any 16 contiguous floats of an all-ones matrix ARE
// the correct operand vector):
//   1. trans_b=0 loaded a contiguous sliding window of B advancing 1 element
//      per k — not a row of B, so not a GEMM for non-uniform data. FMOPA
//      needs B's N-direction contiguous per k-step; col-major B requires a
//      transpose (staged through ZA below), which the old code never did.
//   2. It zeroed ZA per call (C = A*B) although its doc said C += A*B; the
//      TEIR schedules (guarded Zero + accumulation over outer K chunks) and
//      the week3 hand-written kernels both require accumulation. C is now
//      loaded into the ZA accumulators first.
//   3. It used W12 as the MOVA slice index while X12 held the A byte-stride:
//      MOVZ W12 zeroes X12's upper half, so every macro-tile after the first
//      ran with a corrupted stride. Strides now live in x6/x7/x16; x12 is
//      reserved for slice indexing (MOVA's index register must be W12-W15).
//
// Register map:  x0=A x1=B x2=C  x3=ld_a x4=ld_b x5=ld_c (elements)
//   x6=ld_a*4  x7=ld_b*4  x16=ld_c*4   x8=k-chunk counter
//   x9=A ptr   x10=B ptr  x11=C ptr    x12=W12 slice base  x15,x17=scratch
// ZA usage: za2 (+za3) = C accumulators; za0 = A-transpose staging
//   (trans_a=1); za1 = B-transpose staging (trans_b=0, which forces 16-wide
//   n-blocks since only one staging tile is free for B).
// ---------------------------------------------------------------------------

Gemm::error_t Gemm::generate(uint32_t m, uint32_t n, uint32_t k,
                              uint32_t trans_a, uint32_t trans_b, uint32_t trans_c,
                              dtype_t dtype)
{
    if (dtype != dtype_t::fp32) return error_t::err_unsupported_dtype;
    if (m == 0 || n == 0 || k == 0 || m % 16u || n % 16u || k % 16u)
        return error_t::err_shape;

    std::vector<uint32_t> ops;
    ops.reserve(4096);

    const uint32_t esize = 4u;
    const uint32_t step  = 16u;
    const int      lsl   = 2;

    // -----------------------------------------------------------------------
    // Prologue: streaming mode + ZA on; byte strides into x6/x7/x16 (NOT
    // x12-x15 — those are the only legal MOVA slice-index registers, and
    // writing W12 as a slice index would corrupt a stride parked there).
    // -----------------------------------------------------------------------
    ops.push_back(I::sme_smstart_sm());
    ops.push_back(I::sve_ptrue_all(I::p0, I::dtype_t::fp32));
    ops.push_back(I::base_lsl_x(I::x6,  I::x3, (uint32_t)lsl));   // ld_a bytes
    ops.push_back(I::base_lsl_x(I::x7,  I::x4, (uint32_t)lsl));   // ld_b bytes
    ops.push_back(I::base_lsl_x(I::x16, I::x5, (uint32_t)lsl));   // ld_c bytes

    // MOVZ W12 with the slice-group base (fp32 MOVA offsets are 2 bits, so
    // slice s = W12(=s&~3) + offset(s&3); emitted once per group of 4).
    auto slice_group = [&](uint32_t s) {
        if (s % 4u == 0u) ops.push_back(I::base_movz_w(I::w12, s & ~3u));
    };

    // x11 = &C tile (mi, nj); the tile's slice walk advances by ld_c bytes
    // in BOTH layouts (col-major walks columns, row-major walks rows).
    auto emit_c_base = [&](uint32_t mi, uint32_t nj) {
        emit_mov_imm32(ops, 15, (trans_c == 0) ? nj : mi);
        emit_mul(ops, 17, 15, 5);                  // x17 = major * ld_c
        emit_mov_imm32(ops, 15, (trans_c == 0) ? mi : nj);
        emit_add_reg(ops, 17, 17, 15);
        emit_add_lsl(ops, 11, 2, 17, lsl);
    };

    // Move C tile (16 x 16*nb at mi,nj) between memory and za2/za3.
    // load=true: C -> ZA (the accumulate-into-C semantics); false: ZA -> C.
    auto emit_c_phase = [&](uint32_t mi, uint32_t nj, uint32_t nb, bool load) {
        emit_c_base(mi, nj);
        if (trans_c == 0) {
            // col-major: tiles are 16-column runs; za2's 16 columns then za3's
            for (uint32_t t = 0; t < nb; ++t) {
                for (uint32_t s = 0; s < step; ++s) {
                    slice_group(s);
                    if (load) {
                        ops.push_back(I::sve_ld1w_imm(I::z0, I::p0, I::x11, 0));
                        ops.push_back(I::sme_mova_vec_to_tile_v_s(2 + t, 12, s & 3u, I::p0, I::z0));
                    } else {
                        ops.push_back(I::sme_mova_tile_to_vec_v_s(I::z0, I::p0, 2 + t, 12, s & 3u));
                        ops.push_back(I::sve_st1w_imm(I::z0, I::p0, I::x11, 0));
                    }
                    emit_add_reg(ops, 11, 11, 16);
                }
            }
        } else {
            // row-major: each of the 16 rows holds both tiles' vectors
            for (uint32_t s = 0; s < step; ++s) {
                slice_group(s);
                if (load) {
                    ops.push_back(I::sve_ld1w_imm(I::z0, I::p0, I::x11, 0));
                    ops.push_back(I::sme_mova_vec_to_tile_h_s(2, 12, s & 3u, I::p0, I::z0));
                    if (nb == 2) {
                        ops.push_back(I::sve_ld1w_imm(I::z1, I::p0, I::x11, 1));
                        ops.push_back(I::sme_mova_vec_to_tile_h_s(3, 12, s & 3u, I::p0, I::z1));
                    }
                } else {
                    ops.push_back(I::sme_mova_tile_to_vec_h_s(I::z0, I::p0, 2, 12, s & 3u));
                    ops.push_back(I::sve_st1w_imm(I::z0, I::p0, I::x11, 0));
                    if (nb == 2) {
                        ops.push_back(I::sme_mova_tile_to_vec_h_s(I::z1, I::p0, 3, 12, s & 3u));
                        ops.push_back(I::sve_st1w_imm(I::z1, I::p0, I::x11, 1));
                    }
                }
                emit_add_reg(ops, 11, 11, 16);
            }
        }
    };

    // -----------------------------------------------------------------------
    // Macro-tile loops: 16 rows x 16*nb columns per tile. nb=2 normally;
    // trans_b==0 forces nb=1 because za1 is the only free staging tile for
    // B's transpose (za0 may be staging A, za2/za3 hold C).
    // -----------------------------------------------------------------------
    for (uint32_t mi = 0; mi < m; mi += step) {
        for (uint32_t nj = 0; nj < n; ) {
            const uint32_t nb = (trans_b == 0) ? 1u
                                : ((n - nj >= 2 * step) ? 2u : 1u);

            emit_c_phase(mi, nj, nb, /*load=*/true);

            // x9 = &A(mi, k=0): col-major rows are contiguous (constant
            // offset); row-major needs mi*ld_a at runtime.
            if (trans_a == 0) {
                emit_mov_imm32(ops, 15, mi * esize);
                emit_add_reg(ops, 9, 0, 15);
            } else {
                emit_mov_imm32(ops, 15, mi);
                emit_mul(ops, 17, 15, 3);
                emit_add_lsl(ops, 9, 0, 17, lsl);
            }

            // x10 = &B(k=0, nj): mirrored reasoning.
            if (trans_b == 1) {
                emit_mov_imm32(ops, 15, nj * esize);
                emit_add_reg(ops, 10, 1, 15);
            } else {
                emit_mov_imm32(ops, 15, nj);
                emit_mul(ops, 17, 15, 4);
                emit_add_lsl(ops, 10, 1, 17, lsl);
            }

            emit_mov_imm32(ops, 8, k / 16u);       // K-chunk counter

            uint32_t chunk_start = (uint32_t)ops.size();

            if (trans_a == 1) {
                // Stage A chunk: 16 rows (contiguous along K) into za0's
                // horizontal slices; vertical slices then yield the
                // M-direction vectors FMOPA needs. za0[r][kk] = A[mi+r, kc+kk].
                ops.push_back(I::base_mov_reg_x(I::x17, I::x9));
                for (uint32_t r = 0; r < step; ++r) {
                    slice_group(r);
                    ops.push_back(I::sve_ld1w_imm(I::z0, I::p0, I::x17, 0));
                    ops.push_back(I::sme_mova_vec_to_tile_h_s(0, 12, r & 3u, I::p0, I::z0));
                    emit_add_reg(ops, 17, 17, 6);
                }
                emit_add_imm(ops, 9, 9, step * esize);   // next 16 k's
            }
            if (trans_b == 0) {
                // Stage B chunk: 16 columns (contiguous along K) into za1;
                // vertical slice kk = B[kc+kk, nj..nj+15], the N-direction row.
                ops.push_back(I::base_mov_reg_x(I::x17, I::x10));
                for (uint32_t c = 0; c < step; ++c) {
                    slice_group(c);
                    ops.push_back(I::sve_ld1w_imm(I::z2, I::p0, I::x17, 0));
                    ops.push_back(I::sme_mova_vec_to_tile_h_s(1, 12, c & 3u, I::p0, I::z2));
                    emit_add_reg(ops, 17, 17, 7);
                }
                emit_add_imm(ops, 10, 10, step * esize);
            }

            for (uint32_t kk = 0; kk < step; ++kk) {
                if (trans_a == 1 || trans_b == 0) slice_group(kk);

                if (trans_a == 0) {
                    ops.push_back(I::sve_ld1w_imm(I::z0, I::p0, I::x9, 0));
                    emit_add_reg(ops, 9, 9, 6);
                } else {
                    ops.push_back(I::sme_mova_tile_to_vec_v_s(I::z0, I::p0, 0, 12, kk & 3u));
                }

                if (trans_b == 1) {
                    ops.push_back(I::sve_ld1w_imm(I::z2, I::p0, I::x10, 0));
                    if (nb == 2)
                        ops.push_back(I::sve_ld1w_imm(I::z3, I::p0, I::x10, 1));
                    emit_add_reg(ops, 10, 10, 7);
                } else {
                    ops.push_back(I::sme_mova_tile_to_vec_v_s(I::z2, I::p0, 1, 12, kk & 3u));
                }

                ops.push_back(I::sme_fmopa_s(2, I::p0, I::p0, I::z0, I::z2));
                if (nb == 2)
                    ops.push_back(I::sme_fmopa_s(3, I::p0, I::p0, I::z0, I::z3));
            }

            ops.push_back(I::base_subs_imm_x(I::x8, I::x8, 1));
            {
                int32_t offset = (int32_t)chunk_start - (int32_t)ops.size();
                ops.push_back(I::base_b_ne(offset));
            }

            emit_c_phase(mi, nj, nb, /*load=*/false);

            nj += nb * step;
        }
    }

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