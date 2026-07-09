#include "norm/jit_norm.hpp"
#include "week5/jit_engine.hpp"
#include "week6/Instgen.hpp"

// mini_jit::Norm — emits the Sprint-2 SSVE V6 winners at runtime.
//
// The emission below is a 1:1 transcription of the hand-written kernels
// (src/norm/rms_norm_ssve_v6.S / layer_norm_ssve_v6.S), same registers,
// same instruction order.  That is deliberate: the encoding-diff test
// proves the buffer byte-identical to a kernel that already passed the
// full verification suite, so the generator inherits its trust instead
// of re-earning it instruction by instruction.
//
// Branches: backward targets are recorded as word indices when reached
// (label = m_ops.size()); forward targets are emitted as placeholders
// and backpatched once the target index is known.  Offsets are in
// instructions, relative to the branch word itself.

namespace mini_jit {

using I = InstGen;

// --- small emission helpers -------------------------------------------------

namespace {

// backward branch offset: target label -> signed offset from the word
// ABOUT to be emitted (i.e. ops.size() is the branch's own index).
inline int32_t back( const std::vector<uint32_t>& ops, size_t label ) {
    return (int32_t)label - (int32_t)ops.size();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// RMSNorm V6 — see rms_norm_ssve_v6.S for the full design commentary.
//   x19=a x20=b x21=gamma x22=m x23=n x24=stride_a x25=stride_b
// ---------------------------------------------------------------------------
void Norm::emit_rms_v6() {
    std::vector<uint32_t>& ops = m_ops;

    // Prologue: callee-saved regs + the dedicated eps stash slot [sp,#64]
    // (eps must live in MEMORY across smstart — no fp register survives
    // the streaming-mode transition; the Sprint-3 eps bugfix).
    ops.push_back( I::base_stp_pre_x( I::x19, I::x20, I::sp, -80 ) );
    ops.push_back( I::base_stp_off_x( I::x21, I::x22, I::sp,  16 ) );
    ops.push_back( I::base_stp_off_x( I::x23, I::x24, I::sp,  32 ) );
    ops.push_back( I::base_str_imm_x( I::x25, I::sp, 48 ) );
    ops.push_back( I::simd_str_imm_d( I::v8,  I::sp, 56 ) );

    ops.push_back( I::base_mov_reg_x( I::x19, I::x0 ) );   // a
    ops.push_back( I::base_mov_reg_x( I::x20, I::x1 ) );   // b
    ops.push_back( I::base_mov_reg_x( I::x21, I::x2 ) );   // gamma
    ops.push_back( I::base_mov_reg_x( I::x22, I::x3 ) );   // m
    ops.push_back( I::base_mov_reg_x( I::x23, I::x4 ) );   // n
    ops.push_back( I::base_lsl_x( I::x24, I::x5, 2 ) );    // stride_a bytes
    ops.push_back( I::base_lsl_x( I::x25, I::x6, 2 ) );    // stride_b bytes
    ops.push_back( I::fp_fmov_s( I::v8, I::v0 ) );
    ops.push_back( I::simd_str_imm_s( I::v0, I::sp, 64 ) ); // eps stash

    size_t fix_cbz = ops.size();                            // cbz m, row_done
    ops.push_back( 0 );                                     // backpatched

    ops.push_back( I::sme_smstart_sm_only() );
    ops.push_back( I::sve_ptrue_all( I::p0, I::dtype_t::fp32 ) );

    ops.push_back( I::simd_ldr_imm_s( I::v0, I::sp, 64 ) ); // eps reload
    ops.push_back( I::sve_dup_elem_s( I::z8, I::z0, 0 ) );  // z8 = {eps,...}

    ops.push_back( I::fp_fmov_imm_s( I::v0, 0x70 ) );       // s0 = 1.0
    ops.push_back( I::fp_scvtf_s_x( I::v1, I::x23 ) );      // s1 = (float)n
    ops.push_back( I::fp_fdiv_s( I::v0, I::v0, I::v1 ) );
    ops.push_back( I::sve_dup_elem_s( I::z5, I::z0, 0 ) );  // z5 = {1/N,...}

    ops.push_back( I::sve_cntw_x( I::x14 ) );
    ops.push_back( I::base_lsl_x( I::x14, I::x14, 2 ) );    // 4*VL rows/group
    ops.push_back( I::base_movz_x( I::x0, 0 ) );            // row index

    // --- group loop: four full VL-row blocks per iteration ---
    size_t group_loop = ops.size();
    ops.push_back( I::base_add_reg_x( I::x6, I::x0, I::x14 ) );
    ops.push_back( I::base_cmp_reg_x( I::x6, I::x22 ) );
    size_t fix_bhi = ops.size();                            // b.hi tail_blocks
    ops.push_back( 0 );

    // pass 1: sum of squares, one accumulator per row block
    ops.push_back( I::sve_dup_imm_s( I::z2,  0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z16, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z17, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z18, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x23 ) );

    size_t red_col = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0,  I::p0, I::x8, 0 ) );
    ops.push_back( I::sve_ld1w_imm( I::z24, I::p0, I::x8, 1 ) );
    ops.push_back( I::sve_ld1w_imm( I::z25, I::p0, I::x8, 2 ) );
    ops.push_back( I::sve_ld1w_imm( I::z26, I::p0, I::x8, 3 ) );
    ops.push_back( I::sve_fmla_s( I::z2,  I::p0, I::z0,  I::z0 ) );
    ops.push_back( I::sve_fmla_s( I::z16, I::p0, I::z24, I::z24 ) );
    ops.push_back( I::sve_fmla_s( I::z17, I::p0, I::z25, I::z25 ) );
    ops.push_back( I::sve_fmla_s( I::z18, I::p0, I::z26, I::z26 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x24 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, red_col ) ) );

    // inv_rms for the four blocks (FRSQRTE + one Newton-Raphson step)
    ops.push_back( I::sve_fmul_p_s( I::z2,  I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z16, I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z17, I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z18, I::p0, I::z5 ) );
    ops.push_back( I::sve_fadd_p_s( I::z2,  I::p0, I::z8 ) );
    ops.push_back( I::sve_fadd_p_s( I::z16, I::p0, I::z8 ) );
    ops.push_back( I::sve_fadd_p_s( I::z17, I::p0, I::z8 ) );
    ops.push_back( I::sve_fadd_p_s( I::z18, I::p0, I::z8 ) );

    const I::sve_t acc[4]  = { I::z2,  I::z16, I::z17, I::z18 };
    const I::sve_t inv[4]  = { I::z4,  I::z19, I::z20, I::z21 };
    for ( int i = 0; i < 4; ++i ) {
        ops.push_back( I::sve_frsqrte_s( inv[i], acc[i] ) );
        ops.push_back( I::sve_fmul_s(  I::z3, inv[i], acc[i] ) );
        ops.push_back( I::sve_frsqrts_s( I::z3, inv[i], I::z3 ) );
        ops.push_back( I::sve_fmul_s(  inv[i], inv[i], I::z3 ) );
    }

    // pass 2: normalize; gamma broadcast shared across the four blocks
    ops.push_back( I::base_mov_reg_x( I::x8,  I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9,  I::x20 ) );
    ops.push_back( I::base_mov_reg_x( I::x10, I::x21 ) );
    ops.push_back( I::base_mov_reg_x( I::x11, I::x23 ) );

    size_t nrm_col = ops.size();
    ops.push_back( I::sve_ld1rw_s( I::z1, I::p0, I::x10 ) );
    ops.push_back( I::sve_ld1w_imm( I::z0,  I::p0, I::x8, 0 ) );
    ops.push_back( I::sve_ld1w_imm( I::z24, I::p0, I::x8, 1 ) );
    ops.push_back( I::sve_ld1w_imm( I::z25, I::p0, I::x8, 2 ) );
    ops.push_back( I::sve_ld1w_imm( I::z26, I::p0, I::x8, 3 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0,  I::p0, I::z1 ) );
    ops.push_back( I::sve_fmul_p_s( I::z24, I::p0, I::z1 ) );
    ops.push_back( I::sve_fmul_p_s( I::z25, I::p0, I::z1 ) );
    ops.push_back( I::sve_fmul_p_s( I::z26, I::p0, I::z1 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0,  I::p0, I::z4  ) );
    ops.push_back( I::sve_fmul_p_s( I::z24, I::p0, I::z19 ) );
    ops.push_back( I::sve_fmul_p_s( I::z25, I::p0, I::z20 ) );
    ops.push_back( I::sve_fmul_p_s( I::z26, I::p0, I::z21 ) );
    ops.push_back( I::sve_st1w_imm( I::z0,  I::p0, I::x9, 0 ) );
    ops.push_back( I::sve_st1w_imm( I::z24, I::p0, I::x9, 1 ) );
    ops.push_back( I::sve_st1w_imm( I::z25, I::p0, I::x9, 2 ) );
    ops.push_back( I::sve_st1w_imm( I::z26, I::p0, I::x9, 3 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x24 ) );
    ops.push_back( I::base_add_reg_x( I::x9, I::x9, I::x25 ) );
    ops.push_back( I::base_add_imm_x( I::x10, I::x10, 4 ) );
    ops.push_back( I::base_subs_imm_x( I::x11, I::x11, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, nrm_col ) ) );

    ops.push_back( I::sve_addvl( I::x19, I::x19, 4 ) );
    ops.push_back( I::sve_addvl( I::x20, I::x20, 4 ) );
    ops.push_back( I::base_mov_reg_x( I::x0, I::x6 ) );
    ops.push_back( I::base_b( back( ops, group_loop ) ) );

    // --- tail: up to 3 full blocks + partial block, predicated ---
    size_t tail_blocks = ops.size();
    ops[fix_bhi] = I::base_b_cond( I::cond_hi,
                                   (int32_t)tail_blocks - (int32_t)fix_bhi );

    ops.push_back( I::sve_whilelo_s_x( I::p1, I::x0, I::x22 ) );
    size_t fix_bnone = ops.size();                          // b.none block_done
    ops.push_back( 0 );

    ops.push_back( I::sve_dup_imm_s( I::z2, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x23 ) );

    size_t tail_red = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p1, I::x8, 0 ) );
    ops.push_back( I::sve_fmla_s( I::z2, I::p1, I::z0, I::z0 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x24 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, tail_red ) ) );

    ops.push_back( I::sve_fmul_p_s( I::z2, I::p1, I::z5 ) );
    ops.push_back( I::sve_dup_elem_s( I::z3, I::z8, 0 ) );
    ops.push_back( I::sve_fadd_p_s( I::z2, I::p1, I::z3 ) );
    // inactive lanes get eps (not 0) so FRSQRTE never sees 0 -> inf
    ops.push_back( I::sve_sel_s( I::z2, I::p1, I::z2, I::z3 ) );

    ops.push_back( I::sve_frsqrte_s( I::z4, I::z2 ) );
    ops.push_back( I::sve_fmul_s( I::z3, I::z4, I::z2 ) );
    ops.push_back( I::sve_frsqrts_s( I::z3, I::z4, I::z3 ) );
    ops.push_back( I::sve_fmul_p_s( I::z4, I::p1, I::z3 ) );

    ops.push_back( I::base_mov_reg_x( I::x8,  I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9,  I::x20 ) );
    ops.push_back( I::base_mov_reg_x( I::x10, I::x21 ) );
    ops.push_back( I::base_mov_reg_x( I::x11, I::x23 ) );

    size_t tail_nrm = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p1, I::x8, 0 ) );
    ops.push_back( I::sve_ld1rw_s( I::z1, I::p0, I::x10 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p1, I::z1 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p1, I::z4 ) );
    ops.push_back( I::sve_st1w_imm( I::z0, I::p1, I::x9, 0 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x24 ) );
    ops.push_back( I::base_add_reg_x( I::x9, I::x9, I::x25 ) );
    ops.push_back( I::base_add_imm_x( I::x10, I::x10, 4 ) );
    ops.push_back( I::base_subs_imm_x( I::x11, I::x11, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, tail_nrm ) ) );

    ops.push_back( I::sve_addvl( I::x19, I::x19, 1 ) );
    ops.push_back( I::sve_addvl( I::x20, I::x20, 1 ) );
    ops.push_back( I::sve_incw_x( I::x0 ) );
    ops.push_back( I::base_b( back( ops, tail_blocks ) ) );

    size_t block_done = ops.size();
    ops[fix_bnone] = I::base_b_cond( I::cond_eq,
                                     (int32_t)block_done - (int32_t)fix_bnone );
    ops.push_back( I::sme_smstop_sm_only() );

    size_t row_done = ops.size();
    ops[fix_cbz] = I::base_br_cbz_x( I::x22,
                                     (int32_t)row_done - (int32_t)fix_cbz );

    ops.push_back( I::simd_ldr_imm_d( I::v8,  I::sp, 56 ) );
    ops.push_back( I::base_ldr_imm_x( I::x25, I::sp, 48 ) );
    ops.push_back( I::base_ldp_off_x( I::x23, I::x24, I::sp, 32 ) );
    ops.push_back( I::base_ldp_off_x( I::x21, I::x22, I::sp, 16 ) );
    ops.push_back( I::base_ldp_post_x( I::x19, I::x20, I::sp, 80 ) );
    ops.push_back( I::base_br_ret() );
}

// ---------------------------------------------------------------------------
// LayerNorm V6 — see layer_norm_ssve_v6.S for the full design commentary.
//   x19=a x20=b x21=gamma x22=beta x23=m x24=n x25=stride_a x26=stride_b
// ---------------------------------------------------------------------------
void Norm::emit_layer_v6() {
    std::vector<uint32_t>& ops = m_ops;

    ops.push_back( I::base_stp_pre_x( I::x19, I::x20, I::sp, -80 ) );
    ops.push_back( I::base_stp_off_x( I::x21, I::x22, I::sp, 16 ) );
    ops.push_back( I::base_stp_off_x( I::x23, I::x24, I::sp, 32 ) );
    ops.push_back( I::base_stp_off_x( I::x25, I::x26, I::sp, 48 ) );
    ops.push_back( I::simd_str_imm_d( I::v8, I::sp, 64 ) );

    ops.push_back( I::base_mov_reg_x( I::x19, I::x0 ) );   // a
    ops.push_back( I::base_mov_reg_x( I::x20, I::x1 ) );   // b
    ops.push_back( I::base_mov_reg_x( I::x21, I::x2 ) );   // gamma
    ops.push_back( I::base_mov_reg_x( I::x22, I::x3 ) );   // beta
    ops.push_back( I::base_mov_reg_x( I::x23, I::x4 ) );   // m
    ops.push_back( I::base_mov_reg_x( I::x24, I::x5 ) );   // n
    ops.push_back( I::base_lsl_x( I::x25, I::x6, 2 ) );    // stride_a bytes
    ops.push_back( I::base_lsl_x( I::x26, I::x7, 2 ) );    // stride_b bytes
    ops.push_back( I::fp_fmov_s( I::v8, I::v0 ) );
    ops.push_back( I::simd_str_imm_s( I::v0, I::sp, 72 ) ); // eps stash

    size_t fix_cbz = ops.size();
    ops.push_back( 0 );                                     // cbz m, done

    ops.push_back( I::sme_smstart_sm_only() );
    ops.push_back( I::sve_ptrue_all( I::p0, I::dtype_t::fp32 ) );

    ops.push_back( I::simd_ldr_imm_s( I::v0, I::sp, 72 ) );
    ops.push_back( I::sve_dup_elem_s( I::z7, I::z0, 0 ) );  // z7 = {eps,...}

    ops.push_back( I::fp_fmov_imm_s( I::v0, 0x70 ) );       // s0 = 1.0
    ops.push_back( I::fp_scvtf_s_x( I::v1, I::x24 ) );      // s1 = (float)n
    ops.push_back( I::fp_fdiv_s( I::v0, I::v0, I::v1 ) );
    ops.push_back( I::sve_dup_elem_s( I::z4, I::z0, 0 ) );  // z4 = {1/N,...}

    ops.push_back( I::sve_cntw_x( I::x14 ) );
    ops.push_back( I::base_lsl_x( I::x14, I::x14, 2 ) );
    ops.push_back( I::base_movz_x( I::x0, 0 ) );

    // --- group loop ---
    size_t group = ops.size();
    ops.push_back( I::base_add_reg_x( I::x15, I::x0, I::x14 ) );
    ops.push_back( I::base_cmp_reg_x( I::x15, I::x23 ) );
    size_t fix_bhi = ops.size();
    ops.push_back( 0 );                                     // b.hi tail

    // pass 1: mean accumulators
    ops.push_back( I::sve_dup_imm_s( I::z8,  0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z9,  0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z10, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z11, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x24 ) );

    size_t gp1col = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p0, I::x8, 0 ) );
    ops.push_back( I::sve_ld1w_imm( I::z1, I::p0, I::x8, 1 ) );
    ops.push_back( I::sve_ld1w_imm( I::z2, I::p0, I::x8, 2 ) );
    ops.push_back( I::sve_ld1w_imm( I::z3, I::p0, I::x8, 3 ) );
    ops.push_back( I::sve_fadd_p_s( I::z8,  I::p0, I::z0 ) );
    ops.push_back( I::sve_fadd_p_s( I::z9,  I::p0, I::z1 ) );
    ops.push_back( I::sve_fadd_p_s( I::z10, I::p0, I::z2 ) );
    ops.push_back( I::sve_fadd_p_s( I::z11, I::p0, I::z3 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, gp1col ) ) );

    ops.push_back( I::sve_fmul_p_s( I::z8,  I::p0, I::z4 ) );  // mean = sum/N
    ops.push_back( I::sve_fmul_p_s( I::z9,  I::p0, I::z4 ) );
    ops.push_back( I::sve_fmul_p_s( I::z10, I::p0, I::z4 ) );
    ops.push_back( I::sve_fmul_p_s( I::z11, I::p0, I::z4 ) );

    // pass 2: centered variance (the stable two-pass form, decision B)
    ops.push_back( I::sve_dup_imm_s( I::z12, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z13, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z14, 0 ) );
    ops.push_back( I::sve_dup_imm_s( I::z15, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x24 ) );

    size_t gp2col = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p0, I::x8, 0 ) );
    ops.push_back( I::sve_ld1w_imm( I::z1, I::p0, I::x8, 1 ) );
    ops.push_back( I::sve_ld1w_imm( I::z2, I::p0, I::x8, 2 ) );
    ops.push_back( I::sve_ld1w_imm( I::z3, I::p0, I::x8, 3 ) );
    ops.push_back( I::sve_fsub_p_s( I::z0, I::p0, I::z8 ) );
    ops.push_back( I::sve_fsub_p_s( I::z1, I::p0, I::z9 ) );
    ops.push_back( I::sve_fsub_p_s( I::z2, I::p0, I::z10 ) );
    ops.push_back( I::sve_fsub_p_s( I::z3, I::p0, I::z11 ) );
    ops.push_back( I::sve_fmla_s( I::z12, I::p0, I::z0, I::z0 ) );
    ops.push_back( I::sve_fmla_s( I::z13, I::p0, I::z1, I::z1 ) );
    ops.push_back( I::sve_fmla_s( I::z14, I::p0, I::z2, I::z2 ) );
    ops.push_back( I::sve_fmla_s( I::z15, I::p0, I::z3, I::z3 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, gp2col ) ) );

    // var/N + eps, then inv_std via FRSQRTE + one NR step
    const I::sve_t var[4] = { I::z12, I::z13, I::z14, I::z15 };
    const I::sve_t isd[4] = { I::z17, I::z18, I::z19, I::z20 };
    for ( int i = 0; i < 4; ++i ) {
        ops.push_back( I::sve_fmul_p_s( var[i], I::p0, I::z4 ) );
        ops.push_back( I::sve_fadd_p_s( var[i], I::p0, I::z7 ) );
    }
    for ( int i = 0; i < 4; ++i ) {
        ops.push_back( I::sve_frsqrte_s( isd[i], var[i] ) );
        ops.push_back( I::sve_fmul_s( I::z16, isd[i], var[i] ) );
        ops.push_back( I::sve_frsqrts_s( I::z16, isd[i], I::z16 ) );
        ops.push_back( I::sve_fmul_s( isd[i], isd[i], I::z16 ) );
    }

    // pass 3: normalize, scale gamma, shift beta
    ops.push_back( I::base_mov_reg_x( I::x8,  I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9,  I::x20 ) );
    ops.push_back( I::base_mov_reg_x( I::x10, I::x21 ) );
    ops.push_back( I::base_mov_reg_x( I::x11, I::x22 ) );
    ops.push_back( I::base_mov_reg_x( I::x12, I::x24 ) );

    size_t gp3col = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p0, I::x8, 0 ) );
    ops.push_back( I::sve_ld1w_imm( I::z1, I::p0, I::x8, 1 ) );
    ops.push_back( I::sve_ld1w_imm( I::z2, I::p0, I::x8, 2 ) );
    ops.push_back( I::sve_ld1w_imm( I::z3, I::p0, I::x8, 3 ) );
    ops.push_back( I::sve_ld1rw_s( I::z5, I::p0, I::x10 ) );   // gamma
    ops.push_back( I::sve_ld1rw_s( I::z6, I::p0, I::x11 ) );   // beta
    ops.push_back( I::sve_fsub_p_s( I::z0, I::p0, I::z8 ) );
    ops.push_back( I::sve_fsub_p_s( I::z1, I::p0, I::z9 ) );
    ops.push_back( I::sve_fsub_p_s( I::z2, I::p0, I::z10 ) );
    ops.push_back( I::sve_fsub_p_s( I::z3, I::p0, I::z11 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p0, I::z17 ) );
    ops.push_back( I::sve_fmul_p_s( I::z1, I::p0, I::z18 ) );
    ops.push_back( I::sve_fmul_p_s( I::z2, I::p0, I::z19 ) );
    ops.push_back( I::sve_fmul_p_s( I::z3, I::p0, I::z20 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z1, I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z2, I::p0, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z3, I::p0, I::z5 ) );
    ops.push_back( I::sve_fadd_p_s( I::z0, I::p0, I::z6 ) );
    ops.push_back( I::sve_fadd_p_s( I::z1, I::p0, I::z6 ) );
    ops.push_back( I::sve_fadd_p_s( I::z2, I::p0, I::z6 ) );
    ops.push_back( I::sve_fadd_p_s( I::z3, I::p0, I::z6 ) );
    ops.push_back( I::sve_st1w_imm( I::z0, I::p0, I::x9, 0 ) );
    ops.push_back( I::sve_st1w_imm( I::z1, I::p0, I::x9, 1 ) );
    ops.push_back( I::sve_st1w_imm( I::z2, I::p0, I::x9, 2 ) );
    ops.push_back( I::sve_st1w_imm( I::z3, I::p0, I::x9, 3 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_add_reg_x( I::x9, I::x9, I::x26 ) );
    ops.push_back( I::base_add_imm_x( I::x10, I::x10, 4 ) );
    ops.push_back( I::base_add_imm_x( I::x11, I::x11, 4 ) );
    ops.push_back( I::base_subs_imm_x( I::x12, I::x12, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, gp3col ) ) );

    ops.push_back( I::sve_addvl( I::x19, I::x19, 4 ) );
    ops.push_back( I::sve_addvl( I::x20, I::x20, 4 ) );
    ops.push_back( I::base_add_reg_x( I::x0, I::x0, I::x14 ) );
    ops.push_back( I::base_b( back( ops, group ) ) );

    // --- tail ---
    size_t tail = ops.size();
    ops[fix_bhi] = I::base_b_cond( I::cond_hi,
                                   (int32_t)tail - (int32_t)fix_bhi );

    ops.push_back( I::sve_whilelo_s_x( I::p1, I::x0, I::x23 ) );
    size_t fix_bnone = ops.size();
    ops.push_back( 0 );                                     // b.none done

    ops.push_back( I::sve_dup_imm_s( I::z8, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x24 ) );

    size_t tp1 = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p1, I::x8, 0 ) );
    ops.push_back( I::sve_fadd_p_s( I::z8, I::p1, I::z0 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, tp1 ) ) );

    ops.push_back( I::sve_fmul_p_s( I::z8, I::p1, I::z4 ) );
    ops.push_back( I::sve_dup_imm_s( I::z9, 0 ) );
    ops.push_back( I::base_mov_reg_x( I::x8, I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9, I::x24 ) );

    size_t tp2 = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p1, I::x8, 0 ) );
    ops.push_back( I::sve_fsub_p_s( I::z0, I::p1, I::z8 ) );
    ops.push_back( I::sve_fmla_s( I::z9, I::p1, I::z0, I::z0 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_subs_imm_x( I::x9, I::x9, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, tp2 ) ) );

    ops.push_back( I::sve_fmul_p_s( I::z9, I::p1, I::z4 ) );
    ops.push_back( I::sve_fadd_p_s( I::z9, I::p1, I::z7 ) );
    ops.push_back( I::sve_sel_s( I::z9, I::p1, I::z9, I::z7 ) );

    ops.push_back( I::sve_frsqrte_s( I::z5, I::z9 ) );
    ops.push_back( I::sve_fmul_s( I::z3, I::z5, I::z9 ) );
    ops.push_back( I::sve_frsqrts_s( I::z3, I::z5, I::z3 ) );
    ops.push_back( I::sve_fmul_p_s( I::z5, I::p1, I::z3 ) );

    ops.push_back( I::base_mov_reg_x( I::x8,  I::x19 ) );
    ops.push_back( I::base_mov_reg_x( I::x9,  I::x20 ) );
    ops.push_back( I::base_mov_reg_x( I::x10, I::x21 ) );
    ops.push_back( I::base_mov_reg_x( I::x11, I::x22 ) );
    ops.push_back( I::base_mov_reg_x( I::x12, I::x24 ) );

    size_t tp3 = ops.size();
    ops.push_back( I::sve_ld1w_imm( I::z0, I::p1, I::x8, 0 ) );
    ops.push_back( I::sve_ld1rw_s( I::z1, I::p1, I::x10 ) );
    ops.push_back( I::sve_ld1rw_s( I::z2, I::p1, I::x11 ) );
    ops.push_back( I::sve_fsub_p_s( I::z0, I::p1, I::z8 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p1, I::z5 ) );
    ops.push_back( I::sve_fmul_p_s( I::z0, I::p1, I::z1 ) );
    ops.push_back( I::sve_fadd_p_s( I::z0, I::p1, I::z2 ) );
    ops.push_back( I::sve_st1w_imm( I::z0, I::p1, I::x9, 0 ) );
    ops.push_back( I::base_add_reg_x( I::x8, I::x8, I::x25 ) );
    ops.push_back( I::base_add_reg_x( I::x9, I::x9, I::x26 ) );
    ops.push_back( I::base_add_imm_x( I::x10, I::x10, 4 ) );
    ops.push_back( I::base_add_imm_x( I::x11, I::x11, 4 ) );
    ops.push_back( I::base_subs_imm_x( I::x12, I::x12, 1 ) );
    ops.push_back( I::base_b_ne( back( ops, tp3 ) ) );

    ops.push_back( I::sve_addvl( I::x19, I::x19, 1 ) );
    ops.push_back( I::sve_addvl( I::x20, I::x20, 1 ) );
    ops.push_back( I::sve_incw_x( I::x0 ) );
    ops.push_back( I::base_b( back( ops, tail ) ) );

    size_t done = ops.size();
    ops[fix_bnone] = I::base_b_cond( I::cond_eq,
                                     (int32_t)done - (int32_t)fix_bnone );
    ops[fix_cbz] = I::base_br_cbz_x( I::x23,
                                     (int32_t)done - (int32_t)fix_cbz );

    ops.push_back( I::sme_smstop_sm_only() );
    ops.push_back( I::simd_ldr_imm_d( I::v8, I::sp, 64 ) );
    ops.push_back( I::base_ldp_off_x( I::x25, I::x26, I::sp, 48 ) );
    ops.push_back( I::base_ldp_off_x( I::x23, I::x24, I::sp, 32 ) );
    ops.push_back( I::base_ldp_off_x( I::x21, I::x22, I::sp, 16 ) );
    ops.push_back( I::base_ldp_post_x( I::x19, I::x20, I::sp, 80 ) );
    ops.push_back( I::base_br_ret() );
}

// ---------------------------------------------------------------------------

Norm::error_t Norm::generate( ntype_t ntype ) {
    m_ops.clear();
    m_ntype = ntype;

    if ( ntype == ntype_t::rms ) emit_rms_v6();
    else                         emit_layer_v6();

    m_kernel = (void*)JitEngine::generate<void*>( m_ops );
    return ( m_kernel != nullptr ) ? error_t::success : error_t::err_alloc;
}

Norm::rms_kernel_t Norm::get_rms_kernel() const {
    if ( m_ntype != ntype_t::rms ) return nullptr;
    return reinterpret_cast<rms_kernel_t>( m_kernel );
}

Norm::layer_kernel_t Norm::get_layer_kernel() const {
    if ( m_ntype != ntype_t::layer ) return nullptr;
    return reinterpret_cast<layer_kernel_t>( m_kernel );
}

} // namespace mini_jit
