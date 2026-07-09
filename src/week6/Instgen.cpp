#include "week6/Instgen.hpp"
#include <sstream>
#include <iomanip>
#include <bitset>

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::string mini_jit::InstGen::to_string_hex( uint32_t inst ) {
  std::stringstream l_ss;
  l_ss << "0x" << std::hex
               << std::setfill('0')
               << std::setw(8)
               << inst;
  return l_ss.str();
}

std::string mini_jit::InstGen::to_string_bin( uint32_t inst ) {
  std::string l_res = "0b";
  l_res += std::bitset<32>(inst).to_string();
  return l_res;
}

// ---------------------------------------------------------------------------
// Helper: extract 5-bit register id (w/x share the low 5 bits)
// ---------------------------------------------------------------------------
static inline uint32_t reg_id( mini_jit::InstGen::gpr_t r ) {
  return (uint32_t)r & 0x1fu;
}


uint32_t mini_jit::InstGen::base_br_cbnz( gpr_t   reg,
                                          int32_t imm19 ) {
  uint32_t l_ins = 0x35000000u;
  l_ins |= (reg & 0x1fu);
  l_ins |= ((uint32_t)(reg & 0x20u)) << (32 - 6);
  l_ins |= ((uint32_t)(imm19 & 0x7ffffu)) << 5;
  return l_ins;
}

uint32_t mini_jit::InstGen::neon_dp_fmla_vector( simd_fp_t  reg_dest,
                                                 simd_fp_t  reg_src1,
                                                 simd_fp_t  reg_src2,
                                                 arr_spec_t arr_spec ) {
  uint32_t l_ins = 0x0e20cc00u;
  l_ins |= (reg_dest & 0x1fu);
  l_ins |= (reg_src1 & 0x1fu) << 5;
  l_ins |= (reg_src2 & 0x1fu) << 16;
  l_ins |= (arr_spec & 0x40400000u);
  return l_ins;
}

// ---------------------------------------------------------------------------
// SME / SVE instructions
// ---------------------------------------------------------------------------

uint32_t mini_jit::InstGen::sme_smstart_sm() {
  // MSR SVCR, #1  (SMSTART SM) 
  return 0xd503477fu;
}

uint32_t mini_jit::InstGen::sme_smstop_sm() {
  // MSR SVCR, #0  (SMSTOP SM)   
  return 0xd503467fu;
}

uint32_t mini_jit::InstGen::sme_zero_za() {
  // ZERO {ZA}  — 0xC00800FF
  return 0xc00800ffu;
}

uint32_t mini_jit::InstGen::sve_ptrue_all( pred_t  pg,
                                           dtype_t dt ) {
  // PTRUE P<pg>.<T>, ALL
  // fp32 (.S): base 0x2598e000, dtype bits: size=10 (bits 23:22)
  // fp64 (.D): base 0x25D8e000, dtype bits: size=11 (bits 23:22)
  // Pattern ALL = 0x1f (bits 4:0)
  // Pd (bits 3:0)
  // Sprint-4 fix: the fp32 constant was 0x2518e3e0 = size=00 = PTRUE P.B,
  // contradicting this comment.  Functionally masked for every existing
  // caller (ALL-true .B governs .S identically) but caught by the
  // encoding-diff against the assembled norm kernels.
  uint32_t base = (dt == dtype_t::fp32) ? 0x2598e3e0u : 0x25d8e3e0u;
  // The base encodings above use p0; patch in pg
  uint32_t ins = base;
  ins = (ins & ~0xfu) | ((uint32_t)pg & 0xfu);
  return ins;
}

uint32_t mini_jit::InstGen::base_br_ret() {
  // RET  = 0xD65F03C0
  return 0xd65f03c0u;
}

uint32_t mini_jit::InstGen::base_movz_x( gpr_t    rd,
                                          uint32_t imm16,
                                          uint32_t hw ) {
  // MOVZ X<rd>, #imm16, LSL #(hw*16)
  // sf=1, opc=10, hw[1:0] → 0xD2800000 | (hw<<21) | (imm16<<5) | rd
  return 0xd2800000u | ((hw & 3u) << 21) | ((imm16 & 0xffffu) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_movk_x( gpr_t    rd,
                                          uint32_t imm16,
                                          uint32_t hw ) {
  // MOVK X<rd>, #imm16, LSL #(hw*16)
  // sf=1, opc=11 → 0xF2800000 | (hw<<21) | (imm16<<5) | rd
  return 0xf2800000u | ((hw & 3u) << 21) | ((imm16 & 0xffffu) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_movz_w( gpr_t    rd,
                                          uint32_t imm16 ) {
  // MOVZ W<rd>, #imm16   (sf=0)
  return 0x52800000u | ((imm16 & 0xffffu) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_add_reg_x( gpr_t rd,
                                            gpr_t rn,
                                            gpr_t rm ) {
  // ADD X<rd>, X<rn>, X<rm>  sf=1, shifted-reg shift=0
  return 0x8b000000u | (reg_id(rm) << 16) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_add_imm_x( gpr_t    rd,
                                            gpr_t    rn,
                                            uint32_t imm12 ) {
  // ADD X<rd>, X<rn>, #imm12   sf=1, S=0
  return 0x91000000u | ((imm12 & 0xfffu) << 10) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_add_lsl_x( gpr_t    rd,
                                            gpr_t    rn,
                                            gpr_t    rm,
                                            uint32_t shift ) {
  // ADD X<rd>, X<rn>, X<rm>, LSL #shift   (shifted-register)
  return 0x8b000000u | (reg_id(rm) << 16) | ((shift & 0x3fu) << 10) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_lsl_x( gpr_t    rd,
                                        gpr_t    rn,
                                        uint32_t shift ) {
  // LSL X<rd>, X<rn>, #shift  →  UBFM X<rd>, X<rn>, #(-shift MOD 64), #(63-shift)
  // sf=1, N=1: 0xD3400000 | (immr<<16) | (imms<<10) | (rn<<5) | rd
  uint32_t immr = (uint32_t)(-(int32_t)shift) & 63u;
  uint32_t imms = 63u - shift;
  return 0xd3400000u | (immr << 16) | (imms << 10) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_mul_x( gpr_t rd,
                                        gpr_t rn,
                                        gpr_t rm ) {
  // MUL X<rd>, X<rn>, X<rm>  =  MADD X<rd>, X<rn>, X<rm>, XZR
  // 0x9B007C00 | (rm<<16) | (rn<<5) | rd
  return 0x9b007c00u | (reg_id(rm) << 16) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_subs_imm_x( gpr_t    rd,
                                             gpr_t    rn,
                                             uint32_t imm12 ) {
  // SUBS X<rd>, X<rn>, #imm12   sf=1 S=1
  return 0xf1000000u | ((imm12 & 0xfffu) << 10) | (reg_id(rn) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_b_ne( int32_t offset19 ) {
  // B.NE = B.cond with cond=0001
  // encoding: 0x54000001 | ((offset19 & 0x7ffff) << 5)
  return 0x54000001u | ((uint32_t)(offset19 & 0x7ffffu) << 5);
}

// ---------------------------------------------------------------------------
// SVE scalar+scalar loads/stores
// LD1W {Z<zt>.S}, P<pg>/Z, [X<base>]   →  0xA540A000 | (pg<<10) | (base<<5) | zt
// LD1D {Z<zt>.D}, P<pg>/Z, [X<base>]   →  0xA560A000 | ...
// ST1W {Z<zt>.S}, P<pg>,   [X<base>]   →  0xE540E000 | (pg<<10) | (base<<5) | zt
// ST1D {Z<zt>.D}, P<pg>,   [X<base>]   →  0xE560E000 | ...
// ---------------------------------------------------------------------------

uint32_t mini_jit::InstGen::sve_ld1w_scalar( sve_t  zt,
                                             pred_t pg,
                                             gpr_t  rn ) {
  return 0xa540a000u | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_ld1d_scalar( sve_t  zt,
                                             pred_t pg,
                                             gpr_t  rn ) {
  return 0xa560a000u | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_st1w_scalar( sve_t  zt,
                                             pred_t pg,
                                             gpr_t  rn ) {
  return 0xe540e000u | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_st1d_scalar( sve_t  zt,
                                             pred_t pg,
                                             gpr_t  rn ) {
  return 0xe560e000u | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

// ---------------------------------------------------------------------------
// FMOPA outer products
// FMOPA ZA<n>.S, Pn/M, Pm/M, Zn.S, Zm.S
//   base = 0x80820000 | (zm<<16) | (pn<<13) | (pm<<10) | (zn<<5) | za_idx
// FMOPA ZA<n>.D:
//   base = 0x81820000 | ...
// ---------------------------------------------------------------------------

uint32_t mini_jit::InstGen::sme_fmopa_s( uint32_t za_idx,
                                         pred_t   pn,
                                         pred_t   pm,
                                         sve_t    zn,
                                         sve_t    zm ) {
  return 0x80820000u
         | ((uint32_t)zm  << 16)
         | ((uint32_t)pn  << 13)
         | ((uint32_t)pm  << 10)
         | ((uint32_t)zn  <<  5)
         | (za_idx & 0x3u);
}

uint32_t mini_jit::InstGen::sme_fmopa_d( uint32_t za_idx,
                                         pred_t   pn,
                                         pred_t   pm,
                                         sve_t    zn,
                                         sve_t    zm ) {
  return 0x81820000u
         | ((uint32_t)zm  << 16)
         | ((uint32_t)pn  << 13)
         | ((uint32_t)pm  << 10)
         | ((uint32_t)zn  <<  5)
         | (za_idx & 0x1u);   // fp64 only has za0/za1
}

// ---------------------------------------------------------------------------
// MOVA: move ZA tile slice into/from SVE register
//
// MOV Z<zt>.S, P<pg>/M, ZA<n>V.S[W<ws>,0]  (vertical)
//   base = 0xC0828000 | (za_idx * 0x80) | (ws_enc<<13) | (pg<<10) | (offset<<5) | zt
//   ws must be W12..W15, encoded as 0..3 → (ws_reg - 12)
//
// MOV Z<zt>.S, P<pg>/M, ZA<n>H.S[W<ws>,0]  (horizontal)
//   base = 0xC0820000 | ...
// ---------------------------------------------------------------------------

uint32_t mini_jit::InstGen::sme_mova_tile_to_vec_v_s( sve_t    zt,
                                                       pred_t   pg,
                                                       uint32_t za_idx,
                                                       uint32_t ws,
                                                       uint32_t offset ) {
  // ws is the raw W register number (12–15); encode as (ws - 12)
  uint32_t ws_enc = (ws >= 12u) ? (ws - 12u) : ws;
  return 0xc0828000u
         | ((za_idx & 0x3u) << 7)
         | ((ws_enc & 0x3u) << 13)
         | ((uint32_t)pg << 10)
         | ((offset & 0xfu) << 5)
         | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sme_mova_tile_to_vec_h_s( sve_t    zt,
                                                       pred_t   pg,
                                                       uint32_t za_idx,
                                                       uint32_t ws,
                                                       uint32_t offset ) {
  uint32_t ws_enc = (ws >= 12u) ? (ws - 12u) : ws;
  return 0xc0820000u
         | ((za_idx & 0x3u) << 7)
         | ((ws_enc & 0x3u) << 13)
         | ((uint32_t)pg << 10)
         | ((offset & 0xfu) << 5)
         | (uint32_t)zt;
}

// ---------------------------------------------------------------------------
// FMAX Z<zd>.S, P<pg>/M, Z<zd>.S, Z<zm>.S
//   base encoding (verified with LLVM): 0x65868000
//   bits: opc=FMAX, size=10 (.S), Pg<<10, Zm<<5, Zdn
// ---------------------------------------------------------------------------

uint32_t mini_jit::InstGen::sve_fmax_s( sve_t  zd,
                                        pred_t pg,
                                        sve_t  zm ) {
  // FMAX Zdn.S, Pg/M, Zdn.S, Zm.S
  return 0x65868000u
         | ((uint32_t)pg << 10)
         | ((uint32_t)zm <<  5)
         | (uint32_t)zd;
}
// ===========================================================================
// Sprint 4 — encoders for the mini_jit::Norm generator.
// Every base word below is verified against the toolchain-assembled
// instruction words of rms_norm_ssve_v6.S / layer_norm_ssve_v6.S
// (per-encoder unit tests: tests/test_norm.cpp, tag [sprint4][encoders])
// and the Arm ARM field layouts.
// ===========================================================================

uint32_t mini_jit::InstGen::sme_smstart_sm_only() {
  // MSR SVCRSM, #1 — streaming mode on, ZA untouched.
  return 0xd503437fu;
}

uint32_t mini_jit::InstGen::sme_smstop_sm_only() {
  // MSR SVCRSM, #0
  return 0xd503427fu;
}

uint32_t mini_jit::InstGen::base_stp_pre_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm ) {
  // STP Xt, Xt2, [Xn, #simm]!  — imm7 scaled by 8
  return 0xa9800000u
         | (((uint32_t)(simm / 8) & 0x7fu) << 15)
         | (reg_id(rt2) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::base_stp_off_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm ) {
  return 0xa9000000u
         | (((uint32_t)(simm / 8) & 0x7fu) << 15)
         | (reg_id(rt2) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::base_ldp_off_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm ) {
  return 0xa9400000u
         | (((uint32_t)(simm / 8) & 0x7fu) << 15)
         | (reg_id(rt2) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::base_ldp_post_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm ) {
  return 0xa8c00000u
         | (((uint32_t)(simm / 8) & 0x7fu) << 15)
         | (reg_id(rt2) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::base_str_imm_x( gpr_t rt, gpr_t rn, uint32_t uimm ) {
  // STR Xt, [Xn, #uimm] — imm12 scaled by 8
  return 0xf9000000u | (((uimm / 8u) & 0xfffu) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::base_ldr_imm_x( gpr_t rt, gpr_t rn, uint32_t uimm ) {
  return 0xf9400000u | (((uimm / 8u) & 0xfffu) << 10) | (reg_id(rn) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::simd_str_imm_d( simd_fp_t vt, gpr_t rn, uint32_t uimm ) {
  // STR Dt, [Xn, #uimm] — imm12 scaled by 8
  return 0xfd000000u | (((uimm / 8u) & 0xfffu) << 10) | (reg_id(rn) << 5) | ((uint32_t)vt & 0x1fu);
}

uint32_t mini_jit::InstGen::simd_ldr_imm_d( simd_fp_t vt, gpr_t rn, uint32_t uimm ) {
  return 0xfd400000u | (((uimm / 8u) & 0xfffu) << 10) | (reg_id(rn) << 5) | ((uint32_t)vt & 0x1fu);
}

uint32_t mini_jit::InstGen::simd_str_imm_s( simd_fp_t vt, gpr_t rn, uint32_t uimm ) {
  // STR St, [Xn, #uimm] — imm12 scaled by 4
  return 0xbd000000u | (((uimm / 4u) & 0xfffu) << 10) | (reg_id(rn) << 5) | ((uint32_t)vt & 0x1fu);
}

uint32_t mini_jit::InstGen::simd_ldr_imm_s( simd_fp_t vt, gpr_t rn, uint32_t uimm ) {
  return 0xbd400000u | (((uimm / 4u) & 0xfffu) << 10) | (reg_id(rn) << 5) | ((uint32_t)vt & 0x1fu);
}

uint32_t mini_jit::InstGen::base_mov_reg_x( gpr_t rd, gpr_t rm ) {
  // ORR Xd, XZR, Xm
  return 0xaa0003e0u | (reg_id(rm) << 16) | reg_id(rd);
}

uint32_t mini_jit::InstGen::base_cmp_reg_x( gpr_t rn, gpr_t rm ) {
  // SUBS XZR, Xn, Xm
  return 0xeb00001fu | (reg_id(rm) << 16) | (reg_id(rn) << 5);
}

uint32_t mini_jit::InstGen::base_b_cond( cond_t cond, int32_t offset19 ) {
  return 0x54000000u | (((uint32_t)offset19 & 0x7ffffu) << 5) | (uint32_t)cond;
}

uint32_t mini_jit::InstGen::base_b( int32_t offset26 ) {
  return 0x14000000u | ((uint32_t)offset26 & 0x3ffffffu);
}

uint32_t mini_jit::InstGen::base_br_cbz_x( gpr_t rt, int32_t imm19 ) {
  return 0xb4000000u | (((uint32_t)imm19 & 0x7ffffu) << 5) | reg_id(rt);
}

uint32_t mini_jit::InstGen::fp_fmov_s( simd_fp_t vd, simd_fp_t vn ) {
  return 0x1e204000u | (((uint32_t)vn & 0x1fu) << 5) | ((uint32_t)vd & 0x1fu);
}

uint32_t mini_jit::InstGen::fp_fmov_imm_s( simd_fp_t vd, uint32_t imm8 ) {
  return 0x1e201000u | ((imm8 & 0xffu) << 13) | ((uint32_t)vd & 0x1fu);
}

uint32_t mini_jit::InstGen::fp_scvtf_s_x( simd_fp_t vd, gpr_t rn ) {
  return 0x9e220000u | (reg_id(rn) << 5) | ((uint32_t)vd & 0x1fu);
}

uint32_t mini_jit::InstGen::fp_fdiv_s( simd_fp_t vd, simd_fp_t vn, simd_fp_t vm ) {
  return 0x1e201800u | (((uint32_t)vm & 0x1fu) << 16)
         | (((uint32_t)vn & 0x1fu) << 5) | ((uint32_t)vd & 0x1fu);
}

uint32_t mini_jit::InstGen::sve_cntw_x( gpr_t rd ) {
  // CNTW Xd, ALL, MUL #1
  return 0x04a0e3e0u | reg_id(rd);
}

uint32_t mini_jit::InstGen::sve_incw_x( gpr_t rd ) {
  return 0x04b0e3e0u | reg_id(rd);
}

uint32_t mini_jit::InstGen::sve_addvl( gpr_t rd, gpr_t rn, int32_t simm6 ) {
  return 0x04205000u | (reg_id(rn) << 16)
         | (((uint32_t)simm6 & 0x3fu) << 5) | reg_id(rd);
}

uint32_t mini_jit::InstGen::sve_dup_imm_s( sve_t zd, int32_t simm8 ) {
  // DUP Zd.S, #simm8 (size=10)
  return 0x25b8c000u | (((uint32_t)simm8 & 0xffu) << 5) | (uint32_t)zd;
}

uint32_t mini_jit::InstGen::sve_dup_elem_s( sve_t zd, sve_t zn, uint32_t idx ) {
  // DUP Zd.S, Zn.S[idx] — imm2:tsz = idx:100
  uint32_t imm5 = ((idx & 0xfu) << 3) | 0x4u;
  return 0x05202000u | (imm5 << 16) | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zd;
}

uint32_t mini_jit::InstGen::sve_ld1w_imm( sve_t zt, pred_t pg, gpr_t rn, int32_t simm4 ) {
  // LD1W {Zt.S}, Pg/Z, [Xn, #simm4, MUL VL]
  return 0xa540a000u | (((uint32_t)simm4 & 0xfu) << 16)
         | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_st1w_imm( sve_t zt, pred_t pg, gpr_t rn, int32_t simm4 ) {
  return 0xe540e000u | (((uint32_t)simm4 & 0xfu) << 16)
         | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_ld1rw_s( sve_t zt, pred_t pg, gpr_t rn, uint32_t uimm6 ) {
  // LD1RW {Zt.S}, Pg/Z, [Xn, #uimm6*4]
  return 0x8540c000u | ((uimm6 & 0x3fu) << 16)
         | ((uint32_t)pg << 10) | (reg_id(rn) << 5) | (uint32_t)zt;
}

uint32_t mini_jit::InstGen::sve_fmla_s( sve_t zda, pred_t pg, sve_t zn, sve_t zm ) {
  return 0x65a00000u | (((uint32_t)zm & 0x1fu) << 16)
         | ((uint32_t)pg << 10) | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zda;
}

uint32_t mini_jit::InstGen::sve_fadd_p_s( sve_t zdn, pred_t pg, sve_t zm ) {
  return 0x65808000u | ((uint32_t)pg << 10)
         | (((uint32_t)zm & 0x1fu) << 5) | (uint32_t)zdn;
}

uint32_t mini_jit::InstGen::sve_fsub_p_s( sve_t zdn, pred_t pg, sve_t zm ) {
  return 0x65818000u | ((uint32_t)pg << 10)
         | (((uint32_t)zm & 0x1fu) << 5) | (uint32_t)zdn;
}

uint32_t mini_jit::InstGen::sve_fmul_p_s( sve_t zdn, pred_t pg, sve_t zm ) {
  return 0x65828000u | ((uint32_t)pg << 10)
         | (((uint32_t)zm & 0x1fu) << 5) | (uint32_t)zdn;
}

uint32_t mini_jit::InstGen::sve_fmul_s( sve_t zd, sve_t zn, sve_t zm ) {
  return 0x65800800u | (((uint32_t)zm & 0x1fu) << 16)
         | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zd;
}

uint32_t mini_jit::InstGen::sve_frsqrte_s( sve_t zd, sve_t zn ) {
  return 0x658f3000u | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zd;
}

uint32_t mini_jit::InstGen::sve_frsqrts_s( sve_t zd, sve_t zn, sve_t zm ) {
  return 0x65801c00u | (((uint32_t)zm & 0x1fu) << 16)
         | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zd;
}

uint32_t mini_jit::InstGen::sve_whilelo_s_x( pred_t pd, gpr_t rn, gpr_t rm ) {
  // WHILELO Pd.S, Xn, Xm (64-bit operands)
  return 0x25a01c00u | (reg_id(rm) << 16) | (reg_id(rn) << 5) | (uint32_t)pd;
}

uint32_t mini_jit::InstGen::sve_sel_s( sve_t zd, pred_t pg, sve_t zn, sve_t zm ) {
  return 0x05a0c000u | (((uint32_t)zm & 0x1fu) << 16) | ((uint32_t)pg << 10)
         | (((uint32_t)zn & 0x1fu) << 5) | (uint32_t)zd;
}
