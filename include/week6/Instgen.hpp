#ifndef MINI_JIT_INSTGEN_H
#define MINI_JIT_INSTGEN_H

#include <cstdint>
#include <string>

namespace mini_jit {
  class InstGen;
}

class mini_jit::InstGen {
  public:
    //! general-purpose registers
    typedef enum : uint32_t {
      w0  =  0,  w1  =  1,  w2  =  2,  w3  =  3,
      w4  =  4,  w5  =  5,  w6  =  6,  w7  =  7,
      w8  =  8,  w9  =  9,  w10 = 10,  w11 = 11,
      w12 = 12,  w13 = 13,  w14 = 14,  w15 = 15,
      w16 = 16,  w17 = 17,  w18 = 18,  w19 = 19,
      w20 = 20,  w21 = 21,  w22 = 22,  w23 = 23,
      w24 = 24,  w25 = 25,  w26 = 26,  w27 = 27,
      w28 = 28,  w29 = 29,  w30 = 30,

      x0  = 32+0,  x1  = 32+1,  x2  = 32+2,  x3  = 32+3,
      x4  = 32+4,  x5  = 32+5,  x6  = 32+6,  x7  = 32+7,
      x8  = 32+8,  x9  = 32+9,  x10 = 32+10, x11 = 32+11,
      x12 = 32+12, x13 = 32+13, x14 = 32+14, x15 = 32+15,
      x16 = 32+16, x17 = 32+17, x18 = 32+18, x19 = 32+19,
      x20 = 32+20, x21 = 32+21, x22 = 32+22, x23 = 32+23,
      x24 = 32+24, x25 = 32+25, x26 = 32+26, x27 = 32+27,
      x28 = 32+28, x29 = 32+29, x30 = 32+30,

      wzr =       31,
      xzr =    32+31,
      sp  = 64+32+31
    } gpr_t;

    //! simd&fp registers
    typedef enum : uint32_t {
      v0  =  0, v1  =  1, v2  =  2, v3  =  3,
      v4  =  4, v5  =  5, v6  =  6, v7  =  7,
      v8  =  8, v9  =  9, v10 = 10, v11 = 11,
      v12 = 12, v13 = 13, v14 = 14, v15 = 15,
      v16 = 16, v17 = 17, v18 = 18, v19 = 19,
      v20 = 20, v21 = 21, v22 = 22, v23 = 23,
      v24 = 24, v25 = 25, v26 = 26, v27 = 27,
      v28 = 28, v29 = 29, v30 = 30, v31 = 31
    } simd_fp_t;

    //! SVE predicate registers
    typedef enum : uint32_t {
      p0  =  0, p1  =  1, p2  =  2, p3  =  3,
      p4  =  4, p5  =  5, p6  =  6, p7  =  7,
      p8  =  8, p9  =  9, p10 = 10, p11 = 11,
      p12 = 12, p13 = 13, p14 = 14, p15 = 15
    } pred_t;

    //! SVE Z registers
    typedef enum : uint32_t {
      z0  =  0, z1  =  1, z2  =  2, z3  =  3,
      z4  =  4, z5  =  5, z6  =  6, z7  =  7,
      z8  =  8, z9  =  9, z10 = 10, z11 = 11,
      z12 = 12, z13 = 13, z14 = 14, z15 = 15,
      z16 = 16, z17 = 17, z18 = 18, z19 = 19,
      z20 = 20, z21 = 21, z22 = 22, z23 = 23,
      z24 = 24, z25 = 25, z26 = 26, z27 = 27,
      z28 = 28, z29 = 29, z30 = 30, z31 = 31
    } sve_t;

    //! arrangement specifiers
    typedef enum : uint32_t {
      s2 = 0x0,
      s4 = 0x40000000,
      d2 = 0x40400000
    } arr_spec_t;

    //! data type for SME/SVE instructions
    typedef enum : uint32_t {
      fp32 = 0,
      fp64 = 1
    } dtype_t;



    static uint32_t base_br_cbnz( gpr_t   reg,
                                  int32_t imm19 );

    static uint32_t neon_dp_fmla_vector( simd_fp_t   reg_dest,
                                         simd_fp_t   reg_src1,
                                         simd_fp_t   reg_src2,
                                         arr_spec_t  arr_spec );

    static std::string to_string_hex( uint32_t inst );
    static std::string to_string_bin( uint32_t inst );

    // -----------------------------------------------------------------------
    // SME / SVE instructions
    // -----------------------------------------------------------------------

    /**
     * @brief SMSTART SM  — enter Streaming SVE mode.
     * @return instruction encoding.
     */
    static uint32_t sme_smstart_sm();

    /**
     * @brief SMSTOP SM  — exit Streaming SVE mode.
     * @return instruction encoding.
     */
    static uint32_t sme_smstop_sm();

    /**
     * @brief ZERO {ZA}  — zero entire ZA storage.
     * @return instruction encoding.
     */
    static uint32_t sme_zero_za();

    /**
     * @brief PTRUE P<p>.<T>, ALL  — activate all predicate lanes.
     * @param pg   predicate register (p0..p15).
     * @param dt   fp32 → .S  (32-bit element);  fp64 → .D  (64-bit element).
     * @return instruction encoding.
     */
    static uint32_t sve_ptrue_all( pred_t  pg,
                                   dtype_t dt );

    /**
     * @brief RET  — return to caller (X30).
     * @return instruction encoding.
     */
    static uint32_t base_br_ret();

    /**
     * @brief MOVZ X<rd>, #imm16 [, LSL #shift16]
     * @param rd      destination register (x-variant).
     * @param imm16   16-bit immediate (0–65535).
     * @param hw      half-word shift: 0→LSL#0, 1→LSL#16, 2→LSL#32, 3→LSL#48.
     * @return instruction encoding.
     */
    static uint32_t base_movz_x( gpr_t    rd,
                                  uint32_t imm16,
                                  uint32_t hw = 0 );

    /**
     * @brief MOVK X<rd>, #imm16, LSL #(hw*16)
     * @param rd      destination register (x-variant).
     * @param imm16   16-bit immediate.
     * @param hw      half-word shift (0-3).
     * @return instruction encoding.
     */
    static uint32_t base_movk_x( gpr_t    rd,
                                  uint32_t imm16,
                                  uint32_t hw );

    /**
     * @brief MOVZ W<rd>, #imm16  (32-bit MOVZ)
     */
    static uint32_t base_movz_w( gpr_t    rd,
                                  uint32_t imm16 );

    /**
     * @brief ADD X<rd>, X<rn>, X<rm>  (shifted-register, shift=0)
     */
    static uint32_t base_add_reg_x( gpr_t rd,
                                    gpr_t rn,
                                    gpr_t rm );

    /**
     * @brief ADD X<rd>, X<rn>, #imm12
     */
    static uint32_t base_add_imm_x( gpr_t    rd,
                                    gpr_t    rn,
                                    uint32_t imm12 );

    /**
     * @brief ADD X<rd>, X<rn>, X<rm>, LSL #shift
     */
    static uint32_t base_add_lsl_x( gpr_t    rd,
                                    gpr_t    rn,
                                    gpr_t    rm,
                                    uint32_t shift );

    /**
     * @brief LSL X<rd>, X<rn>, #shift  (encoded as UBFM)
     */
    static uint32_t base_lsl_x( gpr_t    rd,
                                 gpr_t    rn,
                                 uint32_t shift );

    /**
     * @brief MUL X<rd>, X<rn>, X<rm>   (MADD with Ra=XZR)
     */
    static uint32_t base_mul_x( gpr_t rd,
                                 gpr_t rn,
                                 gpr_t rm );

    /**
     * @brief SUBS X<rd>, X<rn>, #imm12  — subtract and set flags
     */
    static uint32_t base_subs_imm_x( gpr_t    rd,
                                     gpr_t    rn,
                                     uint32_t imm12 );

    /**
     * @brief B.NE <label>  (branch if not-equal; offset in instructions from B.NE)
     * @param offset19  signed 19-bit instruction offset (branch target - this instruction).
     */
    static uint32_t base_b_ne( int32_t offset19 );

    /**
     * @brief LD1W {Z<zt>.S}, P<pg>/Z, [X<rn>]  — SVE load words (fp32)
     * @param zt   SVE destination register.
     * @param pg   governing predicate.
     * @param rn   base address register (x-variant).
     */
    static uint32_t sve_ld1w_scalar( sve_t  zt,
                                     pred_t pg,
                                     gpr_t  rn );

    /**
     * @brief LD1D {Z<zt>.D}, P<pg>/Z, [X<rn>]  — SVE load doublewords (fp64)
     */
    static uint32_t sve_ld1d_scalar( sve_t  zt,
                                     pred_t pg,
                                     gpr_t  rn );

    /**
     * @brief ST1W {Z<zt>.S}, P<pg>, [X<rn>]  — SVE store words (fp32)
     */
    static uint32_t sve_st1w_scalar( sve_t  zt,
                                     pred_t pg,
                                     gpr_t  rn );

    /**
     * @brief ST1D {Z<zt>.D}, P<pg>, [X<rn>]  — SVE store doublewords (fp64)
     */
    static uint32_t sve_st1d_scalar( sve_t  zt,
                                     pred_t pg,
                                     gpr_t  rn );

    /**
     * @brief FMOPA ZA<n>.S, P<pn>/M, P<pm>/M, Z<zn>.S, Z<zm>.S  (fp32 outer product)
     * @param za_idx  ZA tile index (0–3 for fp32).
     * @param pn      row predicate.
     * @param pm      column predicate.
     * @param zn      A-matrix SVE register.
     * @param zm      B-matrix SVE register.
     */
    static uint32_t sme_fmopa_s( uint32_t za_idx,
                                 pred_t   pn,
                                 pred_t   pm,
                                 sve_t    zn,
                                 sve_t    zm );

    /**
     * @brief FMOPA ZA<n>.D, ... (fp64 outer product)
     */
    static uint32_t sme_fmopa_d( uint32_t za_idx,
                                 pred_t   pn,
                                 pred_t   pm,
                                 sve_t    zn,
                                 sve_t    zm );

    /**
     * @brief MOV Z<zt>.S, P<pg>/M, ZA<za_idx>V.S[W<ws>, <offset>]
     *        (read vertical slice of SME tile into SVE register, fp32)
     * @param zt       destination SVE register.
     * @param pg       governing predicate.
     * @param za_idx   tile index (0–3).
     * @param ws       W-register index for slice base (must be 12–15, encoded as 0–3).
     * @param offset   immediate slice offset (0–15 for fp32).
     */
    static uint32_t sme_mova_tile_to_vec_v_s( sve_t    zt,
                                              pred_t   pg,
                                              uint32_t za_idx,
                                              uint32_t ws,
                                              uint32_t offset = 0 );

    /**
     * @brief MOV Z<zt>.S, P<pg>/M, ZA<za_idx>H.S[W<ws>, <offset>]
     *        (read horizontal slice, fp32)
     */
    static uint32_t sme_mova_tile_to_vec_h_s( sve_t    zt,
                                              pred_t   pg,
                                              uint32_t za_idx,
                                              uint32_t ws,
                                              uint32_t offset = 0 );

    /**
     * @brief FMAX Z<zd>.S, P<pg>/M, Z<zd>.S, Z<zm>.S  (element-wise max, fp32)
     *        Used for ReLU: max(z, 0).
     */
    static uint32_t sve_fmax_s( sve_t  zd,
                                pred_t pg,
                                sve_t  zm );

    // -----------------------------------------------------------------------
    // Sprint 4 — encoders for the mini_jit::Norm generator.
    // Each verified against the toolchain-assembled words of
    // rms_norm_ssve_v6.S / layer_norm_ssve_v6.S (see tests/test_norm.cpp,
    // tag [sprint4][encoders]) and the Arm ARM field layouts.
    // -----------------------------------------------------------------------

    //! condition codes for B.<cond> (Arm ARM C1.2.4)
    typedef enum : uint32_t {
      cond_eq = 0x0,   // equal  (SVE alias: B.NONE)
      cond_ne = 0x1,   // not equal (SVE alias: B.ANY)
      cond_hi = 0x8    // unsigned higher
    } cond_t;

    /**
     * @brief SMSTART SM — enter Streaming SVE mode WITHOUT enabling ZA.
     * Note: sme_smstart_sm() above actually encodes MSR SVCRSMZA,#1
     * (SM *and* ZA — needed by Gemm's fmopa); this is the SM-only form
     * the SSVE norm kernels use.
     */
    static uint32_t sme_smstart_sm_only();

    //! SMSTOP SM — exit Streaming SVE mode only (see sme_smstart_sm_only).
    static uint32_t sme_smstop_sm_only();

    //! STP X<rt>, X<rt2>, [SP/X<rn>, #simm]!  (pre-index; simm in bytes, /8)
    static uint32_t base_stp_pre_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm );

    //! STP X<rt>, X<rt2>, [X<rn>, #simm]  (signed offset; simm in bytes, /8)
    static uint32_t base_stp_off_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm );

    //! LDP X<rt>, X<rt2>, [X<rn>, #simm]  (signed offset; simm in bytes, /8)
    static uint32_t base_ldp_off_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm );

    //! LDP X<rt>, X<rt2>, [X<rn>], #simm  (post-index; simm in bytes, /8)
    static uint32_t base_ldp_post_x( gpr_t rt, gpr_t rt2, gpr_t rn, int32_t simm );

    //! STR X<rt>, [X<rn>, #uimm]  (unsigned offset; uimm in bytes, /8)
    static uint32_t base_str_imm_x( gpr_t rt, gpr_t rn, uint32_t uimm );

    //! LDR X<rt>, [X<rn>, #uimm]  (unsigned offset; uimm in bytes, /8)
    static uint32_t base_ldr_imm_x( gpr_t rt, gpr_t rn, uint32_t uimm );

    //! STR D<vt>, [X<rn>, #uimm]  (SIMD&FP 64-bit; uimm in bytes, /8)
    static uint32_t simd_str_imm_d( simd_fp_t vt, gpr_t rn, uint32_t uimm );

    //! LDR D<vt>, [X<rn>, #uimm]
    static uint32_t simd_ldr_imm_d( simd_fp_t vt, gpr_t rn, uint32_t uimm );

    //! STR S<vt>, [X<rn>, #uimm]  (SIMD&FP 32-bit; uimm in bytes, /4)
    static uint32_t simd_str_imm_s( simd_fp_t vt, gpr_t rn, uint32_t uimm );

    //! LDR S<vt>, [X<rn>, #uimm]
    static uint32_t simd_ldr_imm_s( simd_fp_t vt, gpr_t rn, uint32_t uimm );

    //! MOV X<rd>, X<rm>  (alias of ORR X<rd>, XZR, X<rm>)
    static uint32_t base_mov_reg_x( gpr_t rd, gpr_t rm );

    //! CMP X<rn>, X<rm>  (alias of SUBS XZR, X<rn>, X<rm>)
    static uint32_t base_cmp_reg_x( gpr_t rn, gpr_t rm );

    //! B.<cond> — offset19 in instructions from this instruction.
    static uint32_t base_b_cond( cond_t cond, int32_t offset19 );

    //! B — unconditional branch; offset26 in instructions.
    static uint32_t base_b( int32_t offset26 );

    //! CBZ X<rt>, <label> — offset19 in instructions.
    static uint32_t base_br_cbz_x( gpr_t rt, int32_t imm19 );

    //! FMOV S<vd>, S<vn>
    static uint32_t fp_fmov_s( simd_fp_t vd, simd_fp_t vn );

    //! FMOV S<vd>, #imm8 — imm8 is the encoded VFPExpandImm byte (1.0 = 0x70).
    static uint32_t fp_fmov_imm_s( simd_fp_t vd, uint32_t imm8 );

    //! SCVTF S<vd>, X<rn>  (signed 64-bit int → fp32)
    static uint32_t fp_scvtf_s_x( simd_fp_t vd, gpr_t rn );

    //! FDIV S<vd>, S<vn>, S<vm>
    static uint32_t fp_fdiv_s( simd_fp_t vd, simd_fp_t vn, simd_fp_t vm );

    //! CNTW X<rd>  (pattern ALL, mul #1) — fp32 lanes per vector.
    static uint32_t sve_cntw_x( gpr_t rd );

    //! INCW X<rd>  (pattern ALL, mul #1)
    static uint32_t sve_incw_x( gpr_t rd );

    //! ADDVL X<rd>, X<rn>, #simm6  (add simm6 * VL bytes)
    static uint32_t sve_addvl( gpr_t rd, gpr_t rn, int32_t simm6 );

    //! MOV Z<zd>.S, #simm8  (alias of DUP Z<zd>.S, #simm8)
    static uint32_t sve_dup_imm_s( sve_t zd, int32_t simm8 );

    //! DUP Z<zd>.S, Z<zn>.S[idx]  (broadcast element; idx 0-15 for fp32)
    static uint32_t sve_dup_elem_s( sve_t zd, sve_t zn, uint32_t idx );

    //! LD1W {Z<zt>.S}, P<pg>/Z, [X<rn>, #simm4, MUL VL]
    static uint32_t sve_ld1w_imm( sve_t zt, pred_t pg, gpr_t rn, int32_t simm4 );

    //! ST1W {Z<zt>.S}, P<pg>, [X<rn>, #simm4, MUL VL]
    static uint32_t sve_st1w_imm( sve_t zt, pred_t pg, gpr_t rn, int32_t simm4 );

    //! LD1RW {Z<zt>.S}, P<pg>/Z, [X<rn>, #uimm6*4]  (load + broadcast one fp32)
    static uint32_t sve_ld1rw_s( sve_t zt, pred_t pg, gpr_t rn, uint32_t uimm6 = 0 );

    //! FMLA Z<zda>.S, P<pg>/M, Z<zn>.S, Z<zm>.S
    static uint32_t sve_fmla_s( sve_t zda, pred_t pg, sve_t zn, sve_t zm );

    //! FADD Z<zdn>.S, P<pg>/M, Z<zdn>.S, Z<zm>.S  (predicated, destructive)
    static uint32_t sve_fadd_p_s( sve_t zdn, pred_t pg, sve_t zm );

    //! FSUB Z<zdn>.S, P<pg>/M, Z<zdn>.S, Z<zm>.S
    static uint32_t sve_fsub_p_s( sve_t zdn, pred_t pg, sve_t zm );

    //! FMUL Z<zdn>.S, P<pg>/M, Z<zdn>.S, Z<zm>.S
    static uint32_t sve_fmul_p_s( sve_t zdn, pred_t pg, sve_t zm );

    //! FMUL Z<zd>.S, Z<zn>.S, Z<zm>.S  (unpredicated)
    static uint32_t sve_fmul_s( sve_t zd, sve_t zn, sve_t zm );

    //! FRSQRTE Z<zd>.S, Z<zn>.S  (reciprocal sqrt estimate)
    static uint32_t sve_frsqrte_s( sve_t zd, sve_t zn );

    //! FRSQRTS Z<zd>.S, Z<zn>.S, Z<zm>.S  (Newton-Raphson step)
    static uint32_t sve_frsqrts_s( sve_t zd, sve_t zn, sve_t zm );

    //! WHILELO P<pd>.S, X<rn>, X<rm>  (64-bit scalar operands)
    static uint32_t sve_whilelo_s_x( pred_t pd, gpr_t rn, gpr_t rm );

    //! SEL Z<zd>.S, P<pg>, Z<zn>.S, Z<zm>.S
    static uint32_t sve_sel_s( sve_t zd, pred_t pg, sve_t zn, sve_t zm );
};

#endif