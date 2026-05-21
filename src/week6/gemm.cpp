#include "week6/gemm.hpp"
#include "week5/jit_engine.hpp"
#include <vector>

using namespace mini_jit;

// ---------------------------------------------------------------------------
// Register allocation (AArch64 calling convention + SME constraints)
//
//  X0  = A        (caller-provided, do not clobber)
//  X1  = B        (caller-provided, do not clobber)
//  X2  = C        (caller-provided, do not clobber)
//  X3  = ld_a     (caller-provided, do not clobber)
//  X4  = ld_b     (caller-provided, do not clobber)
//  X5  = ld_c     (caller-provided, do not clobber)
//
//  X9  = ptr_a    (current A column/row pointer inside k-loop)
//  X10 = ptr_b    (current B row/column pointer inside k-loop)
//  X11 = ptr_c    (current C tile pointer for stores)
//
//  X12 = stride_a_bytes  (ld_a * esize, pre-computed)
//  X13 = stride_b_bytes  (ld_b * esize, pre-computed)
//  X14 = stride_c_bytes  (ld_c * esize, pre-computed)
//
//  X15 = scratch / immediate helper
//  X16 = scratch / madd result
//  X17 = scratch / address computation
//
//  X8  = k loop counter
//
//  W12–W15 are the ONLY registers allowed as ZA slice index base registers
//  (architectural constraint of SME mova/st1w indexed forms).
//  We therefore MUST NOT use X12 as anything else while the store loop runs.
//  Since stride_a/b/c are only needed in the k-loop (before the store loop),
//  and the store loop only needs slice indices in W12, there is no conflict as
//  long as we never mix their lifetimes.  We still document this carefully.
//
//  ZA tiles used (fp32, SVL=512 bit → 16×16 elements per tile):
//   za0.s  za1.s
//   za2.s  za3.s
//  → 2×2 tile block covering (2*step) rows of A and (2*step) cols of B
//    per outer iteration; 4 FMOPAs per (2 A-loads + 2 B-loads).
// ---------------------------------------------------------------------------

// ---- tiny helpers ---------------------------------------------------------

// Emit MOVZ + optional MOVK to load a 32-bit immediate into Xreg.
static void emit_mov_imm32(std::vector<uint32_t>& ops, int reg, uint32_t val)
{
    // MOVZ X<reg>, #<val[15:0]>
    // Encoding: sf=1 opc=10 hw=00  →  0xD280_0000 | (imm16 << 5) | Rd
    ops.push_back(0xd2800000u | ((val & 0xFFFFu) << 5) | (uint32_t)reg);

    if (val > 0xFFFFu) {
        // MOVK X<reg>, #<val[31:16]>, LSL #16
        // Encoding: sf=1 opc=11 hw=01  →  0xF2A0_0000 | (imm16 << 5) | Rd
        ops.push_back(0xf2a00000u | (((val >> 16) & 0xFFFFu) << 5) | (uint32_t)reg);
    }
}

// Emit:  ADD X<dst>, X<src>, X<idx>, LSL #shift2   (shifted-register form)
// Encoding: sf=1 S=0 shift=LSL op=ADD  →  0x8B000000 | (shift2<<10) | (Rm<<16) | (Rn<<5) | Rd
static void emit_add_lsl(std::vector<uint32_t>& ops, int dst, int src, int idx, int shift2)
{
    ops.push_back(0x8b000000u
                  | ((uint32_t)idx  << 16)
                  | ((uint32_t)shift2 << 10)
                  | ((uint32_t)src  <<  5)
                  | (uint32_t)dst);
}

// Emit:  ADD X<dst>, X<src>, #imm12   (immediate form, no shift)
// Encoding: sf=1 op=0 S=0  →  0x91000000 | (imm12<<10) | (Rn<<5) | Rd
static void emit_add_imm(std::vector<uint32_t>& ops, int dst, int src, uint32_t imm12)
{
    ops.push_back(0x91000000u | ((imm12 & 0xFFFu) << 10) | ((uint32_t)src << 5) | (uint32_t)dst);
}

// Emit:  MADD X<dst>, X<a>, X<b>, XZR   (i.e. X<dst> = X<a> * X<b>)
// Encoding: sf=1  →  0x9B007C00 | (Ra=31<<10) | (Rm<<16) | (Rn<<5) | Rd
static void emit_mul(std::vector<uint32_t>& ops, int dst, int a, int b)
{
    ops.push_back(0x9b007c00u | ((uint32_t)b << 16) | ((uint32_t)a << 5) | (uint32_t)dst);
}

// Emit:  ADD X<dst>, X<a>, X<b>   (shifted-register, shift=0)
static void emit_add_reg(std::vector<uint32_t>& ops, int dst, int a, int b)
{
    ops.push_back(0x8b000000u | ((uint32_t)b << 16) | ((uint32_t)a << 5) | (uint32_t)dst);
}

// ---------------------------------------------------------------------------
// Gemm::generate
// ---------------------------------------------------------------------------
Gemm::error_t Gemm::generate(uint32_t m, uint32_t n, uint32_t k,
                              uint32_t trans_a, uint32_t trans_b, uint32_t trans_c,
                              dtype_t dtype)
{
    std::vector<uint32_t> ops;
    ops.reserve(512);

    const uint32_t esize = (dtype == dtype_t::fp32) ? 4u : 8u;
    // SVL step in elements (512-bit SVL → 16 fp32 or 8 fp64)
    const uint32_t step  = (dtype == dtype_t::fp32) ? 16u : 8u;
    // log2(esize) used for LSL in address computations
    const int      lsl   = (dtype == dtype_t::fp32) ? 2 : 3;

    // -----------------------------------------------------------------------
    // Prologue
    // -----------------------------------------------------------------------

    // SMSTART SM          ; Enter streaming SVE mode
    // Encoding: MSR SVCR, #1  →  0xD503437F
    ops.push_back(0xd503477fu);

    // PTRUE P0.<T>, ALL   ; Activate all lanes in predicate p0
    if (dtype == dtype_t::fp32)
        ops.push_back(0x2598e3e0u); // ptrue p0.s, all
    else
        ops.push_back(0x25d8e3e0u); // ptrue p0.d, all

    // Pre-compute byte strides from element strides passed in X3/X4/X5
    //   X12 = X3 << lsl   ; stride_a_bytes = ld_a * esize
    //   X13 = X4 << lsl   ; stride_b_bytes = ld_b * esize
    //   X14 = X5 << lsl   ; stride_c_bytes = ld_c * esize
    // LSL Xd, Xn, #imm  is an alias for UBFM; encoding here uses the
    // standard shifted-register ADD trick with XZR … actually we use the
    // dedicated LSL (UBFM) form:
    // LSL X<d>, X<n>, #<s>  →  UBFM Xd, Xn, #(-s MOD 64), #(63-s)
    //  lsl x12, x3, #2  : imms=61(0x3d) immr=62(0x3e) N=1 sf=1
    //  Encoding: 0xD37E_F46C  (pre-verified for lsl #2, reg 12 from reg 3)
    if (lsl == 2) {
        ops.push_back(0xd37ef46cu); // LSL X12, X3, #2   (stride_a = ld_a*4)
        ops.push_back(0xd37ef48du); // LSL X13, X4, #2   (stride_b = ld_b*4)
        ops.push_back(0xd37ef4aeu); // LSL X14, X5, #2   (stride_c = ld_c*4)
    } else {
        ops.push_back(0xd37df46cu); // LSL X12, X3, #3   (stride_a = ld_a*8)
        ops.push_back(0xd37df48du); // LSL X13, X4, #3   (stride_b = ld_b*8)
        ops.push_back(0xd37df4aeu); // LSL X14, X5, #3   (stride_c = ld_c*8)
    }

    // -----------------------------------------------------------------------
    // Tile loops  –  2×2 blocking (za0–za3)
    //
    //  outer N-loop: j  in [0, n, 2*step)
    //  outer M-loop: i  in [0, m, 2*step)
    //    k-loop: accumulate into 4 ZA tiles
    //    store loop: write 4 tiles to C
    // -----------------------------------------------------------------------
    const uint32_t tile2 = 2 * step;   // 32 for fp32, 16 for fp64

    for (uint32_t j = 0; j < n; j += tile2) {
        for (uint32_t i = 0; i < m; i += tile2) {

            // ----------------------------------------------------------------
            // Zero all four ZA tiles
            // ----------------------------------------------------------------
            // ZERO {ZA}  ; zeroes entire ZA storage
            // Encoding: 0xC00800FF
            ops.push_back(0xc00800ffu); // ZERO {ZA}

            // ----------------------------------------------------------------
            // Set up initial A pointer:  X9 = X0 + i*esize
            // ----------------------------------------------------------------
            // MOVZ X15, #(i*esize)[15:0]
            // (+ MOVK if needed)
            emit_mov_imm32(ops, 15, i * esize);
            // ADD X9, X0, X15
            emit_add_reg(ops, 9, 0, 15);

            // ----------------------------------------------------------------
            // Set up initial B pointer for column j:
            //   X10 = X1 + j * ld_b * esize
            //       = X1 + j * X13  (X13 = stride_b_bytes = ld_b*esize)
            // We compute j*X13 via: X17 = j * X4 (element stride),
            //   then X10 = X1 + X17<<lsl
            //
            // MOVZ X15, #j
            // MUL  X17, X15, X4     ; X17 = j * ld_b  (element offset)
            // ADD  X10, X1, X17, LSL #lsl
            // ----------------------------------------------------------------
            emit_mov_imm32(ops, 15, j);
            emit_mul(ops, 17, 15, 4);       // MUL X17, X15, X4
            emit_add_lsl(ops, 10, 1, 17, lsl); // ADD X10, X1, X17, LSL #lsl

            // ----------------------------------------------------------------
            // Second A pointer for i+step:  X16 = X9 + step*esize
            // ----------------------------------------------------------------
            emit_add_imm(ops, 16, 9, step * esize); // ADD X16, X9, #(step*esize)

            // ----------------------------------------------------------------
            // Second B pointer for j+step:  use X13 (stride_b_bytes = ld_b*esize)
            //   ptr_b1 = X10 + step * ld_b * esize = X10 + step * X13
            // We embed this in the k-loop using a separate saved pointer.
            // Store it in a callee-saved-free scratch: X11 (we compute it now,
            // restore it each outer iteration from the formula).
            //
            // X11 = X10 + step * X13 / step ... simplest: emit_mov step, mul stride
            // Actually:  ptr_b1 = X10 + step*stride_b_bytes
            //   step*stride_b_bytes = step*ld_b*esize
            //   We can compute: X15 = step; X17 = X15 * X4 = step*ld_b; X11 = X1 + X17<<lsl + j<<lsl
            // Easier: X11 = X10 + step*X13 — but X13 is ld_b*esize and step is a constant.
            // Use: ADD X11, X10, #(step*esize)  only valid if ld_b==1 (packed).
            // General: step_b_bytes = step * stride_b_bytes = step * ld_b * esize
            //          which isn't known at JIT time … but step is constant.
            //   X17 = step * X4  then  X11 = X1 + j*X4<<lsl + X17<<lsl
            //   Reuse: X11 = X10 + X17<<lsl  where X17 = step*X4
            // ----------------------------------------------------------------
            emit_mov_imm32(ops, 15, step);
            emit_mul(ops, 17, 15, 4);           // MUL X17, X15, X4   ; step*ld_b
            emit_add_lsl(ops, 11, 10, 17, lsl); // ADD X11, X10, X17, LSL #lsl ; ptr_b1

            // Save base pointers so we can reset them after the k-loop if needed.
            // We keep them in place — X9 and X10 will advance each k iteration,
            // X16 and X11 likewise (same stride).  So we just snapshot at loop start.
            // (Nothing to save here — we will re-derive in the store phase.)

            // ----------------------------------------------------------------
            // K-Loop:  accumulate 4 outer products per iteration
            //
            //  za0.s += A[i:i+step,   k] ⊗ B[k, j:j+step]
            //  za1.s += A[i:i+step,   k] ⊗ B[k, j+step:j+2*step]
            //  za2.s += A[i+step:i+2*step, k] ⊗ B[k, j:j+step]
            //  za3.s += A[i+step:i+2*step, k] ⊗ B[k, j+step:j+2*step]
            //
            // Registers:
            //   Z0 = A column / row for rows i..i+step-1
            //   Z1 = A column / row for rows i+step..i+2*step-1
            //   Z2 = B row / col for cols j..j+step-1
            //   Z3 = B row / col for cols j+step..j+2*step-1
            //
            //   X9  = ptr_a0   (updated each k)
            //   X16 = ptr_a1   (updated each k)
            //   X10 = ptr_b0   (updated each k)
            //   X11 = ptr_b1   (updated each k)
            //   X8  = loop counter
            // ----------------------------------------------------------------

            // MOV X8, #k
            emit_mov_imm32(ops, 8, k);

            uint32_t loop_start = (uint32_t)ops.size();

            // -- Load A vectors --
            if (dtype == dtype_t::fp32) {
                // LD1W {Z0.S}, P0/Z, [X9]
                // Encoding: 0xA540A120  (z0, p0, [x9])
                ops.push_back(0xa540a120u); // LD1W {Z0.S}, P0/Z, [X9]

                // LD1W {Z1.S}, P0/Z, [X16]
                // Encoding base: LD1W {Zt.S}, Pg/Z, [Xn]  =  0xA540A000 | (Pg<<10) | (Rn<<5) | Zt
                // p0=0, x16=16, z1=1
                ops.push_back(0xa540a201u); // LD1W {Z1.S}, P0/Z, [X16]
            } else {
                ops.push_back(0xa560a120u); // LD1D {Z0.D}, P0/Z, [X9]
                ops.push_back(0xa560a201u); // LD1D {Z1.D}, P0/Z, [X16]
            }

            // -- Load B vectors --
            if (dtype == dtype_t::fp32) {
                // LD1W {Z2.S}, P0/Z, [X10]
                ops.push_back(0xa540a142u); // LD1W {Z2.S}, P0/Z, [X10]
                // LD1W {Z3.S}, P0/Z, [X11]
                ops.push_back(0xa540a163u); // LD1W {Z3.S}, P0/Z, [X11]
            } else {
                ops.push_back(0xa560a142u); // LD1D {Z2.D}, P0/Z, [X10]
                ops.push_back(0xa560a163u); // LD1D {Z3.D}, P0/Z, [X11]
            }

            // -- Outer products --
            if (dtype == dtype_t::fp32) {
                // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
                // Encoding: 0x80820000 | (Zm<<16) | (Pn<<13) | (Pm<<10) | (Zn<<5) | ZAd
                // za0=0 z0=0 z2=2 p0=0 p0=0
                ops.push_back(0x80820000u); // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S

                // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
                // za1=1 z0=0 z3=3
                ops.push_back(0x80830001u); // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S

                // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
                // za2=2 z1=1 z2=2
                ops.push_back(0x80820022u); // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S

                // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
                // za3=3 z1=1 z3=3
                ops.push_back(0x80830023u); // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
            } else {
                ops.push_back(0x81820000u); // FMOPA ZA0.D, P0/M, P0/M, Z0.D, Z2.D
                ops.push_back(0x81830001u); // FMOPA ZA1.D, P0/M, P0/M, Z0.D, Z3.D
                ops.push_back(0x81820022u); // FMOPA ZA2.D, P0/M, P0/M, Z1.D, Z2.D
                ops.push_back(0x81830023u); // FMOPA ZA3.D, P0/M, P0/M, Z1.D, Z3.D
            }

            // -- Advance A pointers by one K step --
            // trans_a==0 → column-major A: consecutive K elements are ld_a apart
            //              → advance by stride_a_bytes (X12)
            // trans_a==1 → row-major A: consecutive K elements are contiguous (esize apart)
            //              → advance by esize (immediate)
            if (trans_a == 0) {
                // ADD X9,  X9,  X12   ; ptr_a0 += stride_a_bytes
                ops.push_back(0x8b0c0129u); // ADD X9,  X9,  X12
                // ADD X16, X16, X12   ; ptr_a1 += stride_a_bytes
                ops.push_back(0x8b0c0210u); // ADD X16, X16, X12
            } else {
                // ADD X9,  X9,  #esize
                emit_add_imm(ops, 9,  9,  esize); // ADD X9,  X9,  #esize
                // ADD X16, X16, #esize
                emit_add_imm(ops, 16, 16, esize); // ADD X16, X16, #esize
            }

            // -- Advance B pointers by one K step --
            // trans_b==0 → column-major B: consecutive K elements are contiguous (esize apart)
            //              → advance by esize (immediate)
            // trans_b==1 → row-major B: consecutive K elements are ld_b apart
            //              → advance by stride_b_bytes (X13)
            if (trans_b == 0) {
                // ADD X10, X10, #esize
                emit_add_imm(ops, 10, 10, esize); // ADD X10, X10, #esize
                // ADD X11, X11, #esize
                emit_add_imm(ops, 11, 11, esize); // ADD X11, X11, #esize
            } else {
                // ADD X10, X10, X13   ; ptr_b0 += stride_b_bytes
                ops.push_back(0x8b0d014au); // ADD X10, X10, X13
                // ADD X11, X11, X13   ; ptr_b1 += stride_b_bytes
                ops.push_back(0x8b0d016bu); // ADD X11, X11, X13
            }

            // SUBS X8, X8, #1        ; decrement loop counter, set flags
            ops.push_back(0xf1000508u); // SUBS X8, X8, #1

            // B.NE loop_start
            // Encoding: 0x54000001 | (offset19 << 5)
            // offset = loop_start - current_pc  (in instructions)
            {
                int32_t offset = (int32_t)loop_start - (int32_t)ops.size();
                ops.push_back(0x54000001u | ((uint32_t)(offset & 0x7FFFFu) << 5)); // B.NE loop_start
            }

            // ----------------------------------------------------------------
            // Store phase:  write ZA tiles back to C
            //
            // C tile layout (column-major, trans_c==0):
            //   tile za0 → C[i..i+step-1,   j..j+step-1]
            //   tile za1 → C[i..i+step-1,   j+step..j+2*step-1]
            //   tile za2 → C[i+step..,       j..j+step-1]
            //   tile za3 → C[i+step..,       j+step..j+2*step-1]
            //
            // For each tile, iterate over its slices (vertical=columns for col-major).
            // The ZA slice index base MUST be in W12–W15 (architectural constraint).
            // We use W12 as the slice index.
            // X12 was stride_a_bytes — that lifetime is now over (k-loop done),
            // so reusing W12 here is safe.
            //
            // C pointer for tile (ti, tj):
            //   ptr_c = X2 + (j + tj*step)*ld_c*esize + (i + ti*step)*esize
            //         = X2 + ((j+tj*step)*X5 + (i+ti*step)) << lsl
            // ----------------------------------------------------------------

            // Helper lambda (encoded inline below) for each of the 4 sub-tiles:
            // ti ∈ {0,1}, tj ∈ {0,1} → ZA tile index = tj + ti*2
            for (int ti = 0; ti < 2; ++ti) {
                for (int tj = 0; tj < 2; ++tj) {
                    int za_idx = tj + ti * 2;   // 0,1,2,3

                    uint32_t row_off = i + (uint32_t)ti * step;  // first row element index
                    uint32_t col_off = j + (uint32_t)tj * step;  // first col element index

                    // ptr_c = X2 + (col_off * ld_c + row_off) * esize
                    //       = X2 + (col_off * X5 + row_off) << lsl
                    //
                    // MOVZ X15, #col_off
                    emit_mov_imm32(ops, 15, col_off);
                    // MUL X17, X15, X5     ; X17 = col_off * ld_c
                    emit_mul(ops, 17, 15, 5);
                    // ADD X17, X17, #row_off  (row_off fits in 12 bits for step≤512)
                    emit_add_imm(ops, 17, 17, row_off);
                    // ADD X11, X2, X17, LSL #lsl   ; X11 = C + (col_off*ld_c+row_off)*esize
                    emit_add_lsl(ops, 11, 2, 17, lsl);

                    // Iterate over slices of this tile
                    //
                    // ZA tile index encoding for MOVA (verified with llvm-mc):
                    //   Vertical:   za0v=0xC0828000  za1v=0xC0828080  za2v=0xC0828100  za3v=0xC0828180
                    //   Horizontal: za0h=0xC0820000  za1h=0xC0820080  za2h=0xC0820100  za3h=0xC0820180
                    // Tile index occupies bits [8:7], stepping by 0x80 — NOT bits [1:0].
                    const uint32_t mova_v_base = 0xc0828000u + (uint32_t)za_idx * 0x80u;
                    const uint32_t mova_h_base = 0xc0820000u + (uint32_t)za_idx * 0x80u;

                    for (uint32_t s = 0; s < step; ++s) {
                        // MOVZ W12, #s  →  0x52800000 | (imm16<<5) | Rd=12
                        // encodes slice index s into W12 for ZA indexed access
                        ops.push_back(0x5280000cu | (s << 5)); // movz w12, #s

                        if (trans_c == 0) {
                            // Column-major C: vertical slice of ZA = one column of C
                            // MOV Z0.S, P0/M, ZA<n>V.S[W12, 0]
                            // encoding: 0xC0828000 + za_idx*0x80  (slice index via W12)
                            ops.push_back(mova_v_base); // mov z0.s, p0/m, za<n>v.s[w12, 0]

                            // ST1W {Z0.S}, P0, [X11]
                            // encoding: [0x60,0xe1,0x40,0xe5] = 0xe540e160
                            ops.push_back(0xe540e160u); // st1w {z0.s}, p0, [x11]

                            // ADD X11, X11, X14   ; advance ptr_c by stride_c_bytes (next column)
                            // encoding: [0x6b,0x01,0x0e,0x8b] = 0x8b0e016b
                            ops.push_back(0x8b0e016bu); // add x11, x11, x14
                        } else {
                            // Row-major C: horizontal slice of ZA = one row of C
                            // MOV Z0.S, P0/M, ZA<n>H.S[W12, 0]
                            // encoding: 0xC0820000 + za_idx*0x80  (slice index via W12)
                            ops.push_back(mova_h_base); // mov z0.s, p0/m, za<n>h.s[w12, 0]

                            // ST1W {Z0.S}, P0, [X11]
                            ops.push_back(0xe540e160u); // st1w {z0.s}, p0, [x11]

                            // ADD X11, X11, #esize  ; advance ptr_c by one element (next row)
                            // encoding via emit_add_imm: 0x91000000 | (4<<10) | (11<<5) | 11
                            emit_add_imm(ops, 11, 11, esize); // add x11, x11, #esize
                        }
                    }
                } // tj
            } // ti

        } // i
    } // j

    // -----------------------------------------------------------------------
    // Epilogue
    // -----------------------------------------------------------------------

    // SMSTOP SM           ; Exit streaming SVE mode
    // Encoding: [0x7f,0x42,0x03,0xd5] = 0xD503427F  (verified llvm-mc)
    ops.push_back(0xd503467fu); // SMSTOP SM  [0x7f,0x42,0x03,0xd5]

    // RET                 ; Return to caller
    // Encoding: 0xD65F03C0
    ops.push_back(0xd65f03c0u); // RET

    m_kernel = JitEngine::generate<kernel_t>(ops);
    return error_t::success;
}

Gemm::kernel_t Gemm::get_kernel() const {
    return m_kernel;
}