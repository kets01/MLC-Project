#include <vector>
#include <cstdint>
#include "week5/jit_kernel.hpp"

std::vector<uint32_t> get_identity_jit_opcodes() {
    return {
        // 0000000000000000 <ltmp0>:
        0xd503437f,      //smstart  sm
        0x2598e120,      //ptrue    p0.s, vl16
        0x34000184,      //cbz      w4, 0x38 <ltmp0+0x38>
        0x04a1401f,      //index    z31.s, #0x0, #0x1
        0x05a0387e,      //mov      z30.s, w3
        0x049003df,      //mul      z31.s, p0/m, z31.s, z30.s
        0xd2800205,      //mov      x5, #0x10               ; =16
        0xa540a000,      //ld1w     { z0.s }, p0/z, [x0]
        0xe57f8020,      //st1w     { z0.s }, p0, [x1, z31.s, uxtw #2]
        0x8b020800,      //add      x0, x0, x2, lsl #2
        0x91001021,      //add      x1, x1, #0x4
        0xf10004a5,      //subs     x5, x5, #0x1
        0x54ffff61,      //b.ne     0x1c <ltmp0+0x1c>
        0x14000014,      //b        0x84 <ltmp0+0x84>
        0xd2800085,      //mov      x5, #0x4                ; =4
        0xa540a000,      //ld1w     { z0.s }, p0/z, [x0]
        0x8b020806,      //add      x6, x0, x2, lsl #2
        0xa540a0c1,      //ld1w     { z1.s }, p0/z, [x6]
        0x8b0208c7,      //add      x7, x6, x2, lsl #2
        0xa540a0e2,      //ld1w     { z2.s }, p0/z, [x7]
        0x8b0208e8,      //add      x8, x7, x2, lsl #2
        0xa540a103,      //ld1w     { z3.s }, p0/z, [x8]
        0xe540e020,      //st1w     { z0.s }, p0, [x1]
        0x8b030829,      //add      x9, x1, x3, lsl #2
        0xe540e121,      //st1w     { z1.s }, p0, [x9]
        0x8b03092a,      //add      x10, x9, x3, lsl #2
        0xe540e142,      //st1w     { z2.s }, p0, [x10]
        0x8b03094b,      //add      x11, x10, x3, lsl #2
        0xe540e163,      //st1w     { z3.s }, p0, [x11]
        0x8b020900,      //add      x0, x8, x2, lsl #2
        0x8b030961,      //add      x1, x11, x3, lsl #2
        0xf10004a5,      //subs     x5, x5, #0x1
        0x54fffde1,      //b.ne     0x3c <ltmp0+0x3c>
        0xd503427f,      //smstop   sm
        0xd65f03c0,      //ret
    };
}

std::vector<uint32_t> get_zero_jit_opcodes() {
    return {
        // 0000000000000000 <zero_16_16>:
        0xd503437f,      //smstart  sm
        0x2598e120,      //ptrue    p0.s, vl16
        0x25b8c000,      //mov      z0.s, #0x0              ; =0
        0xd2800082,      //mov      x2, #0x4                ; =4
        0xe540e000,      //st1w     { z0.s }, p0, [x0]
        0x8b010803,      //add      x3, x0, x1, lsl #2
        0xe540e060,      //st1w     { z0.s }, p0, [x3]
        0x8b010864,      //add      x4, x3, x1, lsl #2
        0xe540e080,      //st1w     { z0.s }, p0, [x4]
        0x8b010885,      //add      x5, x4, x1, lsl #2
        0xe540e0a0,      //st1w     { z0.s }, p0, [x5]
        0x8b0108a0,      //add      x0, x5, x1, lsl #2
        0xf1000442,      //subs     x2, x2, #0x1
        0x54fffee1,      //b.ne     0x10 <zero_16_16+0x10>
        0xd503427f,      //smstop   sm
        0xd65f03c0,      //ret
    };
}

std::vector<uint32_t> get_relu_jit_opcodes() {
    return {
        // 0000000000000000 <ltmp0>:
        0xd503437f,      //smstart  sm
        0x2598e120,      //ptrue    p0.s, vl16
        0x25b8c01f,      //mov      z31.s, #0x0             ; =0
        0x340001a4,      //cbz      w4, 0x40 <ltmp0+0x40>
        0x04a1401d,      //index    z29.s, #0x0, #0x1
        0x05a0387e,      //mov      z30.s, w3
        0x049003dd,      //mul      z29.s, p0/m, z29.s, z30.s
        0xd2800205,      //mov      x5, #0x10               ; =16
        0xa540a000,      //ld1w     { z0.s }, p0/z, [x0]
        0x658683e0,      //fmax     z0.s, p0/m, z0.s, z31.s
        0xe57d8020,      //st1w     { z0.s }, p0, [x1, z29.s, uxtw #2]
        0x8b020800,      //add      x0, x0, x2, lsl #2
        0x91001021,      //add      x1, x1, #0x4
        0xf10004a5,      //subs     x5, x5, #0x1
        0x54ffff41,      //b.ne     0x20 <ltmp0+0x20>
        0x14000018,      //b        0x9c <ltmp0+0x9c>
        0xd2800085,      //mov      x5, #0x4                ; =4
        0xa540a000,      //ld1w     { z0.s }, p0/z, [x0]
        0x8b020806,      //add      x6, x0, x2, lsl #2
        0xa540a0c1,      //ld1w     { z1.s }, p0/z, [x6]
        0x8b0208c7,      //add      x7, x6, x2, lsl #2
        0xa540a0e2,      //ld1w     { z2.s }, p0/z, [x7]
        0x8b0208e8,      //add      x8, x7, x2, lsl #2
        0xa540a103,      //ld1w     { z3.s }, p0/z, [x8]
        0x658683e0,      //fmax     z0.s, p0/m, z0.s, z31.s
        0x658683e1,      //fmax     z1.s, p0/m, z1.s, z31.s
        0x658683e2,      //fmax     z2.s, p0/m, z2.s, z31.s
        0x658683e3,      //fmax     z3.s, p0/m, z3.s, z31.s
        0xe540e020,      //st1w     { z0.s }, p0, [x1]
        0x8b030829,      //add      x9, x1, x3, lsl #2
        0xe540e121,      //st1w     { z1.s }, p0, [x9]
        0x8b03092a,      //add      x10, x9, x3, lsl #2
        0xe540e142,      //st1w     { z2.s }, p0, [x10]
        0x8b03094b,      //add      x11, x10, x3, lsl #2
        0xe540e163,      //st1w     { z3.s }, p0, [x11]
        0x8b020900,      //add      x0, x8, x2, lsl #2
        0x8b030961,      //add      x1, x11, x3, lsl #2
        0xf10004a5,      //subs     x5, x5, #0x1
        0x54fffd61,      //b.ne     0x44 <ltmp0+0x44>
        0xd503427f,      //smstop   sm
        0xd65f03c0,      //ret
    };
}

std::vector<uint32_t> get_gemm_jit_opcodes() {
    return {
        // 0000000000000000 <ltmp0>:
        0xd503477f,      //smstart
        0x2598e3e0,      //ptrue    p0.s
        0xd2800011,      //mov      x17, #0x0               ; =0
        // 000000000000000c <n_loop>:
        0xd280000f,      //mov      x15, #0x0               ; =0
        // 0000000000000010 <m_loop>:
        0x9b057e29,      //mul      x9, x17, x5
        0x8b09084c,      //add      x12, x2, x9, lsl #2
        0x8b0f098c,      //add      x12, x12, x15, lsl #2
        0xaa0c03ed,      //mov      x13, x12
        0x5280000e,      //mov      w14, #0x0               ; =0
        0xa540a1a0,      //ld1w     { z0.s }, p0/z, [x13]
        0xa541a1a1,      //ld1w     { z1.s }, p0/z, [x13, #0x1, mul vl]
        0xc080c000,      //mov      za0v.s[w14, 0], p0/m, z0.s
        0xc080c028,      //mov      za2v.s[w14, 0], p0/m, z1.s
        0x8b0509ad,      //add      x13, x13, x5, lsl #2
        0x110005ce,      //add      w14, w14, #0x1
        0x710041df,      //cmp      w14, #0x10
        0x54ffff21,      //b.ne     0x24 <m_loop+0x14>
        0x5280000e,      //mov      w14, #0x0               ; =0
        0xa540a1a0,      //ld1w     { z0.s }, p0/z, [x13]
        0xa541a1a1,      //ld1w     { z1.s }, p0/z, [x13, #0x1, mul vl]
        0xc080c004,      //mov      za1v.s[w14, 0], p0/m, z0.s
        0xc080c02c,      //mov      za3v.s[w14, 0], p0/m, z1.s
        0x8b0509ad,      //add      x13, x13, x5, lsl #2
        0x110005ce,      //add      w14, w14, #0x1
        0x710041df,      //cmp      w14, #0x10
        0x54ffff21,      //b.ne     0x48 <m_loop+0x38>
        0x8b0f080a,      //add      x10, x0, x15, lsl #2
        0x8b11082b,      //add      x11, x1, x17, lsl #2
        0xd2804009,      //mov      x9, #0x200              ; =512
        // 0000000000000074 <k_loop>:
        0xa540a140,      //ld1w     { z0.s }, p0/z, [x10]
        0xa541a141,      //ld1w     { z1.s }, p0/z, [x10, #0x1, mul vl]
        0xa540a162,      //ld1w     { z2.s }, p0/z, [x11]
        0xa541a163,      //ld1w     { z3.s }, p0/z, [x11, #0x1, mul vl]
        0x80820000,      //fmopa    za0.s, p0/m, p0/m, z0.s, z2.s
        0x80830001,      //fmopa    za1.s, p0/m, p0/m, z0.s, z3.s
        0x80820022,      //fmopa    za2.s, p0/m, p0/m, z1.s, z2.s
        0x80830023,      //fmopa    za3.s, p0/m, p0/m, z1.s, z3.s
        0x8b03094a,      //add      x10, x10, x3, lsl #2
        0x8b04096b,      //add      x11, x11, x4, lsl #2
        0xf1000529,      //subs     x9, x9, #0x1
        0x54fffea1,      //b.ne     0x74 <k_loop>
        0xaa0c03ed,      //mov      x13, x12
        0x5280000e,      //mov      w14, #0x0               ; =0
        0xc082c000,      //mov      z0.s, p0/m, za0v.s[w14, 0]
        0xc082c101,      //mov      z1.s, p0/m, za2v.s[w14, 0]
        0xe540e1a0,      //st1w     { z0.s }, p0, [x13]
        0xe541e1a1,      //st1w     { z1.s }, p0, [x13, #0x1, mul vl]
        0x8b0509ad,      //add      x13, x13, x5, lsl #2
        0x110005ce,      //add      w14, w14, #0x1
        0x710041df,      //cmp      w14, #0x10
        0x54ffff21,      //b.ne     0xac <k_loop+0x38>
        0x5280000e,      //mov      w14, #0x0               ; =0
        0xc082c080,      //mov      z0.s, p0/m, za1v.s[w14, 0]
        0xc082c181,      //mov      z1.s, p0/m, za3v.s[w14, 0]
        0xe540e1a0,      //st1w     { z0.s }, p0, [x13]
        0xe541e1a1,      //st1w     { z1.s }, p0, [x13, #0x1, mul vl]
        0x8b0509ad,      //add      x13, x13, x5, lsl #2
        0x110005ce,      //add      w14, w14, #0x1
        0x710041df,      //cmp      w14, #0x10
        0x54ffff21,      //b.ne     0xd0 <k_loop+0x5c>
        0x910081ef,      //add      x15, x15, #0x20
        0xf10801ff,      //cmp      x15, #0x200
        0x54fff8c1,      //b.ne     0x10 <m_loop>
        0x91008231,      //add      x17, x17, #0x20
        0xf108023f,      //cmp      x17, #0x200
        0x54fff841,      //b.ne     0xc <n_loop>
        0xd503467f,      //smstop
        0xd65f03c0,      //ret
    };
}