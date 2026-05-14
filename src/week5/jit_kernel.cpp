#include <vector>
#include <cstdint>
#include "week5/jit_kernel.hpp"

std::vector<uint32_t> get_identity_jit_opcodes() {
    return {
        0xd50343df, // smstart (both)
        0x25381c00, // ptrue p0.s, vl16
        0xd2000209, // mov x9, #16
        0xa5404000, // 1: ld1w {z0.s}, p0/z, [x0]
        0xe4404020, // st1w {z0.s}, p0, [x1]
        0x8b220800, // add x0, x0, x2, lsl #2
        0x8b230821, // add x1, x1, x3, lsl #2
        0xf1000529, // subs x9, x9, #1
        0x54ffff21, // b.ne 1b
        0xd50342df, // smstop (both)
        0xd65f03c0  // ret
    };
}

std::vector<uint32_t> get_zero_jit_opcodes() {
    return {
        0xd50343df, 0x25381c00, 0x0528c000, 0xd2000209, 
        0xe4404000, 0x8b210800, 0xf1000529, 0x54ffff61, 
        0xd50342df, 0xd65f03c0
    };
}

std::vector<uint32_t> get_relu_jit_opcodes() {
    return {
        0xd50343df, 0x25381c00, 0x0528c01f, 0xd2000209, 0xa5404000, 
        0x6520a3e0, 0xe4404020, 0x8b220800, 0x8b230821, 0xf1000529, 
        0x54ffff21, 0xd50342df, 0xd65f03c0
    };
}

std::vector<uint32_t> get_gemm_jit_opcodes() {
    return {
        0xd50343df, // smstart (CRITICAL: Must enable ZA state)
        0x25381c00, // ptrue p0.s, vl16
        0xd2000011, // mov x17, #0
        0xd200000f, // 1: mov x15, #0
        0x9b057e29, // 2: mul x9, x17, x5
        0x8b09084c, // add x12, x2, x9, lsl #2
        0x8b0f098c, // add x12, x12, x15, lsl #2
        0x8b0f080a, // add x10, x0, x15, lsl #2
        0x8b11082b, // add x11, x1, x17, lsl #2
        0xd2004009, // mov x9, #512
        0xa5404140, // 3: ld1w {z0.s}, p0/z, [x10]
        0xa5404162, // ld1w {z2.s}, p0/z, [x11]
        0x80a2a000, // fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        0x8b03094a, // add x10, x10, x3, lsl #2
        0x8b04096b, // add x11, x11, x4, lsl #2
        0xf1000529, // subs x9, x9, #1
        0x54ffff41, // b.ne 3b
        0x910081ef, // add x15, x15, #32
        0xf10401ff, // cmp x15, #512
        0x54fffcc1, // b.ne 2b
        0x91008231, // add x17, x17, #32
        0xf104023f, // cmp x17, #512
        0x54fffba1, // b.ne 1b
        0xd50342df, // smstop
        0xd65f03c0  // ret
    };
}