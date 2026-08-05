#ifndef MINI_JIT_TEIR_H
#define MINI_JIT_TEIR_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace mini_jit::teir {

    struct Axis {
        std::string name;
        uint32_t range;
        // Strides: How many floats to skip for each pointer when this axis increments
        uint64_t stride_a = 0;
        uint64_t stride_b = 0;
        uint64_t stride_c = 0;
        uint64_t stride_d = 0;  // 4th data stream (LayerNorm's beta)
    };

    struct Node {
        virtual ~Node() = default;
    };

    struct Iteration : public Node {
        Axis* axis;
        std::shared_ptr<Node> body;
        bool is_parallel = false;
    };

    // Multiple sibling nodes run in sequence under one Iteration (e.g.
    // contraction.teir's `children [@inv_zero, @inv_gemm]`) — Iteration::body
    // is a single Node, so a schedule step with several children wraps them
    // in a Sequence.
    struct Sequence : public Node {
        std::vector<std::shared_ptr<Node>> children;
    };

    struct Invocation : public Node {
        std::string kernel_name;

        // Static per-invocation parameters for kernels whose call signature
        // doesn't fit the (a,b,c,16,16[,16]) fixed-tile convention that
        // identity/gemm use.  The norm primitives take a runtime row count,
        // feature count, leading dimensions and epsilon (decision F in
        // context.md: the kernel operates tile-at-a-time and must never
        // assume it owns the whole tensor, so these travel per invocation
        // rather than being hardcoded in the runtime's dispatch).
        int64_t m = 0, n = 0, k = 0, ld_a = 0, ld_b = 0, ld_c = 0;
        float   eps = 1e-5f;

        // Storage-order flags for the GEMM operands, derived by the parser
        // from each tensor's axis strides in the .teir file (0 = column-
        // major, 1 = row-major — the week6 Gemm::generate convention).
        uint32_t trans_a = 0, trans_b = 0, trans_c = 0;

        // `guard first(@axis)` (contraction.teir): only fires when the
        // current iteration index of guard_axis is 0. nullptr = unguarded.
        Axis* guard_axis = nullptr;
    };

    // Reusable function pointer type. Widened to 4 pointers so one type
    // covers every primitive: d is unused (nullptr) by the 3-pointer ones
    // (identity/gemm/transposition/contraction) and carries LayerNorm's
    // beta for the norm primitives — no separate "4-pointer variant".
    using CompiledKernel = std::function<void(float* a, float* b, float* c, float* d)>;

    struct RuntimeContext {
        float *data_a, *data_b, *data_c, *data_d = nullptr;
        // Current iteration index per axis, for guard evaluation.
        std::map<const Axis*, uint32_t> axis_index;
    };
}

#endif