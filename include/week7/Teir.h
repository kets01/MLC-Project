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
    };

    struct Node {
        virtual ~Node() = default;
    };

    struct Iteration : public Node {
        Axis* axis;
        std::shared_ptr<Node> body;
        bool is_parallel = false;
    };

    struct Invocation : public Node {
        std::string kernel_name;
    };

    // Reusable function pointer type
    using CompiledKernel = std::function<void(float* a, float* b, float* c)>;

    struct RuntimeContext {
        float *data_a, *data_b, *data_c;
    };
}

#endif