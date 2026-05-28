#ifndef MINI_JIT_TEIR_H
#define MINI_JIT_TEIR_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace mini_jit::teir {

    // Represents a dimension/loop axis
    struct Axis {
        std::string name;
        uint32_t range;
        // Memory increments (strides) for the three data pointers
        uint64_t stride_a = 0;
        uint64_t stride_b = 0;
        uint64_t stride_c = 0;
    };

    struct Node {
        virtual ~Node() = default;
    };

    // Iteration node represents a 'for' loop
    struct Iteration : public Node {
        Axis* axis;
        std::shared_ptr<Node> body;
        bool is_parallel = false; // OpenMP trigger
    };

    // Invocation node represents the leaf where the Week 6 kernel is called
    struct Invocation : public Node {
        std::string kernel_name;
    };

    // Reusable function pointer type as required by the task
    using CompiledKernel = std::function<void(float* a, float* b, float* c)>;

    // Execution context to track the moving memory pointers
    struct RuntimeContext {
        float *data_a, *data_b, *data_c;
    };
}

#endif