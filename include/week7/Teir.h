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

    // The "reusable function pointer" type required by the task
    using CompiledKernel = std::function<void(float* a, float* b, float* c)>;

    struct RuntimeContext {
        std::map<std::string, uint32_t> indices;
        float *data_a, *data_b, *data_c;
    };
}

#endif