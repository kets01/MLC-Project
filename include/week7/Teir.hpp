
#ifndef MINI_JIT_TEIR_HPP
#define MINI_JIT_TEIR_HPP

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <cstdint>

namespace mini_jit::teir {

// ──────────────────────────────────────────────────────────────────────────────
// Data-type tag
// ──────────────────────────────────────────────────────────────────────────────
enum class DataType { f32, f64 };

// ──────────────────────────────────────────────────────────────────────────────
// Axis
//   name    – identifier used in the TEIR source
//   extent  – loop trip-count
//   strides – distance in *elements* (not bytes) for each tensor pointer when
//             this axis advances by 1.  Missing tensors have stride 0.
// ──────────────────────────────────────────────────────────────────────────────
struct Axis {
    std::string name;
    uint32_t    extent = 0;
    // Per-tensor strides, keyed by tensor name ("in0","in1","in","out", …)
    std::map<std::string, int64_t> strides;
};

// ──────────────────────────────────────────────────────────────────────────────
// Primitive kinds
// ──────────────────────────────────────────────────────────────────────────────
enum class PrimKind { Zero, Copy, Contraction };

// Axis-role assignment for a primitive  (M/N/K axes)
struct PrimAxes {
    std::vector<Axis*> M, N, K;   // K may be empty for Zero/Copy
};

struct Primitive {
    std::string name;
    PrimKind    kind;
    PrimAxes    axes;
    DataType    data_type = DataType::f32;
};

// ──────────────────────────────────────────────────────────────────────────────
// Schedule nodes
// ──────────────────────────────────────────────────────────────────────────────

struct Node { virtual ~Node() = default; };

// Sequencing: execute children in order
struct Sequence : public Node {
    std::vector<std::shared_ptr<Node>> children;
};

// Iteration over one axis
enum class Policy { sequential, parallel };

struct Iteration : public Node {
    std::string               name;
    int64_t ld_a = 16;
    int64_t ld_b = 16;
    int64_t ld_c = 16;
    Axis*                     axis;
    std::unique_ptr<Axis> axis;
    Policy                    policy   = Policy::sequential;
    std::shared_ptr<Node>     body;      // may be Sequence
};

// Guard kinds – currently only first(axis) is defined in the spec
enum class GuardKind { none, first };

struct Invocation : public Node {
    std::string  name;
    Primitive*   primitive = nullptr;

    // Optional guard: skip invocation unless condition holds
    GuardKind    guard      = GuardKind::none;
    Axis*        guard_axis = nullptr; // the axis referenced by first(axis)
};

// ──────────────────────────────────────────────────────────────────────────────
// Top-level TEIR object (one per .teir file)
// ──────────────────────────────────────────────────────────────────────────────
struct TeirObject {
    std::string name;

    // Ordered tensor names in the order they appear as function arguments
    std::vector<std::string>                  tensor_names; // e.g. {"in0","in1","out"}
    std::map<std::string, Axis>               axes;
    std::map<std::string, Primitive>          primitives;
    std::vector<std::shared_ptr<Node>>        roots;        // ordered root nodes
};

// ──────────────────────────────────────────────────────────────────────────────
// Compiled kernel type
//   Pointers are passed in the order they appear in TeirObject::tensor_names.
//   The caller is responsible for allocating correctly-sized buffers.
// ──────────────────────────────────────────────────────────────────────────────
using KernelFn = std::function<void(float** tensors)>;

} // namespace mini_jit::teir
#endif // MINI_JIT_TEIR_H