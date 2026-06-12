#include "week7/TeirRuntime.hpp"

#include <stdexcept>
#include <sstream>
#include <numeric>
#include <iostream>

#if __has_include(<omp.h>)
#  include <omp.h>
#  define HAS_OMP 1
#else
#  define HAS_OMP 0
#endif

namespace mini_jit::teir {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int64_t ESIZE_F32 = 4; // bytes per float

// Byte stride of axis 'ax' on tensor 'tensor_name'; 0 if not present.
static int64_t byte_stride(const Axis* ax, const std::string& tensor_name) {
    auto it = ax->strides.find(tensor_name);
    return (it != ax->strides.end()) ? it->second : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
TeirRuntime::TeirRuntime() {}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel caches
// Each (M,N) or (M,N,K) triple is compiled once and reused.
// ─────────────────────────────────────────────────────────────────────────────
mini_jit::Unary::kernel_t TeirRuntime::get_zero_kernel(uint32_t M, uint32_t N) {
    KernelKey key{M, N, 0};
    auto it = zero_cache_.find(key);
    if (it != zero_cache_.end()) return it->second;

    unary_gen_.generate(M, N, /*trans_b=*/0,
                        mini_jit::Unary::dtype_t::fp32,
                        mini_jit::Unary::ptype_t::zero);
    auto k = unary_gen_.get_kernel();
    zero_cache_[key] = k;
    return k;
}

mini_jit::Unary::kernel_t TeirRuntime::get_copy_kernel(uint32_t M, uint32_t N) {
    KernelKey key{M, N, 0};
    auto it = copy_cache_.find(key);
    if (it != copy_cache_.end()) return it->second;

    unary_gen_.generate(M, N, /*trans_b=*/0,
                        mini_jit::Unary::dtype_t::fp32,
                        mini_jit::Unary::ptype_t::identity);
    auto k = unary_gen_.get_kernel();
    copy_cache_[key] = k;
    return k;
}

mini_jit::Gemm::kernel_t TeirRuntime::get_gemm_kernel(uint32_t M, uint32_t N, uint32_t K,
                                                        uint32_t trans_a, uint32_t trans_b,
                                                        uint32_t trans_c) {
    GemmKey key{M, N, K, trans_a, trans_b, trans_c};
    auto it = gemm_cache_.find(key);
    if (it != gemm_cache_.end()) return it->second;

    gemm_gen_.generate(M, N, K, trans_a, trans_b, trans_c,
                       mini_jit::Gemm::dtype_t::fp32);
    auto k = gemm_gen_.get_kernel();
    gemm_cache_[key] = k;
    return k;
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────
std::string TeirRuntime::validate_primitive(const Primitive& prim) const {
    if (prim.data_type != DataType::f32)
        return "Primitive @" + prim.name + ": only f32 is supported";

    if (prim.kind == PrimKind::Contraction) {
        if (prim.axes.M.size() != 1)
            return "Primitive @" + prim.name
                   + ": Contraction requires exactly 1 M axis (got "
                   + std::to_string(prim.axes.M.size()) + ")";
        if (prim.axes.N.size() != 1)
            return "Primitive @" + prim.name
                   + ": Contraction requires exactly 1 N axis (got "
                   + std::to_string(prim.axes.N.size()) + ")";
        if (prim.axes.K.size() != 1)
            return "Primitive @" + prim.name
                   + ": Contraction requires exactly 1 K axis (BRGEMM not yet supported, got "
                   + std::to_string(prim.axes.K.size()) + ")";
    } else {
        // Zero / Copy
        if (prim.axes.M.empty())
            return "Primitive @" + prim.name + ": M axes must be non-empty";
        if (prim.axes.N.empty())
            return "Primitive @" + prim.name + ": N axes must be non-empty";
    }
    return "";
}

// ─────────────────────────────────────────────────────────────────────────────
// Leading-dimension derivation from axis byte strides  (spec §10.6)
//
// For GEMM (col-major by convention):
//   A(in0): if M-role axis has byte_stride == ESIZE_F32 → col-major A,  lda = K_stride/esize
//            if K-role axis has byte_stride == ESIZE_F32 → row-major A,  lda = M_stride/esize, trans_a=1
//   B(in1): if K-role axis has byte_stride == ESIZE_F32 → col-major B,  ldb = N_stride/esize
//            if N-role axis has byte_stride == ESIZE_F32 → row-major B,  ldb = K_stride/esize, trans_b=1
//   C(out): if M-role axis has byte_stride == ESIZE_F32 → col-major C,  ldc = N_stride/esize
//            if N-role axis has byte_stride == ESIZE_F32 → row-major C,  ldc = M_stride/esize, trans_c=1
//
// For Zero: finds which of the two role axes is unit-stride on out → rows;
//           the other gives ld.
// For Copy: same logic applied to both in0/in and out independently.
// ─────────────────────────────────────────────────────────────────────────────

struct GemmParams {
    uint32_t M, N, K;
    uint32_t trans_a, trans_b, trans_c;
    int64_t  lda, ldb, ldc;
    // Names of tensors to use as A, B, C (from TeirObject::tensor_names order)
    std::string t_a, t_b, t_c;
};

struct ZeroParams {
    uint32_t M, N;  // rows, cols passed to Unary
    int64_t  ldb;   // leading dim of output
    std::string t_out;
};

struct CopyParams {
    uint32_t M, N;
    int64_t  lda, ldb;
    std::string t_src, t_dst;
};

static GemmParams derive_gemm(const Primitive& prim, const TeirObject& obj) {
    Axis* ax_m = prim.axes.M[0];
    Axis* ax_n = prim.axes.N[0];
    Axis* ax_k = prim.axes.K[0];

    // Identify tensor names (convention: in0=A, in1=B, out=C)
    // tensor_names order follows declaration in TEIR file.
    std::string t_a = "in0", t_b = "in1", t_c = "out";
    // Fallback for transposition-style teirs that use "in"/"out"
    if (obj.axes.empty()) {} // just use defaults
    // Check actual names present
    bool has_in0 = false, has_in1 = false;
    for (auto& tn : obj.tensor_names) {
        if (tn == "in0") has_in0 = true;
        if (tn == "in1") has_in1 = true;
    }
    if (!has_in0 && obj.tensor_names.size() >= 1) t_a = obj.tensor_names[0];
    if (!has_in1 && obj.tensor_names.size() >= 2) t_b = obj.tensor_names[1];

    // A = in0: M and K axes
    int64_t m_stride_a = byte_stride(ax_m, t_a);
    int64_t k_stride_a = byte_stride(ax_k, t_a);
    uint32_t trans_a; int64_t lda;
    if (m_stride_a == ESIZE_F32) {   // M unit → col-major A
        trans_a = 0; lda = k_stride_a / ESIZE_F32;
    } else {                          // K unit → row-major A
        trans_a = 1; lda = m_stride_a / ESIZE_F32;
    }

    // B = in1: K and N axes
    int64_t k_stride_b = byte_stride(ax_k, t_b);
    int64_t n_stride_b = byte_stride(ax_n, t_b);
    uint32_t trans_b; int64_t ldb;
    if (k_stride_b == ESIZE_F32) {   // K unit → col-major B
        trans_b = 0; ldb = n_stride_b / ESIZE_F32;
    } else {                          // N unit → row-major B
        trans_b = 1; ldb = k_stride_b / ESIZE_F32;
    }

    // C = out: M and N axes
    int64_t m_stride_c = byte_stride(ax_m, t_c);
    int64_t n_stride_c = byte_stride(ax_n, t_c);
    uint32_t trans_c; int64_t ldc;
    if (m_stride_c == ESIZE_F32) {   // M unit → col-major C
        trans_c = 0; ldc = n_stride_c / ESIZE_F32;
    } else {                          // N unit → row-major C
        trans_c = 1; ldc = m_stride_c / ESIZE_F32;
    }

    return { ax_m->extent, ax_n->extent, ax_k->extent,
             trans_a, trans_b, trans_c, lda, ldb, ldc,
             t_a, t_b, t_c };
}

static ZeroParams derive_zero(const Primitive& prim, const TeirObject& obj) {
    // Zero acts on 'out' only (last tensor by convention).
    std::string t_out = "out";
    if (obj.tensor_names.size() >= 1) t_out = obj.tensor_names.back();

    Axis* ax_m = prim.axes.M[0];
    Axis* ax_n = prim.axes.N[0];

    int64_t m_stride = byte_stride(ax_m, t_out);
    int64_t n_stride = byte_stride(ax_n, t_out);

    // Unit-stride axis is the row axis; the other gives the ld.
    uint32_t rows, cols; int64_t ldb;
    if (m_stride == ESIZE_F32) {
        rows = ax_m->extent; cols = ax_n->extent;
        ldb = n_stride / ESIZE_F32;
    } else {
        // n_stride == ESIZE_F32
        rows = ax_n->extent; cols = ax_m->extent;
        ldb = m_stride / ESIZE_F32;
    }
    return { rows, cols, ldb, t_out };
}

static CopyParams derive_copy(const Primitive& prim, const TeirObject& obj) {
    // Copy: in → out  (first two tensor names)
    std::string t_src = "in0", t_dst = "out";
    if (!obj.tensor_names.empty()) {
        for (auto& tn : obj.tensor_names)
            if (tn == "in" || tn == "in0") { t_src = tn; break; }
        t_dst = obj.tensor_names.back();
    }

    Axis* ax_m = prim.axes.M[0];
    Axis* ax_n = prim.axes.N[0];

    // Determine m (rows) and n (cols) from src unit-stride axis
    int64_t m_stride_src = byte_stride(ax_m, t_src);
    int64_t n_stride_src = byte_stride(ax_n, t_src);
    int64_t m_stride_dst = byte_stride(ax_m, t_dst);
    int64_t n_stride_dst = byte_stride(ax_n, t_dst);

    // Which axis is unit on src → that's the row axis
    uint32_t rows, cols;
    int64_t lda, ldb;

    if (n_stride_src == ESIZE_F32) {
        // N unit on src → col-major src with N as rows: rows=M, cols=N, lda=M_stride/esize
        rows = ax_m->extent; cols = ax_n->extent;
        lda  = m_stride_src / ESIZE_F32;
    } else {
        // M unit on src → col-major src with M as rows: rows=N, cols=M, lda=N_stride/esize
        rows = ax_n->extent; cols = ax_m->extent;
        lda  = n_stride_src / ESIZE_F32;
    }

    // dst leading dimension: stride of the col axis (the non-unit one)
    if (n_stride_dst == ESIZE_F32) {
        ldb = m_stride_dst / ESIZE_F32;
    } else {
        ldb = n_stride_dst / ESIZE_F32;
    }

    return { rows, cols, lda, ldb, t_src, t_dst };
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile
// ─────────────────────────────────────────────────────────────────────────────
TeirRuntime::CompileResult TeirRuntime::compile(const TeirObject& obj) {
    // Validate every primitive first
    for (auto& [name, prim] : obj.primitives) {
        std::string err = validate_primitive(prim);
        if (!err.empty()) return { nullptr, err };
    }
    if (obj.roots.empty())
        return { nullptr, "TEIR object has no roots" };
    if (obj.tensor_names.empty())
        return { nullptr, "TEIR object has no tensors" };

    // Pre-warm the kernel cache for every primitive so the first execution
    // doesn't pay JIT cost inside a parallel region.
    for (auto& [name, prim] : obj.primitives) {
        try {
            if (prim.kind == PrimKind::Zero) {
                auto zp = derive_zero(prim, obj);
                get_zero_kernel(zp.M, zp.N);
            } else if (prim.kind == PrimKind::Copy) {
                auto cp = derive_copy(prim, obj);
                get_copy_kernel(cp.M, cp.N);
            } else if (prim.kind == PrimKind::Contraction) {
                auto gp = derive_gemm(prim, obj);
                get_gemm_kernel(gp.M, gp.N, gp.K, gp.trans_a, gp.trans_b, gp.trans_c);
            }
        } catch (const std::exception& e) {
            return { nullptr,
                     std::string("Kernel generation failed for @") + name + ": " + e.what() };
        }
    }

    // Return a closure that captures a copy of obj and a pointer to this runtime.
    // NOTE: the closure captures 'this', so the TeirRuntime must outlive the kernel.
    return { [this, obj](float** tensors) {
        ExecContext ctx;
        for (size_t i = 0; i < obj.tensor_names.size(); ++i)
            ctx.ptrs[obj.tensor_names[i]] = tensors[i];
        for (auto& root : obj.roots)
            traverse(root.get(), ctx, obj);
    }, "" };
}

TeirRuntime::CompileResult TeirRuntime::compile_file(const std::string& path) {
    TeirObject obj = parse_file(path);
    return compile(obj);
}

// ─────────────────────────────────────────────────────────────────────────────
// Traversal
//
// Key rule from spec §10.4: pointer for tensor j at invocation =
//   base_j + sum_over_ancestor_axes( offset_j,ax + stride_j,ax * index_ax )
//
// The strides in TeirObject::Axis::strides are BYTE values (from the .teir file).
// ExecContext::ptrs are float*, so we advance by stride_bytes / sizeof(float).
// ─────────────────────────────────────────────────────────────────────────────
void TeirRuntime::traverse(const Node*       node,
                            ExecContext&      ctx,
                            const TeirObject& obj) const {
    if (!node) return;

    // ── Sequence (multiple children of an iteration node) ────────────────
    if (auto* seq = dynamic_cast<const Sequence*>(node)) {
        for (auto& child : seq->children)
            traverse(child.get(), ctx, obj);
        return;
    }

    // ── Iteration ────────────────────────────────────────────────────────
    if (auto* iter = dynamic_cast<const Iteration*>(node)) {
        Axis*    ax     = iter->axis;
        uint32_t extent = ax->extent;

        // Snapshot base pointers before the loop starts.
        // Each iteration i advances by i * stride[tensor] from the base.
        std::map<std::string, float*> base_ptrs = ctx.ptrs;

        // Lambda for sequential body
        auto run_body = [&](uint32_t i) {
            // Advance each tensor pointer by i * (byte_stride / sizeof(float))
            for (auto& [tname, base] : base_ptrs) {
                int64_t bstride = byte_stride(ax, tname);
                ctx.ptrs[tname] = base + (static_cast<int64_t>(i) * bstride) / ESIZE_F32;
            }
            ctx.iter_idx[ax->name] = i;
            traverse(iter->body.get(), ctx, obj);
        };

        if (iter->policy == Policy::parallel) {
#if HAS_OMP
            #pragma omp parallel for schedule(static)
            for (uint32_t i = 0; i < extent; ++i) {
                // Each thread gets its own private copy of the context.
                // We pre-compute this thread's pointers from the base snapshot.
                ExecContext local_ctx;
                local_ctx.iter_idx = ctx.iter_idx;    // copy ancestor indices
                for (auto& [tname, base] : base_ptrs) {
                    int64_t bstride = byte_stride(ax, tname);
                    local_ctx.ptrs[tname] = base + (static_cast<int64_t>(i) * bstride) / ESIZE_F32;
                }
                local_ctx.iter_idx[ax->name] = i;
                traverse(iter->body.get(), local_ctx, obj);
            }
#else
            for (uint32_t i = 0; i < extent; ++i) run_body(i);
#endif
        } else {
            for (uint32_t i = 0; i < extent; ++i) run_body(i);
        }

        // Restore context pointers to the pre-loop state.
        ctx.ptrs = base_ptrs;
        return;
    }

    // ── Invocation ───────────────────────────────────────────────────────
    if (auto* inv = dynamic_cast<const Invocation*>(node)) {
        // Evaluate guard: first(axis) → only execute when that axis index == 0
        if (inv->guard == GuardKind::first && inv->guard_axis) {
            auto it = ctx.iter_idx.find(inv->guard_axis->name);
            if (it == ctx.iter_idx.end() || it->second != 0)
                return;   // guard not satisfied
        }
        invoke_primitive(inv, ctx, obj);
        return;
    }

    throw std::runtime_error("TeirRuntime::traverse: unknown node type");
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive invocation
//
// Leading dimensions are derived from the TEIR axis byte strides using the
// unit-stride detection logic in derive_gemm / derive_zero / derive_copy above.
// This is the key fix vs. the original code which hardcoded lda=K, ldb=N, ldc=N.
// ─────────────────────────────────────────────────────────────────────────────
void TeirRuntime::invoke_primitive(const Invocation* inv,
                                    ExecContext&      ctx,
                                    const TeirObject& obj) const {
    const Primitive* prim = inv->primitive;

    auto get_ptr = [&](const std::string& tname) -> float* {
        auto it = ctx.ptrs.find(tname);
        return (it != ctx.ptrs.end()) ? it->second : nullptr;
    };

    switch (prim->kind) {

    case PrimKind::Zero: {
        // kernel(nullptr, out_ptr, 0, ldb)
        auto zp = derive_zero(*prim, obj);
        float* out = get_ptr(zp.t_out);
        auto k = const_cast<TeirRuntime*>(this)->get_zero_kernel(zp.M, zp.N);
        k(nullptr, out, 0, zp.ldb);
        break;
    }

    case PrimKind::Copy: {
        // kernel(src, dst, lda, ldb)
        auto cp = derive_copy(*prim, obj);
        float* src = get_ptr(cp.t_src);
        float* dst = get_ptr(cp.t_dst);
        auto k = const_cast<TeirRuntime*>(this)->get_copy_kernel(cp.M, cp.N);
        k(src, dst, cp.lda, cp.ldb);
        break;
    }

    case PrimKind::Contraction: {
        // kernel(A, B, C, lda, ldb, ldc)
        auto gp = derive_gemm(*prim, obj);
        float* a = get_ptr(gp.t_a);
        float* b = get_ptr(gp.t_b);
        float* c = get_ptr(gp.t_c);
        auto k = const_cast<TeirRuntime*>(this)->get_gemm_kernel(
                     gp.M, gp.N, gp.K, gp.trans_a, gp.trans_b, gp.trans_c);
        k(a, b, c, gp.lda, gp.ldb, gp.ldc);
        break;
    }

    }
}

} // namespace mini_jit::teir