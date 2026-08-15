#include "week7/TeirParser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mini_jit::teir {

namespace {

// ---------------------------------------------------------------------
// Tokenizer: the grammar needs only "words" — any run of characters that is
// neither whitespace nor single-char punctuation — plus the punctuation
// below.  No comments, no string literals.
// ---------------------------------------------------------------------

struct Token {
    std::string text;
};

std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> toks;
    size_t i = 0;
    const std::string punct = "{}[]():,";
    while (i < src.size()) {
        char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (punct.find(c) != std::string::npos) {
            toks.push_back({std::string(1, c)});
            ++i;
            continue;
        }
        size_t start = i;
        while (i < src.size() &&
               !std::isspace(static_cast<unsigned char>(src[i])) &&
               punct.find(src[i]) == std::string::npos) {
            ++i;
        }
        toks.push_back({src.substr(start, i - start)});
    }
    return toks;
}

// ---------------------------------------------------------------------
// Cursor over the token stream.  Fails loud, so a malformed .teir file is a
// clear error rather than a wrong tree.
// ---------------------------------------------------------------------

class Cursor {
  public:
    explicit Cursor(std::vector<Token> toks) : m_toks(std::move(toks)) {}

    bool at_end() const { return m_pos >= m_toks.size(); }

    const std::string& peek() const {
        if (at_end()) throw std::runtime_error("teir: unexpected end of file");
        return m_toks[m_pos].text;
    }

    std::string next() {
        if (at_end()) throw std::runtime_error("teir: unexpected end of file");
        return m_toks[m_pos++].text;
    }

    void expect(const std::string& literal) {
        std::string got = next();
        if (got != literal)
            throw std::runtime_error("teir: expected '" + literal + "', got '" + got + "'");
    }

    bool check(const std::string& literal) const {
        return !at_end() && m_toks[m_pos].text == literal;
    }

  private:
    std::vector<Token> m_toks;
    size_t m_pos = 0;
};

// ---------------------------------------------------------------------
// Pre-resolution representation: mirrors the grammar directly, with @name
// references left as strings until build_tree() resolves them.
// ---------------------------------------------------------------------

struct ParsedAxis {
    std::string name;
    uint32_t extent = 0;
    std::unordered_map<std::string, uint64_t> strides; // tensor name -> byte stride
};

struct ParsedPrimitive {
    std::string name;
    std::string type; // "Copy" | "Zero" | "Contraction" | "RMSNorm" | "LayerNorm"
    std::unordered_map<std::string, std::vector<std::string>> axis_groups; // "M"/"N"/"K" -> axis names
    std::unordered_map<std::string, std::string> metadata;
};

struct ParsedIter {
    std::string name;
    std::string axis_name;
    bool is_parallel = false;
    std::vector<std::string> children;
};

struct ParsedInvoke {
    std::string name;
    std::string primitive_name;
    bool has_guard = false;
    std::string guard_axis_name;
};

struct ParsedProgram {
    std::vector<std::string> tensor_order;               // declaration order -> pointer slot
    std::unordered_map<std::string, ParsedAxis> axes;
    std::unordered_map<std::string, ParsedPrimitive> primitives;
    std::unordered_map<std::string, ParsedIter> iters;
    std::unordered_map<std::string, ParsedInvoke> invokes;
    std::vector<std::string> roots;
};

// ---------------------------------------------------------------------
// Recursive-descent parsing of one `teir @name { ... }` block into a
// ParsedProgram. One function per grammar production.
// ---------------------------------------------------------------------

void parse_tensor_decl(Cursor& c, ParsedProgram& prog) {
    c.expect("tensor");
    std::string name = c.next(); // "%name"
    if (!name.empty() && name[0] == '%') name = name.substr(1);
    c.expect(":");
    c.next(); // dtype, e.g. "f32" — informational only, every kernel here is fp32
    prog.tensor_order.push_back(name);
}

void parse_axis_decl(Cursor& c, ParsedProgram& prog) {
    c.expect("axis");
    ParsedAxis axis;
    std::string aname = c.next(); // "@name"
    axis.name = (!aname.empty() && aname[0] == '@') ? aname.substr(1) : aname;
    c.expect("extent");
    axis.extent = static_cast<uint32_t>(std::stoul(c.next()));
    c.expect("strides");
    c.expect("{");
    while (!c.check("}")) {
        std::string tensor = c.next();
        c.expect(":");
        uint64_t stride = std::stoull(c.next());
        axis.strides[tensor] = stride;
        if (c.check(",")) c.next(); // optional trailing/inter-entry comma
    }
    c.expect("}");
    prog.axes[axis.name] = std::move(axis);
}

std::vector<std::string> parse_name_list(Cursor& c) {
    std::vector<std::string> names;
    c.expect("[");
    while (!c.check("]")) {
        std::string n = c.next();
        if (!n.empty() && n[0] == '@') n = n.substr(1);
        names.push_back(n);
        if (c.check(",")) c.next();
    }
    c.expect("]");
    return names;
}

void parse_primitive_decl(Cursor& c, ParsedProgram& prog) {
    c.expect("primitive");
    ParsedPrimitive prim;
    std::string pname = c.next();
    prim.name = (!pname.empty() && pname[0] == '@') ? pname.substr(1) : pname;
    c.expect(":");
    prim.type = c.next();
    c.expect("axes");
    c.expect("{");
    while (!c.check("}")) {
        std::string group = c.next(); // "M" | "N" | "K"
        c.expect(":");
        prim.axis_groups[group] = parse_name_list(c);
        if (c.check(",")) c.next();
    }
    c.expect("}");
    c.expect("metadata");
    c.expect("{");
    while (!c.check("}")) {
        std::string key = c.next();
        c.expect(":");
        std::string value = c.next();
        prim.metadata[key] = value;
        if (c.check(",")) c.next();
    }
    c.expect("}");
    prog.primitives[prim.name] = std::move(prim);
}

void parse_schedule(Cursor& c, ParsedProgram& prog) {
    c.expect("schedule");
    c.expect("{");
    c.expect("roots");
    prog.roots = parse_name_list(c);

    while (!c.check("}")) {
        if (c.check("iter")) {
            c.next();
            ParsedIter it;
            std::string iname = c.next();
            it.name = (!iname.empty() && iname[0] == '@') ? iname.substr(1) : iname;
            c.expect("axis");
            std::string axname = c.next();
            it.axis_name = (!axname.empty() && axname[0] == '@') ? axname.substr(1) : axname;
            c.expect("policy");
            std::string policy = c.next();
            it.is_parallel = (policy == "parallel");
            c.expect("children");
            it.children = parse_name_list(c);
            prog.iters[it.name] = std::move(it);
        } else if (c.check("invoke")) {
            c.next();
            ParsedInvoke inv;
            std::string iname = c.next();
            inv.name = (!iname.empty() && iname[0] == '@') ? iname.substr(1) : iname;
            c.expect("primitive");
            std::string pname = c.next();
            inv.primitive_name = (!pname.empty() && pname[0] == '@') ? pname.substr(1) : pname;
            if (c.check("guard")) {
                c.next();
                c.expect("first");
                c.expect("(");
                std::string axname = c.next();
                inv.guard_axis_name = (!axname.empty() && axname[0] == '@') ? axname.substr(1) : axname;
                c.expect(")");
                inv.has_guard = true;
            }
            prog.invokes[inv.name] = std::move(inv);
        } else {
            throw std::runtime_error("teir: expected 'iter' or 'invoke', got '" + c.peek() + "'");
        }
    }
    c.expect("}");
}

ParsedProgram parse_program(Cursor& c) {
    ParsedProgram prog;
    c.expect("teir");
    c.next(); // "@name"
    c.expect("{");
    while (!c.check("}")) {
        if (c.check("tensor")) parse_tensor_decl(c, prog);
        else if (c.check("axis")) parse_axis_decl(c, prog);
        else if (c.check("primitive")) parse_primitive_decl(c, prog);
        else if (c.check("schedule")) parse_schedule(c, prog);
        else throw std::runtime_error("teir: unexpected top-level token '" + c.peek() + "'");
    }
    c.expect("}");
    return prog;
}

// ---------------------------------------------------------------------
// Resolution: names -> the runtime's actual Axis/Node objects.
// ---------------------------------------------------------------------

// tensor name -> pointer slot (0=a,1=b,2=c,3=d), by declaration order, which
// is the convention every hand-written build_*_tree() already follows.
std::unordered_map<std::string, int> slot_map(const ParsedProgram& prog) {
    std::unordered_map<std::string, int> slots;
    for (size_t i = 0; i < prog.tensor_order.size(); ++i)
        slots[prog.tensor_order[i]] = static_cast<int>(i);
    return slots;
}

// Element stride (the file's strides are in bytes, fp32 -> /4) for one
// tensor along one parsed axis; 0 if that tensor doesn't move on this axis.
uint64_t elem_stride(const ParsedAxis& ax, const std::string& tensor) {
    auto it = ax.strides.find(tensor);
    return (it == ax.strides.end()) ? 0 : (it->second / 4);
}

// Product of extents of every axis in a primitive's named group ("M"/"N"/"K").
// Every current file lists exactly one axis per group; the product
// generalizes cleanly if a future file ever lists more than one.
uint32_t group_extent(const ParsedPrimitive& prim, const ParsedProgram& prog,
                       const std::string& group) {
    auto it = prim.axis_groups.find(group);
    if (it == prim.axis_groups.end()) return 0;
    uint32_t total = 1;
    for (const auto& axname : it->second) {
        auto ax_it = prog.axes.find(axname);
        if (ax_it == prog.axes.end())
            throw std::runtime_error("teir: primitive references unknown axis @" + axname);
        total *= ax_it->second.extent;
    }
    return total;
}

// The single axis of a primitive's group (all current files use exactly
// one axis per M/N/K group) — needed to look up that axis's stride table
// when computing a kernel's leading dimension.
const ParsedAxis& group_axis(const ParsedPrimitive& prim, const ParsedProgram& prog,
                              const std::string& group) {
    auto it = prim.axis_groups.find(group);
    if (it == prim.axis_groups.end() || it->second.empty())
        throw std::runtime_error("teir: primitive '" + prim.name + "' has no '" + group + "' axis");
    return prog.axes.at(it->second.front());
}

// Fill an Invocation's kernel_name and shape/layout from its primitive's
// declared type.  Layout and leading dimensions are DERIVED from the tensors'
// axis strides in the file — the runtime hardcodes no layout assumption.  A
// direction with element stride 1 is the contiguous one and the other's
// stride is the leading dimension; anything contiguous in neither direction
// is rejected loudly, since streaming mode has no gather loads on this target
// and no kernel could serve it.
void fill_invocation_params(Invocation& inv, const ParsedPrimitive& prim,
                            const ParsedProgram& prog) {
    const auto& type = prim.type;

    // Classify one operand tile: contiguous along `minor` (stride 1) with
    // leading dimension = the `major` axis stride -> returns {trans, ld}
    // where trans=1 means the tile's N/K-direction (2nd index) is the
    // contiguous one (row-major in the GEMM convention).
    auto classify = [&](const ParsedAxis& ax_row, const ParsedAxis& ax_col,
                        const std::string& tensor) -> std::pair<uint32_t, int64_t> {
        uint64_t s_row = elem_stride(ax_row, tensor);
        uint64_t s_col = elem_stride(ax_col, tensor);
        if (s_col == 1) return {1u, static_cast<int64_t>(s_row)};   // row-major
        if (s_row == 1) return {0u, static_cast<int64_t>(s_col)};   // col-major
        throw std::runtime_error("teir: tensor '" + tensor + "' of primitive '" +
                                 prim.name + "' is contiguous in neither direction");
    };

    if (type == "Copy") {
        inv.m = group_extent(prim, prog, "M");
        inv.n = group_extent(prim, prog, "N");
        const ParsedAxis& axM = group_axis(prim, prog, "M");
        const ParsedAxis& axN = group_axis(prim, prog, "N");
        auto [ti, ldi] = classify(axM, axN, "in");
        auto [to, ldo] = classify(axM, axN, "out");
        inv.kernel_name = "identity";
        if (ti == to) {
            // Same orientation on both sides: a plain contiguous copy
            // (Unary identity, trans_b=0, which walks m*n contiguous
            // floats — so the tile must be fully dense).
            int64_t contig_extent = ti ? inv.n : inv.m;
            if (ldi != contig_extent || ldo != contig_extent)
                throw std::runtime_error("teir: Copy with padded tiles is not supported");
            inv.trans_b = 0;
        } else {
            // Opposite orientations: the transposing copy — Unary identity
            // with trans_b=1 (A column-major in, B row-major out), exactly
            // the course Unary.h layout flag. The Unary convention fixes
            // A as the column-major side; if the FILE declares the input
            // row-contiguous instead, swap the roles (m<->n) so the
            // in-tensor is read as the column-major operand.
            inv.trans_b = 1;
            if (ti == 1) std::swap(inv.m, inv.n);
            inv.ld_a = ldi;
            inv.ld_b = ldo;
        }
    } else if (type == "Zero") {
        inv.kernel_name = "zero";
        inv.m = group_extent(prim, prog, "M");
        inv.n = group_extent(prim, prog, "N");
        const ParsedAxis& axM = group_axis(prim, prog, "M");
        const ParsedAxis& axN = group_axis(prim, prog, "N");
        // The tile may be padded/interleaved (contraction.teir's q x s tile
        // has other axes between consecutive q's): record the contiguous
        // run direction and its stride; the runtime zeroes run by run.
        // Normalize so m = number of runs, n = contiguous run length,
        // ld_b = distance between runs.
        auto [tz, ldz] = classify(axM, axN, "out");
        if (tz == 0) std::swap(inv.m, inv.n);
        inv.ld_b = ldz;
    } else if (type == "Contraction") {
        inv.kernel_name = "gemm";
        inv.m = group_extent(prim, prog, "M");
        inv.n = group_extent(prim, prog, "N");
        inv.k = group_extent(prim, prog, "K");
        const ParsedAxis& axM = group_axis(prim, prog, "M");
        const ParsedAxis& axN = group_axis(prim, prog, "N");
        const ParsedAxis& axK = group_axis(prim, prog, "K");
        auto [ta, lda] = classify(axM, axK, "in0");   // A is M x K
        auto [tb, ldb] = classify(axK, axN, "in1");   // B is K x N
        auto [tc, ldc] = classify(axM, axN, "out");   // C is M x N
        inv.trans_a = ta; inv.ld_a = lda;
        inv.trans_b = tb; inv.ld_b = ldb;
        inv.trans_c = tc; inv.ld_c = ldc;
    } else if (type == "RMSNorm" || type == "LayerNorm") {
        inv.kernel_name = (type == "RMSNorm") ? "rmsnorm" : "layernorm";
        inv.m = group_extent(prim, prog, "M");
        inv.n = group_extent(prim, prog, "N");
        // The frozen norm kernels are column-major (context.md §7): rows
        // contiguous along M, ld = stride along the feature (N) axis. A
        // file declaring anything else cannot be served — fail loudly.
        if (elem_stride(group_axis(prim, prog, "M"), "in") != 1 ||
            elem_stride(group_axis(prim, prog, "M"), "out") != 1)
            throw std::runtime_error("teir: " + type + " tile must be column-major "
                                     "(M direction contiguous)");
        inv.ld_a = static_cast<int64_t>(elem_stride(group_axis(prim, prog, "N"), "in"));
        inv.ld_b = static_cast<int64_t>(elem_stride(group_axis(prim, prog, "N"), "out"));
        auto eps_it = prim.metadata.find("eps");
        if (eps_it != prim.metadata.end()) inv.eps = std::stof(eps_it->second);
    } else {
        throw std::runtime_error("teir: unknown primitive type '" + type + "'");
    }
}

// Build the runtime Node tree for one schedule node name (an iter or an
// invoke), recursively. `axes_out` accumulates heap-allocated Axis objects
// so callers can keep them alive as long as the returned tree (same
// intentional-leak convention the hand-written build_*_tree() functions
// already use — no owner ever frees an Axis today).
std::shared_ptr<Node> build_node(const std::string& name, const ParsedProgram& prog,
                                 std::unordered_map<std::string, Axis*>& axes_out) {
    auto iter_it = prog.iters.find(name);
    if (iter_it != prog.iters.end()) {
        const ParsedIter& pit = iter_it->second;

        Axis*& ax = axes_out[pit.axis_name];
        if (!ax) {
            const ParsedAxis& pax = prog.axes.at(pit.axis_name);
            auto slots = slot_map(prog);
            ax = new Axis{pax.name, pax.extent, 0, 0, 0, 0};
            for (const auto& [tensor, slot] : slots) {
                uint64_t s = elem_stride(pax, tensor);
                switch (slot) {
                    case 0: ax->stride_a = s; break;
                    case 1: ax->stride_b = s; break;
                    case 2: ax->stride_c = s; break;
                    case 3: ax->stride_d = s; break;
                }
            }
        }

        auto node = std::make_shared<Iteration>();
        node->axis = ax;
        node->is_parallel = pit.is_parallel;

        if (pit.children.size() == 1) {
            node->body = build_node(pit.children.front(), prog, axes_out);
        } else {
            auto seq = std::make_shared<Sequence>();
            for (const auto& child : pit.children)
                seq->children.push_back(build_node(child, prog, axes_out));
            node->body = seq;
        }
        return node;
    }

    auto invoke_it = prog.invokes.find(name);
    if (invoke_it != prog.invokes.end()) {
        const ParsedInvoke& pinv = invoke_it->second;
        auto prim_it = prog.primitives.find(pinv.primitive_name);
        if (prim_it == prog.primitives.end())
            throw std::runtime_error("teir: invoke references unknown primitive @" + pinv.primitive_name);

        auto node = std::make_shared<Invocation>();
        fill_invocation_params(*node, prim_it->second, prog);

        if (pinv.has_guard) {
            Axis*& gax = axes_out[pinv.guard_axis_name];
            if (!gax) {
                const ParsedAxis& pax = prog.axes.at(pinv.guard_axis_name);
                gax = new Axis{pax.name, pax.extent, 0, 0, 0, 0};
            }
            node->guard_axis = gax;
        }
        return node;
    }

    throw std::runtime_error("teir: unresolved schedule reference @" + name);
}

} // namespace

std::shared_ptr<Node> parse_teir_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("teir: cannot open '" + path + "'");
    std::stringstream buf;
    buf << file.rdbuf();

    Cursor cursor(tokenize(buf.str()));
    ParsedProgram prog = parse_program(cursor);

    if (prog.roots.empty())
        throw std::runtime_error("teir: schedule has no roots");

    std::unordered_map<std::string, Axis*> axes_out;
    return build_node(prog.roots.front(), prog, axes_out);
}

} // namespace mini_jit::teir
