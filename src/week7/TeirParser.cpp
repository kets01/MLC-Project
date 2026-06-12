#include "week7/TeirParser.hpp"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <regex>

namespace mini_jit::teir {

// ──────────────────────────────────────────────────────────────────────────────
// Minimal tokeniser / helper
// ──────────────────────────────────────────────────────────────────────────────
namespace {

// Strip // line comments and return clean source
std::string strip_comments(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    size_t i = 0;
    while (i < src.size()) {
        if (i + 1 < src.size() && src[i] == '/' && src[i+1] == '/') {
            while (i < src.size() && src[i] != '\n') ++i;
        } else {
            out += src[i++];
        }
    }
    return out;
}

struct Tokens {
    std::vector<std::string> toks;
    size_t pos = 0;

    bool eof() const { return pos >= toks.size(); }

    const std::string& peek() const {
        static const std::string empty;
        return eof() ? empty : toks[pos];
    }

    std::string consume() {
        if (eof()) throw std::runtime_error("Unexpected end of token stream");
        return toks[pos++];
    }

    std::string expect(const std::string& tok) {
        std::string got = consume();
        if (got != tok)
            throw std::runtime_error("Expected '" + tok + "' but got '" + got + "'");
        return got;
    }

    bool try_consume(const std::string& tok) {
        if (!eof() && toks[pos] == tok) { ++pos; return true; }
        return false;
    }
};

// Very simple tokeniser: splits on whitespace and isolates { } [ ] : , @
Tokens tokenise(const std::string& src) {
    Tokens t;
    std::string cur;
    auto flush = [&]{ if (!cur.empty()) { t.toks.push_back(cur); cur.clear(); } };

    for (char c : src) {
        if (std::isspace((unsigned char)c)) { flush(); }
        else if (c == '{' || c == '}' || c == '[' || c == ']'
              || c == ':' || c == ',') {
            flush();
            t.toks.push_back(std::string(1,c));
        } else {
            cur += c;
        }
    }
    flush();
    return t;
}

// Parse an @identifier and return the name part (without @)
std::string parse_at(Tokens& t) {
    std::string tok = t.consume();
    if (tok.empty() || tok[0] != '@')
        throw std::runtime_error("Expected @identifier, got '" + tok + "'");
    return tok.substr(1);
}

// Parse a %identifier and return the name part (without %)
std::string parse_pct(Tokens& t) {
    std::string tok = t.consume();
    if (tok.empty() || tok[0] != '%')
        throw std::runtime_error("Expected %identifier, got '" + tok + "'");
    return tok.substr(1);
}

// Parse a list enclosed in [ @a, @b, … ] and return the names (without @)
std::vector<std::string> parse_at_list(Tokens& t) {
    t.expect("[");
    std::vector<std::string> result;
    while (!t.eof() && t.peek() != "]") {
        result.push_back(parse_at(t));
        t.try_consume(",");
    }
    t.expect("]");
    return result;
}

DataType parse_dtype(Tokens& t) {
    std::string tok = t.consume();
    if (tok == "f32") return DataType::f32;
    if (tok == "f64") return DataType::f64;
    throw std::runtime_error("Unknown data type: " + tok);
}

PrimKind parse_prim_kind(const std::string& s) {
    if (s == "Zero")        return PrimKind::Zero;
    if (s == "Copy")        return PrimKind::Copy;
    if (s == "Contraction") return PrimKind::Contraction;
    throw std::runtime_error("Unknown primitive kind: " + s);
}

} // anonymous namespace

// ──────────────────────────────────────────────────────────────────────────────
// Main parser
// ──────────────────────────────────────────────────────────────────────────────
TeirObject parse(const std::string& source) {
    Tokens t = tokenise(strip_comments(source));

    TeirObject obj;

    // teir @name {
    t.expect("teir");
    obj.name = parse_at(t);
    t.expect("{");

    // We use temporary storage keyed by name; ownership lives in obj.axes / obj.primitives.
    // Iteration/Invocation nodes reference raw pointers into those maps, so we
    // build axes first, then primitives, then schedule.

    // Collect raw schedule declarations before building the node tree, because
    // forward references are common.
    struct IterDecl {
        std::string name;
        std::string axis_name;
        Policy      policy;
        std::vector<std::string> child_names;
    };
    struct InvDecl {
        std::string name;
        std::string prim_name;
        GuardKind   guard      = GuardKind::none;
        std::string guard_axis;
    };
    struct RootsDecl {
        std::vector<std::string> names;
    };

    std::vector<IterDecl> iter_decls;
    std::vector<InvDecl>  inv_decls;
    std::vector<std::string> root_names;
    //bool in_schedule = false;

    while (!t.eof() && t.peek() != "}") {
        std::string kw = t.consume();

        // ── tensor ───────────────────────────────────────────────────────
        if (kw == "tensor") {
            std::string tname = parse_pct(t);
            t.expect(":");
            /*DataType dt =*/ parse_dtype(t);  // currently ignored in struct
            obj.tensor_names.push_back(tname);
        }
        // ── axis ─────────────────────────────────────────────────────────
        else if (kw == "axis") {
            Axis ax;
            ax.name = parse_at(t);
            t.expect("extent");
            ax.extent = static_cast<uint32_t>(std::stoul(t.consume()));
            // optional strides block
            if (t.peek() == "strides") {
                t.consume(); // "strides"
                t.expect("{");
                while (!t.eof() && t.peek() != "}") {
                    // tensor_name: stride_value [,]
                    std::string tname = t.consume();
                    t.expect(":");
                    int64_t stride = std::stoll(t.consume());
                    ax.strides[tname] = stride;
                    t.try_consume(",");
                }
                t.expect("}");
            }
            obj.axes[ax.name] = std::move(ax);
        }
        // ── primitive ────────────────────────────────────────────────────
        else if (kw == "primitive") {
            Primitive prim;
            prim.name = parse_at(t);
            t.expect(":");
            prim.kind = parse_prim_kind(t.consume());
            // axes { M: [@a,@b] N: [@c] K: [@d] }
            t.expect("axes");
            t.expect("{");
            while (!t.eof() && t.peek() != "}") {
                std::string role = t.consume(); // M, N, or K
                t.expect(":");
                auto names = parse_at_list(t);
                t.try_consume(",");

                auto to_ptrs = [&](const std::vector<std::string>& ns)
                               -> std::vector<Axis*> {
                    std::vector<Axis*> ptrs;
                    for (auto& n : ns) {
                        auto it = obj.axes.find(n);
                        if (it == obj.axes.end())
                            throw std::runtime_error("Unknown axis @" + n);
                        ptrs.push_back(&it->second);
                    }
                    return ptrs;
                };

                if      (role == "M") prim.axes.M = to_ptrs(names);
                else if (role == "N") prim.axes.N = to_ptrs(names);
                else if (role == "K") prim.axes.K = to_ptrs(names);
                else throw std::runtime_error("Unknown axis role: " + role);
            }
            t.expect("}");
            // metadata { data_type: f32 }
            if (t.peek() == "metadata") {
                t.consume();
                t.expect("{");
                while (!t.eof() && t.peek() != "}") {
                    std::string key = t.consume();
                    t.expect(":");
                    std::string val = t.consume();
                    t.try_consume(",");
                    if (key == "data_type") {
                        if (val == "f32") prim.data_type = DataType::f32;
                        else if (val == "f64") prim.data_type = DataType::f64;
                    }
                }
                t.expect("}");
            }
            obj.primitives[prim.name] = std::move(prim);
        }
        // ── schedule ─────────────────────────────────────────────────────
        else if (kw == "schedule") {
            t.expect("{");
            //in_schedule = true;
            while (!t.eof() && t.peek() != "}") {
                std::string skw = t.consume();

                if (skw == "roots") {
                    root_names = parse_at_list(t);
                }
                else if (skw == "iter") {
                    IterDecl d;
                    d.name = parse_at(t);
                    t.expect("axis");
                    d.axis_name = parse_at(t);
                    t.expect("policy");
                    std::string pol = t.consume();
                    if      (pol == "sequential") d.policy = Policy::sequential;
                    else if (pol == "parallel")   d.policy = Policy::parallel;
                    else throw std::runtime_error("Unknown policy: " + pol);
                    t.expect("children");
                    d.child_names = parse_at_list(t);
                    iter_decls.push_back(std::move(d));
                }
                else if (skw == "invoke") {
                    InvDecl d;
                    d.name = parse_at(t);
                    t.expect("primitive");
                    d.prim_name = parse_at(t);
                    // optional: guard first(@axis)
                    if (t.peek() == "guard") {
                        t.consume();
                        std::string gkind = t.consume(); // e.g. "first(@t)"
                        // The grammar is: guard first(@axis_name)
                        // tokenised as: "first(@t)" or "first" "(" "@t" ")"
                        // Our tokeniser keeps parens as non-delimiters, so
                        // "first(@t)" is likely one token.  Handle both cases.
                        if (gkind.rfind("first(", 0) == 0) {
                            // one token like "first(@t)"
                            d.guard = GuardKind::first;
                            // extract axis name: strip "first(@" and ")"
                            auto inner = gkind.substr(7); // after "first(@"
                            if (!inner.empty() && inner.back() == ')')
                                inner.pop_back();
                            d.guard_axis = inner;
                        } else if (gkind == "first") {
                            // two-token form: first ( @axis )
                            // consume "(" "@axis" ")"
                            t.try_consume("(");
                            d.guard_axis = parse_at(t);
                            t.try_consume(")");
                            d.guard = GuardKind::first;
                        } else {
                            throw std::runtime_error("Unknown guard: " + gkind);
                        }
                    }
                    inv_decls.push_back(std::move(d));
                }
                else {
                    throw std::runtime_error("Unknown schedule keyword: " + skw);
                }
            }
            t.expect("}"); // close schedule
        }
        else {
            throw std::runtime_error("Unknown keyword: " + kw);
        }
    }
    t.expect("}"); // close teir block

    // ── Build node tree from flat declarations ────────────────────────────
    // Map name → shared_ptr<Node>
    std::map<std::string, std::shared_ptr<Node>> node_map;

    // Create invocation nodes
    for (auto& d : inv_decls) {
        auto inv = std::make_shared<Invocation>();
        inv->name = d.name;
        auto pit = obj.primitives.find(d.prim_name);
        if (pit == obj.primitives.end())
            throw std::runtime_error("Unknown primitive @" + d.prim_name);
        inv->primitive   = &pit->second;
        inv->guard       = d.guard;
        if (!d.guard_axis.empty()) {
            auto ait = obj.axes.find(d.guard_axis);
            if (ait == obj.axes.end())
                throw std::runtime_error("Unknown guard axis @" + d.guard_axis);
            inv->guard_axis = &ait->second;
        }
        node_map[d.name] = inv;
    }

    // Create iteration nodes (children resolved via node_map)
    // Iterations may reference each other, so do two passes
    for (auto& d : iter_decls) {
        auto iter = std::make_shared<Iteration>();
        iter->name   = d.name;
        auto ait = obj.axes.find(d.axis_name);
        if (ait == obj.axes.end())
            throw std::runtime_error("Unknown axis @" + d.axis_name);
        iter->axis   = &ait->second;
        iter->policy = d.policy;
        node_map[d.name] = iter; // register before resolving children
    }
    // Second pass: resolve children
    for (auto& d : iter_decls) {
        auto iter = std::dynamic_pointer_cast<Iteration>(node_map.at(d.name));
        if (d.child_names.size() == 1) {
            auto cit = node_map.find(d.child_names[0]);
            if (cit == node_map.end())
                throw std::runtime_error("Unknown child node @" + d.child_names[0]);
            iter->body = cit->second;
        } else {
            // Multiple children → wrap in a Sequence
            auto seq = std::make_shared<Sequence>();
            for (auto& cn : d.child_names) {
                auto cit = node_map.find(cn);
                if (cit == node_map.end())
                    throw std::runtime_error("Unknown child node @" + cn);
                seq->children.push_back(cit->second);
            }
            iter->body = seq;
        }
    }

    // Resolve roots
    for (auto& rn : root_names) {
        auto rit = node_map.find(rn);
        if (rit == node_map.end())
            throw std::runtime_error("Unknown root @" + rn);
        obj.roots.push_back(rit->second);
    }

    return obj;
}

TeirObject parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open TEIR file: " + path);
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return parse(src);
}

} // namespace mini_jit::teir