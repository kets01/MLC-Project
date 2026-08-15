# Apple BNNS as a vendor baseline — investigation and negative result

**Status: closed, not measured.** This is a developer note, deliberately kept
out of the project report. It records a baseline we investigated and could not
obtain a number for, what we established along the way, and why we stopped.
It follows the same convention as `sprint3_errors.md`, `sprint4_errors.md` and
`sprint6_errors.md`: a result that did not work is still a result, provided the
evidence is written down.

Everything below was verified on the course machine — **Apple M4, macOS 15.2
(build 24C101), Command Line Tools SDK, no Xcode** — not recalled.

---

## 1. Why we looked at BNNS at all

Sprint 7.5 compared our kernels against PyTorch and ExecuTorch and found we were
ahead of both. That is a weaker claim than "state of the art", because neither
is a specialist kernel: PyTorch's CPU RMSNorm decomposes into elementwise ops
rather than running a fused kernel. The obvious missing baseline was Apple's own
library, which is the vendor implementation for this hardware.

An earlier attempt at a vendor comparison (Sprint 6 §2.7–2.8) had gone badly
wrong: it presented our own `vDSP_svesq` + `vDSP_vsmul` composition as
"Accelerate RMSNorm", and called `vDSP_normalize` "LayerNorm" when it has no
eps, no γ and no β. This investigation was scoped from the start to avoid
repeating that.

## 2. What we established (these findings stand)

### 2.1 BNNS has LayerNorm — via a *generic* constructor

```c
BNNSFilter BNNSFilterCreateLayerNormalization(BNNSFilterType normType,
                                              const BNNSLayerParametersNormalization *,
                                              const BNNSFilterParameters * _Nullable)
__API_DEPRECATED("Use BNNSGraph* APIs", macos(11.0, 15.0), ...);
```

Its own documentation says it *"currently support[s] batch, instance, layer and
group norm"*, selected by `normType`:

```c
BNNSBatchNorm    = 2
BNNSInstanceNorm = 3
BNNSLayerNorm    = 4     // <-- the only correct choice for this project
BNNSGroupNorm    = 5
```

**This is a trap worth naming.** Passing the wrong enumerator would silently
benchmark a different algorithm and produce a plausible-looking, wrong
comparison. Any future attempt must (a) pass `BNNSLayerNorm` explicitly,
(b) `static_assert` its value so an SDK renumbering breaks the build, (c) gate
the output against the float64 reference before timing, and (d) run a negative
control with `BNNSInstanceNorm` and require it *not* to match — because (a)–(c)
would all still pass if BNNS ignored `normType` entirely.

### 2.2 BNNS has **no RMSNorm** on macOS 15.2

Exhaustive grep of the BNNS header set
(`bnns.h`, `bnns_constants.h`, `bnns_structures.h`, `bnns_graph.h`): every
occurrence of "RMS" is `RMSProp` — the optimizer — or unrelated prose. There is
no RMS *normalization* filter, and `BNNSFilterType` has no RMS enumerator.

### 2.3 Apple's current BNNSGraph *does* have both — but not in this SDK

Current Accelerate exposes a BNNSGraph builder with named ops:
`layerNorm(axes:epsilon:)`, `layerNorm(weight:bias:axes:epsilon:)` and
`rmsNorm(scale:epsilon:)`. Those are a **later Accelerate release than this OS
ships**. Measured on this machine's Swift overlay:

```
$SDK/usr/lib/swift/Accelerate.swiftmodule/arm64e-apple-macos.swiftinterface
  16 987 lines, 644 BNNS declarations
  BNNSGraph   : 117 occurrences     (compile / execute only)
  layerNorm   :   0 occurrences
  rmsNorm     :   0 occurrences

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
public enum BNNSGraph { ... }
```

This is not a toolchain gap that installing Xcode fixes: a newer SDK would let
those calls *compile*, but they are availability-gated and would not *run* on
macOS 15.2.

**So the correct statement for the report is narrow and defensible:** Apple
provides native LayerNorm *and* RMSNorm operations in current Accelerate; on the
course's macOS 15.2 environment, only LayerNorm is reachable, and there is no
direct RMSNorm API.

### 2.4 Descriptor findings (undocumented, determined by probing)

* γ/β must be described as `BNNSDataLayoutImageCHW` with size `{N, 1, 1}`
  (per-feature, broadcast across rows) or `{N, 1, M}` (per-element).
* `BNNSDataLayoutVector{N}`, `{M}`, `{1}` and CHW `{1,1,M}` are **rejected at
  creation**, returning `NULL` with no diagnostic.
* That the accepted parameter shapes are indexed by the *normalized* axis is
  corroborating evidence that the layer-norm branch is selected — a
  batch/instance norm would want per-channel parameters of length `M`.

## 3. Where it stopped: created but not applicable

`BNNSFilterCreateLayerNormalization` returns a non-NULL filter, and
`BNNSFilterApply` then returns **-1** with no further diagnostic.

The decisive experiment was to shrink to the smallest possible case — a single
logical sample, `M = 1`, `N = 8`, γ = 1, β = 0, no strides, no batching — so
that descriptor problems could not be confused with batching semantics:

| variable | values tried | result |
|---|---|---|
| `normalization_axis` | 0, 1, 2 | all −1 |
| `M` | **1**, 2, 128, 1024, 4096 | all −1 |
| γ/β | absent; CHW {N,1,1}; {N,1,M}; Vector{N}, {M}, {1} | all −1 (last three also fail creation) |
| data location | in descriptor vs supplied at `Apply` | all −1 |
| apply path | `BNNSFilterApply`, `BNNSFilterApplyBatch` | all −1 |
| `BNNSFilterParameters` | `NULL`, zero-initialised | all −1 |
| layout | `ImageCHW`, `Vector` | all −1 |

**The `M = 1` row is the informative one.** With a single sample there is no
ambiguity about whether `M` lies inside the normalization domain, and the axis
argument has nothing to disambiguate — yet it fails identically at every axis.
So **BNNS's interpretation of the tensor dimensions and the normalization axis
is not the cause.** Further descriptor permutation was judged unlikely to pay.

### 3.1 Control: the deprecated filter path itself works

To separate "our configuration is wrong" from "this path is dead on this OS", a
structurally identical use of a different deprecated filter:

```c
BNNSLayerParametersActivation p{};
p.i_desc = p.o_desc = <BNNSDataLayoutVector{8}, data = NULL>;
p.activation.function = BNNSActivationFunctionRectifiedLinear;
BNNSFilter f = BNNSFilterCreateLayerActivation(&p, nullptr);
BNNSFilterApply(f, in, out);      // rc = 0, output correct
```

ReLU **creates and applies correctly**, with `data = NULL` descriptors and the
same `Apply` call shape. It carries the **identical** deprecation annotation,
`macos(11.0, 15.0)` — so "deprecated, therefore stubbed out" does not explain
the difference either.

One suggestive but unproven detail: in `bnns.h` the normalization declaration is
bracketed together with `BNNSFilterCreateLayerLoss`, i.e. the training-oriented
filters. We could not turn that observation into a working call from outside the
framework.

**Honest conclusion:** on macOS 15.2 we could create BNNS layer-normalization
filters but not execute them, for a reason the API does not expose, while the
surrounding deprecated filter infrastructure works. We do not claim the API is
broken — only that we could not drive it, and we record exactly what was tried.

## 4. Minimal reproduction

Self-contained; needs only the Command Line Tools.

```cpp
// c++ -std=c++17 -O1 min.cpp -framework Accelerate -o min && ./min
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <vector>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
int main() {
  const size_t N = 8;
  std::vector<float> in{1,2,3,4,5,6,7,8}, out(N, -99.f), g(N, 1.f), b(N, 0.f);
  auto chw = [&](void* d) {
    BNNSNDArrayDescriptor x{};
    x.layout = BNNSDataLayoutImageCHW;
    x.size[0] = N; x.size[1] = 1; x.size[2] = 1;
    x.data = d; x.data_type = BNNSDataTypeFloat32; return x;
  };
  BNNSLayerParametersNormalization p{};
  p.i_desc = chw(nullptr);  p.o_desc = chw(nullptr);
  p.gamma_desc = chw(g.data());  p.beta_desc = chw(b.data());
  p.epsilon = 1e-5f;
  p.activation.function = BNNSActivationFunctionIdentity;
  p.normalization_axis = 0;                       // 1, 2 behave identically
  BNNSFilter f = BNNSFilterCreateLayerNormalization(BNNSLayerNorm, &p, nullptr);
  printf("create = %s\n", f ? "ok" : "null");     // -> ok
  printf("apply rc = %d\n", BNNSFilterApply(f, in.data(), out.data()));  // -> -1
}
```

## 5. Decision

**BNNS is dropped as a baseline; no number is reported.** Reporting a
LayerNorm figure we could not obtain, or composing an RMSNorm out of primitives
and labelling it a vendor kernel, are both worse than reporting nothing — the
second is precisely the Sprint-6 §2.7 error.

Not attempted, and recorded as such rather than as an untried box:

* **BNNSGraph via CoreML** (`BNNSGraphCompileFromFile` → `BNNSGraphContextMake`),
  the non-deprecated path the deprecation message itself names. This is the
  route a future attempt should take. It requires building a CoreML model
  containing the op, so the resulting number would also include CoreML lowering
  decisions — in particular, whether RMSNorm lowers to a fused op or a
  decomposition on this release, which is itself worth knowing.

**What the report should say**, and all it should say: Apple provides native
LayerNorm and RMSNorm operations in current Accelerate/BNNSGraph; on the
course's macOS 15.2 environment there is **no direct RMSNorm API**, and the
reachable LayerNorm entry point is deprecated in favour of BNNSGraph. We
therefore compare against PyTorch and ExecuTorch, and state that a specialist
vendor kernel was not measured.
