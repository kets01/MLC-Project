# context.md — MLC-Norm Project Context

> Full context for any AI coding assistant working on MLC-Norm. Read this first.
> - **This file** = what the project is, the target architecture, the kernels, the design decisions.
> - `CLAUDE.md` = how to work (git, commits, tests, correctness gates, ISA rules, build, skills).
> - `ROADMAP.md` = the route: sprint-by-sprint, a correct scalar slice first, then SSVE, then SME, then integration.
>
> Important: the architecture below is the **TARGET vision**, reached by adding layers.
> The project starts as a single correct reference + a thin SSVE kernel (see ROADMAP Sprints 0–2),
> not the full JIT + SME + TEIR system.
>
> Naming: the repository is **MLC-project**; "MLC-Norm" is the short handle used in these docs.

---

## 1. What this project is

**MLC-Norm is the final project for the Machine Learning Compilers Lab (Friedrich-Schiller-Universität
Jena): optimizing the two dominant Transformer normalization primitives — LayerNorm and RMSNorm — for
AArch64 with the Scalable Matrix Extension (SME) and its Streaming SVE (SSVE) mode.**

Normalization is not an afterthought in a Transformer: a norm runs in **every block**, twice per layer.
These primitives are **memory-bandwidth bound** — they read and write the whole activation tensor while
doing very little arithmetic per element — so they sit squarely on the bandwidth ceiling of the machine.
Making them fast is a data-movement problem, not a flop-counting one.

The project delivers, for both norms:

- **Hand-written SME/SSVE kernels** that compute the norm over the feature dimension on a target machine
  exposing SME (AArch64; in practice an Apple M-series chip such as the **M4**, which ships SME1).
- **JIT (dynamic instruction emission)**: the kernels are *generated at runtime* by an extended `mini_jit`
  (the code generator built across the lab), parameterized by shape — not pre-compiled fixed binaries.
- **TEIR integration**: the norms are registered as **primitives** the Tiled Execution IR runtime can place
  inside a loop nest, so a real model graph (e.g. LLaMA inference) can call them through the compiler.
- **An evaluation**: numerical correctness against a C++ reference, and effective-bandwidth (**GiB/s**)
  benchmarks with an **ablation** of the optimizations.

It is **not** a research contribution to normalization, and **not** a from-scratch compiler. It is a
focused harness-and-kernel engineering exercise: take two well-understood math primitives and make them
run well on real SME hardware, the right way, end to end through the lab's compiler stack.

---

## 2. Why this project exists (context for decisions)

The capstone of a semester-long lab that built a code-generation stack bottom-up: AArch64 assembly →
NEON → SSVE/SME microkernels → JIT code generation (`mini_jit`) → TEIR runtime → applications. The norm
project is where those layers are *used together* on a primitive that actually matters in inference.

Decisions optimise for:

- **Learning the stack, not just the result.** Understanding *why* a kernel hits (or misses) peak
  bandwidth matters more than a one-off fast number. This is a vehicle for understanding SME, the roofline,
  and JIT codegen.
- **Correctness as a precondition, not a hope.** A fast kernel that is numerically wrong is worthless;
  every performance claim must stand on a verified-correct kernel (decision B).
- **Honest engineering.** Surface and explain the real problems — the bandwidth ceiling, single-pass
  numerical stability, streaming-mode transition cost — rather than quoting only the best run.
- **Composability.** The kernel is a *leaf the compiler schedules*, not a standalone program. It must
  obey the shared primitive contract so TEIR can call it (decision A, F).
- **A clean git history** that tells the story: reference → vectorized → tiled → JIT → integrated.

When choosing between "impressive but fragile" and "simpler but solid and explained", prefer the latter.

---

## 3. Architecture — MVP first, full system later

### 3a. MVP architecture (what we build first — ROADMAP Sprints 0–2)

A **correct scalar reference plus a single hand-written kernel**, exercised directly through a test/bench
driver. No JIT, no TEIR yet — just prove the math and the data layout end to end.

```
test driver / bench
        │
        ├── reference C++  (scalar LayerNorm / RMSNorm)   ← ground truth
        └── hand-written SSVE kernel (one tile / one row block)
                 │  reduces over the feature axis, then normalizes
                 ▼
        numerical check (vs reference) + GiB/s measurement
```

### 3b. Target architecture (the vision — ROADMAP Sprint 4+)

The norm lives as a **JIT-generated primitive inside the TEIR compiler flow**, the way a real model invokes it:

```
   Transformer model (e.g. LLaMA inference)
                 │
                 ▼
   TEIR — tensor expression IR
   (axes · iteration/invocation nodes · schedule)
                 │   places the norm as a primitive in the loop nest
                 ▼
   LayerNorm / RMSNorm primitive  ◄── canonical signature (decision A)
                 │
                 ▼
   JIT code generation  (mini_jit, extended with Norm)
   dynamic instruction emission · parameterized by shape (decision F)
                 │
                 ▼
   SSVE / SME instructions
   Vector-Length-Agnostic code · ZA-tile usage (decision D)
                 │
                 ▼
   verified correct (vs C++) · measured in GiB/s · ablation (decisions B, E)
```

- **TEIR layer** (`src/teir/`): register LayerNorm/RMSNorm as primitives; the runtime generates the loop
  nest from iteration nodes and invokes the generated kernel for each tile. (Course TEIR conventions.)
- **JIT generator** (`src/mini_jit/Norm.*`): the new `mini_jit::Norm` generator alongside the existing
  `Unary` and `Gemm` generators; emits instruction words at runtime and returns a function pointer.
- **Kernels** (SSVE/SME): the reduction over the feature dimension (SSVE vector work) and the tiled
  load/store/move over a 16×16 block (SME ZA tile); see §5.
- **Reference** (`src/reference/`): plain C++ implementations that are the correctness oracle — never
  optimized, kept obviously-correct on purpose (decision B).

**Shared contract** (`src/mini_jit/Norm.h`): one canonical kernel signature (pointers, leading dimensions,
`trans_b`-style layout flag, ε, the γ/β pointers) used identically by the reference, the JIT generator, and
the TEIR registration (decision A).

---

## 4. Key design decisions (the "why" behind the build)

These are the cross-cutting decisions; `ROADMAP.md` references them as A–F.

- **(A) One canonical primitive interface.** The kernel signature for the norm primitives is defined once —
  following the lab's unary-kernel convention (`a`, `b`, `ld_a`, `ld_b`, layout flag, plus ε and the γ/β
  parameter pointers) — and is the single source of truth. The C++ reference, the `mini_jit` generator, and
  the TEIR registration all use **the same** signature; no layer invents its own. *Pinned down only after
  the reference and first kernel exist*, so the contract reflects what the kernels actually need.

- **(B) Correctness gates performance.** Every kernel is numerically verified against the scalar C++
  reference (within an FP32 tolerance) **before any benchmark number is trusted**. A fast wrong kernel is
  worthless. Numerical stability of the reduction (variance for LayerNorm, sum-of-squares for RMSNorm) is
  *part of* correctness, not a separate concern (see §8).

- **(C) Two norms, two compute strategies — not one kernel with a flag.** LayerNorm is **two-pass**
  (pass 1: mean & variance; pass 2: normalize, scale γ, shift β) → higher arithmetic intensity, high
  stability. RMSNorm is **single-pass** (one sum-of-squares reduction; no mean, no β) → lower arithmetic
  intensity, ~10–40% higher throughput at the same accuracy. The two reduction structures are a deliberate
  design axis and an **ablation variable**, not a parameter toggle.

- **(D) Vector-Length-Agnostic (VLA) code.** No hard-coded streaming vector length (SVL) or element count.
  The Streaming Vector Length is **queried at runtime** and loops are written to any SVL; predication
  handles the tail. ZA-tile usage is parameterized on the tile geometry, not a magic `16`. The kernel must
  run unchanged across SME implementations with different vector widths.

- **(E) Memory-bandwidth bound → optimise movement, report GiB/s.** Normalization touches every element with
  ~O(1) arithmetic and at least one read + one write, so it is **bandwidth bound**. The headline metric is
  **effective bandwidth (GiB/s)** against the platform's measured peak — *not* GFLOPS. Optimizations target
  **data movement** (keeping the activation row resident between the two LayerNorm passes, tiling for the ZA
  register, avoiding redundant loads), not flop count. Measure against the roofline; know the ceiling before
  optimizing.

- **(F) JIT codegen with a clean primitive boundary.** Kernels are **emitted at runtime** by the extended
  `mini_jit` (dynamic instruction emission), parameterized by shape; emission is a one-time cost and the
  returned function pointer is reused across all invocations. The generator blocks the **tiled (row)
  dimension** to the FP32 tile granularity and leaves the **normalized axis arbitrary** (predicated tails).
  That granularity is `SVL/32` FP32 elements — **16 on the lab's 512-bit-SVL target (Apple M4), not a
  hard-coded constant** — so it stays consistent with VLA (decision D): a properly VLA generator derives the
  multiple from the *queried* SVL, and on a machine with a different SVL the tile width (and the required
  multiple) changes with it. Where the lab's exercises say "multiples of 16", that is a consequence of the
  target hardware, not a magic number baked into the code. The kernel is a **leaf the compiler schedules** —
  it must respect leading dimensions and operate tile-at-a-time so TEIR can compose it into a loop nest,
  never assuming it owns the whole tensor.

---

## 5. Instruction-set strategy (right instructions per task — a performance pillar)

SME exposes two complementary tools; matching each part of the kernel to the right one is how the bandwidth
ceiling gets approached:

- **SSVE (Streaming SVE) — the reduction and elementwise work.** The per-row reduction over the feature
  dimension (sum for the mean, sum-of-squares for variance/RMS) and the final elementwise
  normalize-scale-shift are **vector** operations along one axis. SSVE's predicated, vector-length-agnostic
  instructions (`FMLA`/`FADDV`-style reductions, predicated loads/stores for the tail) are the right fit.
  These run for every element, so their efficiency dominates the bandwidth number.

- **SME ZA tile — the tiled movement and reuse.** When the work is naturally a 2D block (a 16×16 FP32 tile,
  per the lab's microkernel convention) — loading a tile, transposing/moving it, or reusing a row of scale
  factors across a column block — the **ZA accumulator tile** and its tile load/store instructions move data
  in the shape the hardware wants. The rule of thumb: reach for the ZA tile only when the data genuinely
  moves as a 2D block (a tile load/store or transpose). The per-row statistic, by contrast, is a *horizontal*
  reduction — many elements along the feature axis collapsed into one number per row — and the ZA tile is a
  **matrix accumulator** built for outer-product / matmul-style accumulation, which a horizontal reduction
  maps onto poorly. So compute the reduction with an SSVE vector reduction (the `FADDV`-style instruction
  above), and reserve ZA for the 2D movement; don't contort the reduction to run through ZA just because the
  extension is available.

**Tile sizes and the streaming vector length are NEVER hard-coded.** SVL is queried at runtime (decision D);
tile geometry is a generator parameter. Entering and leaving **streaming mode** (`SMSTART`/`SMSTOP`) has a
real cost — keep the whole norm inside one streaming region, don't toggle per element (see §8).

> The exact instruction mnemonics and ZA addressing are an SME-version concern — confirm against the target
> machine's SME level (SME1 on Apple M4) at coding time; don't assume SME2-only instructions are available.

---

## 6. Performance model & the roofline (decision E detail)

- **Know the ceiling first.** Normalization's arithmetic intensity is low (a handful of FLOPs per element
  against 4–8 bytes moved), so its roofline position is on the **bandwidth-bound** side. Peak achievable is
  ~`bytes_moved / peak_bandwidth`, not anything FLOP-derived. Establish the machine's measured peak bandwidth
  (e.g. a STREAM-style probe) as the target line before optimizing.

- **Count the bytes honestly.** LayerNorm two-pass naively reads the row twice; an optimized kernel keeps the
  row resident in registers/cache between passes so it is effectively read once. RMSNorm is single-pass by
  construction. Report effective bandwidth as *useful bytes (1 read + 1 write) / time*, and state whether the
  intermediate reads were eliminated.

- **Streaming-mode overhead is part of the latency.** For small tensors the fixed cost of entering streaming
  mode can dominate; for large tensors it amortizes. Measure both regimes; don't quote only the large-tensor
  number.

- **Measure before optimising.** Benchmark the reference and each kernel variant under the same harness, same
  shapes, warmed up, multiple repetitions. Optimise the variant the measurement says is slow — not a guess.

- **Ablation is the deliverable.** Report GiB/s for: scalar reference → naive SSVE → VLA SSVE → ZA-tiled →
  single-pass-vs-two-pass, so each optimization's contribution is visible (decision C, E).

---

## 7. Kernel approach (the two norms, precisely)

The math, over the feature dimension of size `d` (ε is a small constant for stability):

- **LayerNorm** — `y = γ ⊙ (x − μ) / √(σ² + ε) + β`, with `μ = mean(x)`, `σ² = var(x)`.
  **Two-pass**: pass 1 reduces to mean and variance; pass 2 normalizes and applies the learned gain γ and
  bias β. Higher arithmetic intensity, numerically stable. Used in BERT, GPT-2/3, the original Transformer
  (Ba, Kiros & Hinton, 2016 · arXiv:1607.06450).

- **RMSNorm** — `y = γ ⊙ x / RMS(x)`, with `RMS(x) = √( (1/d) Σ xᵢ² + ε )`.
  **Single-pass**: one sum-of-squares reduction, then normalize and apply γ. No mean subtraction, **no β**.
  Lower arithmetic intensity, ~10–40% faster throughput, same accuracy. Used in LLaMA, Gemma, Mistral
  (Zhang & Sennrich, NeurIPS 2019 · arXiv:1910.07467).

The reference C++ implementations of both are written first and kept deliberately simple — they are the
correctness oracle (decision B), never the performance target.

> Note: γ and β are inputs to the kernel (learned per-feature vectors), not computed. The kernel computes the
> normalization statistics and applies the given γ/β; it does not train anything.

---

## 8. The hard problem: numerical stability & streaming overhead (honest-engineering selling point)

Two real problems the project surfaces rather than hides:

- **Single-pass variance is numerically dangerous.** Computing variance as `E[x²] − E[x]²` in one pass is
  cheap but suffers catastrophic cancellation in FP32 for large or shifted inputs. LayerNorm therefore uses
  the stable **two-pass** formulation (mean first, then variance from centered values). RMSNorm sidesteps the
  problem entirely — it only needs `Σ xᵢ²`, no mean subtraction — which is *part of why* it is both faster
  and stable. The verification (decision B) must include inputs that would expose cancellation, not just
  benign data. Document this as a deliberate accuracy/throughput tradeoff, not an accident.

- **Streaming mode is not free.** SME work runs inside a streaming region; `SMSTART`/`SMSTOP` and the
  associated state transition cost cycles. Toggling per row destroys performance. The kernel keeps the entire
  norm inside one streaming region and measures the small-tensor regime where this cost is visible (§6).
  Reporting *when* the kernel wins and when the overhead dominates is itself a result.

These are exactly the "explain the real tradeoff" talking points the lab rewards.

---

## 9. TEIR integration & compiler flow (the "real grounding" layer — Sprint 4)

- **Register the norms as TEIR primitives.** TEIR describes tensor ops as axes + primitives + a schedule of
  iteration/invocation nodes. The norm becomes a primitive the runtime can place in a generated loop nest,
  invoking the JIT-generated kernel per tile. This is what lets a real graph (LLaMA-style) call the kernel
  *through the compiler*, not via a hand-written test driver.

- **Loop-nest integration.** The primitive must behave under TEIR's scheduling: correct leading-dimension
  handling, tile-at-a-time operation, composing with the iteration nodes the runtime emits (decision F).

- **Correctness through the full path.** Verify the TEIR-invoked result against the C++ reference for the
  same shapes, so integration bugs (layout, stride, tile boundaries) are caught — not just the bare kernel
  (decision B).

> TEIR specifics follow the lab's runtime conventions; see `https://tnzr.org/compile/chapters/teir.html` and
> the course `teir/` exercise. Confirm the primitive-registration API against the version of the runtime the
> implementation actually uses.

---

## 10. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Target ISA | AArch64 + **SME / SSVE** | Streaming SVE for reductions; ZA tile for 2D movement |
| Target hardware | SME-capable AArch64 (e.g. **Apple M4**, SME1) | Confirm SME level before using SME2-only instructions |
| Kernels | Hand-written assembly, then **JIT-emitted** | 16×16 FP32 tiles; VLA, no hard-coded SVL |
| Code generator | **`mini_jit`** (extended with `Norm`) | New `mini_jit::Norm` alongside `Unary`/`Gemm`; dynamic instruction emission |
| Compiler IR | **TEIR** (tensor expression IR) | Norms registered as primitives; loop-nest integration |
| Reference | Plain **C++** | Correctness oracle; intentionally un-optimized (decision B) |
| Build | **CMake** + a C++ toolchain | Same toolchain as the lab exercises |
| Parallelism | **OpenMP** for the outer loop nest (via TEIR) | The kernel itself is single-tile; TEIR parallelizes across tiles |
| Tests | Course unit-test setup (e.g. Catch2/CTest) | Numerical checks vs reference; tail/edge shapes |
| Benchmark | Custom GiB/s harness + ablation | Effective bandwidth vs measured peak; multiple repetitions |
| Profiling | Counters / timing in the harness | Measure before optimizing (§6) |
| CI | GitHub Actions (build + tests) | Note: SME execution may not be available on CI runners — see CLAUDE.md |

No live-data, no external services — this is a self-contained native project. Config that varies (shapes,
tile sizes) is passed to the generator/harness, never hard-coded into kernels (decisions D, F).

---

## 11. Evaluation & benchmarking methodology (Sprint 5)

The proposal's evaluation has two halves; both are part of "done":

- **Correctness** — numerical verification of every kernel (and the TEIR-integrated path) against the C++
  reference within an FP32 tolerance, including inputs designed to stress numerical stability (§8). No
  performance number is reported for a kernel that has not passed (decision B).
- **Performance** — **GiB/s** effective bandwidth against the machine's measured peak, across a range of
  shapes (small to large feature dimensions, multiple row counts), with a clear statement of bytes counted
  (§6). Plus an **ablation**: scalar → naive SSVE → VLA SSVE → ZA-tiled → single-pass vs two-pass, so each
  optimization's contribution is attributable, and LayerNorm-vs-RMSNorm is quantified on the same harness.

> Story for the report/defense: "normalization is bandwidth-bound and runs in every block; we generate VLA
> SME kernels for both norms, verify them against a C++ reference, and show how close we get to the roofline —
> and why RMSNorm's single pass wins." Shows the tool was matched to the bottleneck, and the result is honest.

---

## 12. Repository layout (target — reconcile with the actual MLC-project repo)

```
mlc-project/
├── context.md  CLAUDE.md  ROADMAP.md  README.md
├── CMakeLists.txt
├── src/
│   ├── mini_jit/
│   │   ├── Norm.h   Norm.cpp        # NEW: LayerNorm/RMSNorm JIT generator (canonical signature, decision A)
│   │   ├── Unary.h  Gemm.h          # existing lab generators (extended/reused)
│   │   ├── instructions/            # SSVE/SME instruction-word encoders
│   │   └── Kernel.*                 # buffer + get_kernel (function pointer)
│   ├── reference/                   # plain C++ LayerNorm/RMSNorm — correctness oracle (decision B)
│   └── teir/                        # TEIR primitive registration + loop-nest integration (Sprint 4)
├── test/                            # numerical correctness tests vs reference (incl. stability cases)
├── bench/                           # GiB/s benchmark + ablation drivers (decision E)
├── data/                            # test tensors / shape configs
└── .github/workflows/ci.yml         # build + tests (note SME-execution limits on runners)
```

> This mirrors the lab's `mini_jit` conventions (`Unary`, `Gemm` → add `Norm`) and is the **expected** shape.
> Confirm and adjust against the real `MLC-project` repo before relying on exact paths.

---

## 13. How to build & run locally (once built out)

```bash
# Build (on an SME-capable AArch64 machine, e.g. Apple M4)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Correctness: every kernel verified against the C++ reference
ctest --test-dir build --output-on-failure

# Performance: effective bandwidth (GiB/s) + ablation
./build/bench/norm_bench   # reports GiB/s per kernel variant and shape
```

> Building is portable; **running the SME kernels requires SME hardware**. CI can build and run the C++
> reference and host-portable tests, but SME execution belongs on the target machine (see CLAUDE.md §CI).

---

## 14. Current status

The lab stack (assembly → NEON → SSVE/SME microkernels → `mini_jit` → TEIR) exists from the weekly exercises;
this project extends it with the two norm primitives. Build order and per-sprint tasks live in `ROADMAP.md`:
start at Sprint 0 (reference + harness + CI), then add one layer at a time — SSVE kernel, VLA, SME ZA tile,
JIT, TEIR integration, evaluation.

> The precise "what's already implemented" checklist should be filled in from the actual `MLC-project` repo
> (clone it / share the URL) — these three docs describe the target and the route, and `ROADMAP.md` is where
> the live status is tracked.
