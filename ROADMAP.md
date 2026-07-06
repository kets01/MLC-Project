# ROADMAP.md — MLC-Norm

> The path from today's lab foundation to the full vision in `context.md`.
> Golden rule: get a **correct, measured** norm kernel working end-to-end before optimizing or integrating it.
> Each sprint ends with something that BUILDS, is VERIFIED (vs the C++ reference), and is COMMITTED.
>
> `context.md` = the destination (JIT-generated SME/SSVE norm primitives, integrated into TEIR, measured at the roofline).
> `CLAUDE.md` = working conventions (git, commits, tests, correctness gates, ISA rules) and the tooling list.
> This file = the route. We start from the completed lab stack and add one layer at a time.
>
> The repo is **MLC-Project** (`github.com/kets01/MLC-Project`); "MLC-Norm" is the short handle for the final project.

---

## Where we start: the lab foundation already exists

The weekly lab (weeks 1–7) is **done and merged**, and the norm project is built on top of it. What is already in place:

| Lab week | What it gives the norm project |
|---|---|
| **week1** | AArch64 assembly basics (inner/outer product), GDB workflow |
| **week2** | NEON/SVE primitives (`fmadd`, permute) |
| **week3** | Hand-written **SME microkernels** (`identity`/`zero`/`relu`, GEMM 16×16…512), `cpu_supports_sme()` runtime guard |
| **week5** | **JIT engine** + executable kernel buffer (emit instruction words at runtime, get a function pointer) |
| **week6** | **Code generation**: `mini_jit::Unary` (SSVE, dynamic loop bounds `M×N/16`, ptypes identity/zero/relu), `mini_jit::Gemm` (SME `fmopa`, arbitrary K), `mini_jit::InstGen` (GPR/SIMD/**SVE predicate p0–p15**/Z-register encoders) |
| **week7** | **TEIR runtime**: `Axis`, `Iteration` (`is_parallel`), `Invocation`, stride handling, `CompiledKernel = void(float*,float*,float*)`; runs transposition/matmul/contraction |

This means the norm project does **not** start from scratch: the SSVE machinery (week6 `Unary`), the SME machinery (week3/week6 GEMM), the JIT (week5), and the TEIR runtime (week7) are the building blocks we extend. The norm kernels are the first *new primitive* added on top.

---

## Guiding principles

- **Learning first.** Understanding *why* a kernel sits where it does on the roofline matters more than a fast one-off number. After each non-trivial step, be able to explain it.
- **One new thing at a time.** Never debug two unfamiliar things at once (e.g. don't bring up the JIT *and* TEIR integration in the same step).
- **Vertical slices.** Each sprint delivers a thin slice that runs end-to-end — *input → kernel → result verified against the reference → GiB/s measured* — not a horizontal layer that can't be exercised.
- **Correctness gates performance (decision B).** No GiB/s number is reported for a kernel that hasn't passed numerical verification against the C++ reference.
- **Tests & CI from day one.** Every kernel ships with a Catch2 test; CI builds on every push. (Already the repo's habit.)
- **Docs grow with the project.** Update the Sphinx report at the END of every sprint, the way the weekly reports already work, so it always reflects the real current state and the latest GiB/s tables.
- **Measure before optimising (decision E/F).** Never tune blind — measure the reference and each variant on the same harness, then fix the real bottleneck.
- **Honest scope.** The full JIT + SME-tiled + TEIR-integrated system is the target, reached by adding layers — not the starting point. Reference and SSVE come first.
- **SME runs on hardware, not CI.** CI runners are Apple Silicon **without SME** (M1/M2), so SME/JIT kernels build in CI but are *run and benchmarked locally on M4*. Plan tests accordingly (see CLAUDE.md §CI).

---

## Cross-cutting design decisions (apply across all sprints)

Full rationale in `context.md §4`. Referenced here as A–F:

### A. One canonical primitive interface (single source of truth)
The norm kernel signature is defined once — extending the lab's unary convention (`a`, `b`, `ld_a`, `ld_b`, layout flag) with the γ/β pointers and ε — and shared **identically** by the C++ reference, the `mini_jit::Norm` JIT generator, and the TEIR registration. No layer invents its own. *Pinned down only after the reference and first kernel exist.*

### B. Correctness gates performance
Every kernel is verified against the scalar C++ reference (FP32 tolerance) **before** any benchmark is trusted. A fast wrong kernel is worthless. Numerical stability of the reduction is part of correctness — include stress inputs, not just benign data.

### C. Two norms, two compute strategies (an ablation axis, not a flag)
LayerNorm = **two-pass** (mean & variance, then normalize-scale-shift), higher arithmetic intensity, stable. RMSNorm = **single-pass** (one sum-of-squares; no mean, no β), lower intensity, ~10–40% faster at equal accuracy. The two reduction structures are deliberately different code and a measured ablation variable.

### D. Vector-Length-Agnostic (VLA)
No hard-coded streaming vector length. Query SVL at runtime; predicated tails (the `InstGen` p0–p15 predicates already exist); ZA-tile granularity = `SVL/32` FP32 elements (**16 on the M4's 512-bit SVL**, derived, not a magic constant). Runs unchanged across SME implementations.

### E. Memory-bandwidth bound → optimise movement, report GiB/s
Normalization is bandwidth-bound (touch every element, ~O(1) arithmetic). The metric is **effective bandwidth (GiB/s)** vs the machine's measured peak — not GFLOPS. Optimise data movement (keep the row resident between LayerNorm's two passes, tile for ZA, kill redundant loads). Match the repo's existing "GiB/s (Read + Write)" reporting.

### F. JIT codegen with a clean primitive boundary
Kernels are **emitted at runtime** by extending `mini_jit` with a `Norm` generator (following the week6 `Unary` pattern); emission is one-time, the function pointer is reused. The kernel respects leading dimensions and works tile-at-a-time so the week7 TEIR runtime can compose it into a loop nest — it never assumes it owns the whole tensor.

---

## Tooling (where it gets used)

Full rationale in `CLAUDE.md §8`. This is a low-level C++/AArch64-assembly project, so the document/UI skills don't apply; the real references are the hardware/architecture docs and the existing lab code:

- **The existing week5–7 code is the primary template** — `mini_jit::Unary`/`InstGen` (Sprint 3 JIT), the week7 TEIR runtime (Sprint 4). Read them before extending.
- **Arm Architecture Reference Manual (SME/SSVE)** + the lab's `tnzr.org/compile` material — the authority on instruction encodings and ZA/streaming semantics. Verify against the M4's SME level (SME1).
- **skill-creator** (optional, early): write a small MLC-Norm conventions skill as a learning step — no rush.
> Learning note: a generator can emit correct instructions you don't yet understand. Always have the agent explain the encoding/why — comprehension over an opaque build.

---

## MVP definition (Sprints 0–2)

A **correct, VLA SSVE kernel** for both norms — verified against the C++ reference and measured in GiB/s — driven from a benchmark app, no JIT and no TEIR yet. The JIT + SME-tiling + TEIR integration from `context.md` arrives in Sprint 3+.

---

## Sprint 0 — Norm scaffold on the existing foundation

**Goal:** prove the norm module is wired into the existing build/test/CI/docs machinery, with placeholder logic.

- [x] Lab foundation present (weeks 1–7 built, CI green, Catch2 + CMake + Sphinx pipeline working).
- [x] **Verify the baseline first — before adding anything:** on a clean checkout of the current repo, run a fresh `cmake` configure + `cmake --build` and the **full existing test suite** (weeks 1–7); confirm everything passes locally and CI is green on the untouched tree. This establishes a known-good baseline, so any breakage later is attributable to the norm work rather than inherited from the starting state.
- [x] Create the norm module following the per-week pattern: `src/norm/` (+ `CMakeLists.txt`), `include/norm/`, `apps/main_norm.cpp`, `tests/test_norm.cpp`; register `add_subdirectory(src/norm)` in the root `CMakeLists.txt`.
- [x] Placeholder reference + a trivial Catch2 test that builds and runs (host-portable, no SME yet).
- [x] Benchmark app stub in `apps/main_norm.cpp` that prints a GiB/s line (even on dummy data) — establishes the measurement path.
- [x] CI: add the norm test to `tests.yml` **in the host-runnable (non-SME) group**, so it runs on the macOS runner like week1/week2.
- [x] Docs: add `docs/source/project_norm.rst` (or a `weeks/`-style entry) and list it in `index.rst` — "Sprint 0: scaffold".
- [x] Feature branch + PR; conventional commit messages; build green.

**Done when:** `cmake --build` produces `test_norm` and `main_norm`, the test passes in CI, and the docs section exists.
**Tooling:** none required (optionally try skill-creator).
**Learning focus:** how a new primitive slots into the lab's CMake/Catch2/Sphinx/CI structure.

---

## Sprint 1 — The C++ reference + correctness & bandwidth harness (the oracle and the ruler)

**Goal:** an obviously-correct reference for both norms, the verification harness, and the GiB/s measurement — everything later is judged against these.

- [x] Scalar **C++ reference**, kept deliberately simple (decision B): LayerNorm `y = γ(x−μ)/√(σ²+ε)+β` (two-pass), RMSNorm `y = γ·x/√(mean(x²)+ε)` (single-pass). γ/β are inputs.
- [x] **Numerical verification harness** (Catch2): compare a kernel's output to the reference within an FP32 tolerance, over several shapes — **including stability-stress inputs** (large magnitude / shifted values) that would expose single-pass cancellation (context.md §8).
- [x] **Bandwidth harness**: measure effective GiB/s = useful bytes (1 read + 1 write) / time, warmed up, repeated; establish the machine's **peak bandwidth** (a STREAM-style probe) as the roofline target (decision E).
- [x] Define the **canonical kernel signature** now that the reference exists (decision A) in `include/norm/`.
- [x] Tests + a first GiB/s table for the reference itself; update the Sphinx report.

**Done when:** both references are verified-correct and a reproducible GiB/s number exists for them, with the roofline target recorded.
**Tooling:** none.
**Learning focus:** the roofline, honest byte-counting, why two-pass variance is the stable choice.

---

## Sprint 2 — Hand-written SSVE kernel (the MVP norm)

**Goal:** a correct, vectorized, **VLA** norm kernel for both norms — the MVP.

**RMSNorm (Ketsia):**
- [x] Hand-written SSVE kernel (`rms_norm_ssve.S`): column-major loop order (outer VL-row blocks, inner N columns) using `LD1W`/`LD1RW`/`ST1W` — avoids gather loads restricted in SSVE without SMEFA64.
- [x] VLA (decision D): `WHILELO`/`INCW`/`ADDVL` outer loop; predicated tail handles any M.
- [x] Guard with `cpu_supports_sme()`; 5 Catch2 test cases skip on CI, all pass on M4 (20 356 assertions, incl. stress inputs and mismatched leading dims).
- [x] Verified vs reference; 18–25 GiB/s on M4 (23–32 % of scalar peak); report updated.

**LayerNorm (Mariza):**
- [ ] Hand-written SSVE kernel: two-pass (mean + variance in pass 1, normalize-scale-shift in pass 2), reduction is SSVE not ZA.
- [ ] VLA (decision D): predicated tail for any N.
- [ ] Guard with `cpu_supports_sme()`; Catch2 tests skip on CI, run on M4.
- [ ] Verified vs reference (incl. stress inputs); GiB/s measured; report updated.

**Done when:** both norms are correct and vectorized, measured in GiB/s on M4, with a clear gap-to-roofline noted.
**Tooling:** none (week6 `InstGen`/`Unary` as reference).
**Learning focus:** SSVE reductions, predication/tails, VLA, keeping data resident.

> 🎉 End of Sprint 2 = a working, correct, measured MVP norm kernel. Good point to commit, refresh the report, and breathe. Everything below is enrichment toward the full `context.md` vision.

---

## Sprint 3 — JIT generation (extend mini_jit with a Norm generator)

**Goal:** generate the norm kernel at runtime instead of hand-writing it — the `mini_jit::Norm` generator.

- [ ] Add `mini_jit::Norm` following the week6 `Unary` pattern: `generate(...)` emits instruction words via `InstGen`; `get_kernel()` returns a function pointer (decision F). Parameterize on shape and norm type.
- [ ] Reproduce the Sprint-2 SSVE kernel through the generator; verify the JIT-emitted kernel matches the reference (within tolerance).
- [ ] **SME ZA-tile usage where it earns its place** (decision D/§5): the 2D tiled load/store/transpose path (e.g. column-major input), tile granularity derived from SVL — *not* the 1D reduction.
- [ ] Keep the whole norm inside one streaming region (`SMSTART`/`SMSTOP` once — context.md §8).
- [ ] Tests: JIT-emitted kernel vs reference; benchmark JIT vs hand-written; report emission (one-time) vs execution cost.

**Done when:** the norm kernel is JIT-generated, verified, and at least matches the hand-written kernel's GiB/s.
**Tooling:** week5 JIT engine + week6 `InstGen`/`Unary` as the template.
**Learning focus:** dynamic instruction emission, ZA-tile addressing, streaming-region management.

---

## Sprint 4 — TEIR integration (norm as a compiler primitive)

**Goal:** let the week7 TEIR runtime place the norm in a loop nest and invoke the JIT kernel — the `context.md` flow, end to end.

- [ ] Register the norm as a TEIR **primitive** (an `Invocation` the runtime resolves to the generated kernel), reconciling the canonical signature with TEIR's `CompiledKernel = void(float*,float*,float*)` convention (map input/γβ/output onto a/b/c).
- [ ] Drive a small multi-row tensor through TEIR `Iteration` nodes → per-tile `Invocation` → generated kernel; honor strides and leading dimensions (decision F).
- [ ] Verify the **TEIR-invoked** result against the C++ reference for the same shapes (catches layout/stride/tile-boundary bugs, not just the bare kernel).
- [ ] Optional: use `is_parallel` on the outer (row) axis for OpenMP across tiles; the kernel stays single-tile.
- [ ] Update the report with the integrated path and its GiB/s.

**Done when:** a model-style call reaches the norm kernel *through TEIR* and is verified correct.
**Tooling:** week7 TEIR runtime as the template.
**Learning focus:** primitive registration, loop-nest composition, stride/layout correctness.

---

## Sprint 5 — Optimization, roofline & the ablation study

**Goal:** the proposal's evaluation deliverable — close on the roofline and attribute every gain.

- [ ] Measure-driven optimization (decision E/F): identify the real bottleneck from the harness, then tune (resident rows, load/store scheduling, ZA tiling, fewer streaming transitions).
- [ ] **Ablation study**: GiB/s for scalar reference → naive SSVE → VLA SSVE → JIT → ZA-tiled, and **LayerNorm two-pass vs RMSNorm single-pass** on the same harness (decision C). Each row attributable.
- [ ] Report GiB/s **vs measured peak bandwidth** across shapes (small→large feature dims, varying row counts); state bytes counted.
- [ ] Verify every optimized configuration still matches the reference (decision B).

**Done when:** an ablation table shows each optimization's contribution and how close to peak the best kernel gets.
**Tooling:** none new.
**Learning focus:** roofline analysis, ablation methodology, attributing speed-ups.

---

## Sprint 6 — Numerical-stability & streaming-overhead depth (honest engineering)

**Goal:** the interview/report talking point — surface and explain the real tradeoffs (context.md §8).

- [ ] Characterize **numerical stability**: show where naive single-pass variance breaks (catastrophic cancellation) and why LayerNorm's two-pass and RMSNorm's mean-free reduction are stable; quantify error vs the reference on stress inputs.
- [ ] Characterize **streaming-mode overhead**: measure the small-tensor regime where `SMSTART`/`SMSTOP` dominates vs the large-tensor regime where it amortizes; report *when* the SME kernel wins.
- [ ] Document both as deliberate tradeoffs (not accidents) in the report — this is a strength.

**Done when:** the report explains the accuracy/throughput and small/large-tensor tradeoffs with data.
**Tooling:** none new.
**Learning focus:** FP numerics, fixed-cost amortization — strong defense talking points.

---

## Sprint 7 — Report & ship

- [ ] Final Sphinx report section: motivation (norm in every block, bandwidth-bound), the two norms, the SSVE/JIT/SME/TEIR path, correctness, the GiB/s ablation, the stability/overhead analysis.
- [ ] Final benchmark tables refreshed on M4; figures (roofline, ablation bars).
- [ ] README updated; report deployed to GitHub Pages via the existing `docs.yml`.
- [ ] Tidy CI: norm host-tests in the runnable group; SME tests clearly marked local-only (mirroring the commented-out week3 group).

**Done when:** the report tells the full story and the integrated kernel is verified + measured.

---

## How to use this file with Claude / Claude Code

- Work **one sprint at a time**. Don't pull tasks from a later sprint early.
- Session start: "Read context.md and CLAUDE.md, then let's do Sprint N."
- Tick boxes as we go; the checked history shows progress.
- Every kernel ships with a Catch2 test; build with `-Werror` clean and verify vs the reference before committing.
- SME kernels: build in CI, **run and benchmark on M4**; keep CI green on the M1/M2 runners by skipping SME tests when `cpu_supports_sme()` is false.
- If a sprint feels too big, split its first checkbox into a vertical slice (one norm, one shape, verified) and get THAT running before continuing.
