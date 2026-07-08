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

- **The existing week5–7 code is the primary template** — the week3 SME kernels + week6 `Gemm` (Sprint 3 ZA), `mini_jit::Unary`/`InstGen` (Sprint 4 JIT), the week7 TEIR runtime (Sprint 5). Read them before extending.
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

## Sprint 2 — Hand-written SSVE kernel (the MVP norm) + roofline validation & bandwidth optimization

**Goal:** a correct, vectorized, **VLA** norm kernel for both norms, measured against a **validated** roofline, with the hand-written optimization space explored to its verdict — the MVP, plus the evidence for what the JIT should emit.

### 2a. Roofline validation (shared — DO THIS BEFORE any further optimization)

The V0–V3 conclusions hinge on "% of peak", so the peak itself must be validated first — otherwise every subsequent optimization decision is calibrated against the wrong ceiling.

- [ ] **Check what "peak" measures.** Determine whether the 79.58 GiB/s "vectorised peak" is a **chip-wide** (multi-core) STREAM figure or a **single-core** one. The kernel runs on ONE core; a single P-core physically cannot reach full-chip bandwidth (the memory system saturates per-core well below the SoC aggregate). If the probe was multi-threaded, the "31–34 % of peak" conclusion understates how good the kernel already is.
- [ ] **Measure the single-core roofline:** re-run the STREAM-style probe pinned to one P-core, single-threaded, with the same vectorised access pattern the kernel uses (`LD1W`/`ST1W`, same stride). Record **both** ceilings (single-core and chip-wide) in the report and in the benchmark table header.
- [ ] **Recompute every % -of-peak against the single-core ceiling.** This is the number that judges kernel quality; the chip-wide ceiling is the target for TEIR/OpenMP scaling (Sprint 5), not for a single-threaded kernel.
- [ ] **Validate the byte-counting convention.** The current RMSNorm is two-pass over memory (reduction pass reads x, normalize pass reads x again, then writes y). State explicitly whether GiB/s counts *useful* bytes (1R+1W, the algorithm's minimum) or *moved* bytes (2R+1W as implemented) — and use one convention consistently for kernels AND the STREAM probe. If useful-bytes is the convention, the two-pass structure shows up honestly as a lower % of peak, which is precisely the gap pass-residency (V6) attacks.
- [ ] **Characterize the small-N regime separately:** the N=64 numbers (~22 %) sit well below the N=2048 numbers (~32 %) — quantify the fixed per-row/per-call overhead (loop setup, `inv_rms` serialization, streaming-region entry) so small-N is explained rather than averaged away. (Full streaming-overhead study stays in Sprint 7; here just record the regime split.)
- [ ] **Decision gate:** if V1 is already near the *single-core* roofline (≳80–85 %), stop hand-written SSVE tuning — record "single-core bandwidth-bound; further scaling comes from threading across rows (Sprint 5)" and move on (the Sprint-3 ZA prototype then tests the *residency* levers, not raw throughput). Only if real single-core headroom remains do V4–V6 below proceed. Optimizing against headroom that doesn't exist wastes the schedule.

### 2b. RMSNorm (Ketsia)

- [x] Hand-written SSVE kernel (`rms_norm_ssve.S`): column-major loop order (outer VL-row blocks, inner N columns) using `LD1W`/`LD1RW`/`ST1W` — avoids gather loads restricted in SSVE without SMEFA64.
- [x] VLA (decision D): `WHILELO`/`INCW`/`ADDVL` outer loop; predicated tail handles any M.
- [x] Guard with `cpu_supports_sme()`; 5 Catch2 test cases skip on CI, all pass on M4 (20 356 assertions, incl. stress inputs and mismatched leading dims).
- [x] Verified vs reference; 18–25 GiB/s on M4 (23–32 % of scalar peak); report updated.
- [x] **Ablation study (V0–V3):** three kernel variants measured against the V0 baseline to isolate the performance ceiling:
  - [x] **V1** (`rms_norm_ssve_v1.S`): replaces FSQRT + FDIV (inv_rms) with FRSQRTE + one Newton-Raphson step; halves inv_rms latency from ~24 to ~12 cycles. **Result: +6.5 % at M=128, N=2048** — the only variant with a measurable gain.
  - [x] **V2** (`rms_norm_ssve_v2.S`): pre-computes `1/N` once (scalar FDIV before the outer loop) and replaces the per-block vector FDIV with a vector FMUL. **Result: ≈0 % — the FDIV fires once per outer block, not per element; not the bottleneck.**
  - [x] **V3** (`rms_norm_ssve_v3.S`): adds ×2 column-loop unroll to V2 (peel-one for odd N, then pairs). **Result: ≈0 % — the M4's hardware prefetcher and OOO execution already overlap sequential column loads; branch overhead is not limiting.**
  - [x] 10 ablation Catch2 tests (tagged `[sprint2][ablation]`); all pass on M4; ablation benchmark table added to `apps/main_norm.cpp`.
  - **Interim conclusion:** kernel is bandwidth-bound at **~31–34 % of the 79.58 GiB/s vectorised peak** (best: 26.74 GiB/s, V1) — *pending re-statement against the single-core ceiling (2a)*. The two-pass sequential structure is the structural ceiling; closing the gap further requires residency/pipelining (V4–V6 below) or ZA tiling / pass-fusion at the tile level (Sprint 3).

**Bandwidth/ILP ablation, round two (V4–V6) — gated by the 2a decision gate.** What V0–V3 proved: arithmetic tweaks (V1) buy little, scalar hoisting (V2) and loop unrolling (V3) buy nothing — the prefetcher and OOO core already cover them. What remains untested are the levers that change *memory behaviour*, not arithmetic. Each variant: verified vs reference (same suite, same stress inputs), benchmarked on the standard M×N grid, and kept **only if it beats the incumbent against the single-core roofline**; every result — positive or negative — gets an ablation-table row with its explanation (a explained negative is a result, not a failure).

- [ ] **V4 — multi-accumulator reduction ILP:** 2–4 independent `FMLA` accumulator Z-registers per row block, combined once at the end before the `inv_rms` step. Breaks the single-accumulator dependency chain so more loads can be in flight behind independent FMAs. (Targets: latency-bound accumulation, which V1's +6.5 % hints still exists at mid sizes.)
- [ ] **V5 — load software-pipelining:** issue the next iteration's `LD1W`s into spare Z-registers ahead of the current iteration's `FMLA`s (explicit double-buffering of loads). Distinct from V3's unrolling: V3 reduced branch overhead (already free); V5 raises the number of outstanding loads, which is the actual bandwidth-utilization lever. Optionally test explicit `PRFW` prefetch — expected ≈0 given V3's finding, but cheap to confirm and worth an ablation row.
- [ ] **V6 — two-pass residency blocking:** restructure so the normalize pass re-reads x from **cache, not DRAM**: block the (row-block × column) working set to fit L1/L2 where N allows, or interleave reduction and normalize at the block level. This attacks the structural 2R+1W → effective 1R+1W gap identified in 2a — the largest remaining lever available without ZA. Measure with both byte conventions to show the DRAM-traffic reduction explicitly.
- [ ] **Vectorized `inv_rms` across row blocks (V6b, cheap add-on):** where multiple row-blocks' sums are available together, compute their `FRSQRTE`+NR on one packed vector instead of per-block — amortizes V1's remaining serial cost. Only worthwhile if profiling still shows `inv_rms` in the critical path after V4/V5.
- [ ] Update the ablation table + report; **final Sprint-2 RMSNorm conclusion** states: best variant, % of single-core roofline, which levers were exhausted, and what is deliberately left for the hand-written ZA kernel / tile-level pass fusion (Sprint 3), the JIT (Sprint 4), and threading (Sprint 5).

### 2c. LayerNorm (Mariza)

- [ ] Hand-written SSVE kernel **V0**: two-pass (mean + variance in pass 1, normalize-scale-shift in pass 2), reduction is SSVE not ZA. Use the **stable two-pass variance** (centered second pass), never `E[x²]−E[x]²` (decision B / context.md §8).
- [ ] VLA (decision D): predicated tail for any N.
- [ ] Guard with `cpu_supports_sme()`; Catch2 tests skip on CI, run on M4.
- [ ] Verified vs reference (incl. stress inputs: large-magnitude / shifted values — the cancellation cases); GiB/s measured against **both ceilings from 2a**; report updated.
- [ ] **Ablation study (mirroring RMSNorm, plus the LayerNorm-specific axis):**
  - [ ] Replay the applicable arithmetic variants first (V1 FRSQRTE+NR for `inv_std`; V2 pre-computed 1/N) — LayerNorm's two-pass structure shifts the hotspot profile vs RMSNorm; characterize separately rather than assuming the RMSNorm verdicts transfer.
  - [ ] Apply the winning memory-behaviour levers from 2b (V4 multi-accumulator, V5 load pipelining) — pass 1 accumulates TWO statistics (Σx and Σ(x−µ)² ingredients), so the accumulator-ILP payoff may differ.
  - [ ] **Residency between the two passes (the LayerNorm V6):** keep the row block resident in registers/cache from pass 1 into pass 2 — LayerNorm reads x up to 3× naively (mean, variance, normalize), so residency blocking has MORE headroom here than in RMSNorm; count bytes under both conventions to show it.
  - [ ] **Welford single-pass variant vs two-pass-resident (the algorithmic ablation, decision C):** implement a vectorized Welford (per-lane partial `(mean, M2, count)`, merged with the parallel-combine formula), verify on the stress inputs, and measure accuracy AND GiB/s against the two-pass kernel. Hypothesis to test: on SIMD the two-pass wins because Welford's per-element recurrence serializes — but either outcome is a headline ablation row, and this is the comparison production libraries actually face.
- [ ] **LayerNorm vs RMSNorm on the same harness, same shapes** (decision C): quantify the single-pass-vs-two-pass gap — the proposal's "10–40 % faster" claim, now with our own numbers.

### 2d. Sprint close-out (docs + commits — same discipline as Sprints 0/1)

- [ ] **Sphinx report updated:** refresh `docs/source/project_norm.rst` with the Sprint-2 story — kernel design (both norms), the full V0–V6 ablation table stated against **both validated ceilings**, the byte-counting convention, the Welford-vs-two-pass verdict, and the small-N/large-N regime split. Rebuild locally; `docs.yml` deploy stays green.
- [ ] **Commit discipline maintained throughout, not at the end:** each variant/test lands as a small atomic commit with a conventional message (`feat(norm): …`, `perf(norm): add V4 multi-accumulator ILP`, `test(norm): …`) on a feature branch; PR into `main` with CI green before merge. The V-by-V history *is* the ablation narrative in git form.

**Done when:** both norms are correct and vectorized; every % -of-peak is stated against the **validated single-core roofline** (with the chip ceiling recorded for Sprint 5); the V4–V6 (and Welford) verdicts are in the ablation table with explanations; the Sprint-2 conclusion says explicitly which levers are exhausted and which are deliberately deferred to the ZA kernel (Sprint 3), the JIT (Sprint 4), and threading (Sprint 5); and the report section is merged with the sprint PR green.
**Tooling:** none (week6 `InstGen`/`Unary` as reference; `objdump` for spot-checking assembly).
**Learning focus:** SSVE reductions, predication/tails, VLA, roofline validation (peak of *what*), byte-counting honesty, memory-level parallelism vs arithmetic tuning, Welford-vs-two-pass on SIMD.

> 🎉 End of Sprint 2 = a working, correct, measured MVP norm kernel with a validated roofline and an exhausted hand-written **SSVE** optimization space. Good point to commit, refresh the report, and breathe. Everything below is enrichment toward the full `context.md` vision — the ZA kernel (Sprint 3) now has a settled SSVE baseline to beat, and the JIT (Sprint 4) will have trusted hand-written counterparts for both architectures.

---

## Sprint 3 — Hand-written SME/ZA kernel + measurement (the second kernel architecture)

**Goal:** the ZA-tile variant of the norm, hand-written and measured with the same discipline as Sprint 2 — so the JIT (Sprint 4) has a trusted hand-written counterpart for BOTH kernel architectures, and the ablation can attribute exactly what ZA buys on top of the exhausted SSVE space.

> Why hand-written first (not "born in the generator"): the Sprint-4 JIT is verified by the
> **encoding-diff check** — assembling the winning `.S` and diffing its instruction words
> against the generator's buffer. That methodology requires a hand-written kernel to diff
> against. Prototyping ZA in `.S` also keeps the fast iteration loop (edit → rebuild →
> measure) for the most invasive structural change, and mirrors the lab's own order
> (week3 hand-written SME before week5/6 codegen). One new thing at a time: this sprint's
> new thing is ZA; Sprint 4's new thing is emission.

**Honest expectation-setting (write this down BEFORE measuring):** ZA does **not** increase
DRAM bandwidth. If Sprint 2a's verdict was "near single-core roofline", ZA cannot beat the
SSVE winner on streaming throughput — its levers are different: the ZA array as **extra
on-core storage** for two-pass residency (a 16×16-FP32-tile staging buffer beyond the 32
Z-registers), tile-level **pass fusion**, batched multi-row handling of the `inv_rms`/`inv_std`
step, and 2D load/store movement for the column-major layout. State the hypothesis per lever,
then measure. "ZA adds nothing here, and here is why" is a fully valid — and defensible —
ablation outcome; a timeboxed prototype with an explained verdict beats an open-ended chase.

**RMSNorm ZA variant (Ketsia):**
- [ ] Hand-written ZA kernel (`rms_norm_za.S`): 16×16 FP32 tile loads/stores (granularity
      derived from SVL, decision D), reduction **stays SSVE** (context.md §5 — no horizontal
      reduction through the matrix accumulator); ZA used for tile staging/residency and the
      2D movement.
- [ ] `SMSTART`/`SMSTOP` manage **PSTATE.ZA as well as PSTATE.SM**; whole norm in one
      streaming region; ZA state handled correctly across calls (lazy-save / caller
      conventions checked against the AAPCS64 SME rules).
- [ ] Tile-level **pass fusion / residency**: reduce a tile, keep it in ZA, normalize it from
      ZA — the deferred Sprint-2 lever, now with the storage to express it.
- [ ] Verified vs reference (same Catch2 suite, same stress inputs, mismatched leading dims);
      skip-on-CI guard as usual.
- [ ] Benchmarked on the standard M×N grid against **both ceilings from 2a**; ablation rows
      vs the SSVE winner, each lever's contribution attributed (or its absence explained).

**LayerNorm ZA variant (Mariza) — gated:**
- [ ] Only if the RMSNorm ZA verdict is positive (or the residency lever shows promise —
      LayerNorm's up-to-3-reads structure has more residency headroom): apply the winning ZA
      structure to LayerNorm; otherwise record the reasoned skip as an ablation note.

**Sprint close-out (docs + commits):**
- [ ] **Sphinx report updated:** Sprint-3 section — the per-lever hypotheses *as written before measuring*, the ZA kernel design (ZA staging, SSVE reduction, PSTATE.ZA handling), and the measured verdict vs the SSVE winner (win/loss/tie, each explained). Rebuild locally; `docs.yml` green.
- [ ] **Commit discipline throughout:** atomic conventional commits per working step (`feat(norm): add ZA-staged rms_norm kernel`, `test(norm): …`) on a feature branch; PR into `main`, CI green before merge.

**Done when:** a verified ZA kernel exists with a measured verdict vs the SSVE winner (win,
loss, or tie — each with its explanation), the best kernel **per architecture** is frozen
for the JIT, and the report section is merged with the sprint PR green.
**Tooling:** week3 hand-written SME kernels + week6 `Gemm` (ZA/`fmopa` usage) as templates;
Arm ARM for ZA load/store & tile-slice addressing (SME1 on the M4 — no SME2-only instructions);
GDB for tile-state inspection.
**Learning focus:** ZA addressing and tile slices, PSTATE.ZA management, when a matrix
accumulator does and doesn't help a bandwidth-bound vector op.

---

## Sprint 4 — JIT generation (extend mini_jit with a Norm generator, emitting BOTH winners)

**Goal:** generate the norm kernels at runtime instead of hand-writing them — the `mini_jit::Norm` generator, covering the SSVE winner (Sprint 2) and, if it earned its row, the ZA winner (Sprint 3).

- [ ] Add `mini_jit::Norm` following the week6 `Unary` pattern: `generate(...)` emits instruction words via `InstGen`; `get_kernel()` returns a function pointer (decision F). Parameterize on shape, norm type, and **kernel path (SSVE / ZA)**.
- [ ] **Emit the winners, not the baselines — one path at a time, encoding-diff first per path:** assemble the winning `.S` with the toolchain, `objdump` its instruction words, and diff word-by-word against the generator's buffer for one fixed shape — a green diff proves the generator byte-identical to a kernel already trusted, and turns debugging into "word K differs". SSVE path first (fewer new encoders), then the ZA path by the same method.
- [ ] Missing `InstGen` encoders (`FRSQRTE`, `FRSQRTS`, `LD1RW`, `FADDV`, ZA tile load/store…) added with per-encoder unit tests against the Arm ARM.
- [ ] Verify each JIT-emitted path against the reference (same Catch2 suite, same tolerances, same stress inputs); benchmark parity vs its hand-written counterpart (within noise).
- [ ] **Shape-dependent emission choices** become the JIT-native optimization: pick SSVE vs ZA path per shape from the measured Sprint-2/3 verdicts (e.g. ZA-residency only when the working set warrants it; small-N specialization) — decisions a hand-written kernel can't express.
- [ ] Keep the whole norm inside one streaming region (`SMSTART`/`SMSTOP` once — context.md §8).
- [ ] Tests: JIT-emitted kernels vs reference; benchmark JIT vs hand-written per path; report emission (one-time) vs execution cost.

**Sprint close-out (docs + commits):**
- [ ] **Sphinx report updated:** Sprint-4 section — the generator design (`mini_jit::Norm`, new `InstGen` encoders), the encoding-diff methodology as the verification story, the parity tables (per path), and emission-vs-execution cost. Rebuild locally; `docs.yml` green.
- [ ] **Commit discipline throughout:** atomic conventional commits (`feat(jit): add FRSQRTE encoder`, `feat(jit): emit SSVE norm path`, `test(jit): encoding diff vs rms_norm_ssve_v1`) on a feature branch; PR into `main`, CI green (the generator and its encoder unit tests build and run host-portable in CI; emitted-kernel execution tests skip without SME as usual).

**Done when:** both norm kernels are JIT-generated on the path(s) that earned it, verified, at least matching their hand-written counterparts' GiB/s, and the report section is merged with the sprint PR green.
**Tooling:** week5 JIT engine + week6 `InstGen`/`Unary` as the template; `objdump`/`llvm-mc` for the encoding diff; Arm ARM for the new encoders.
**Learning focus:** dynamic instruction emission, encoding verification, streaming-region management, emission-time specialization.

---

## Sprint 5 — TEIR integration (norm as a compiler primitive)

**Goal:** let the week7 TEIR runtime place the norm in a loop nest and invoke the JIT kernel — the `context.md` flow, end to end.

- [ ] Register the norm as a TEIR **primitive** (an `Invocation` the runtime resolves to the generated kernel), reconciling the canonical signature with TEIR's `CompiledKernel = void(float*,float*,float*)` convention (map input/γβ/output onto a/b/c).
- [ ] Drive a small multi-row tensor through TEIR `Iteration` nodes → per-tile `Invocation` → generated kernel; honor strides and leading dimensions (decision F).
- [ ] Verify the **TEIR-invoked** result against the C++ reference for the same shapes (catches layout/stride/tile-boundary bugs, not just the bare kernel).
- [ ] Use `is_parallel` on the outer (row) axis for OpenMP across tiles; the kernel stays single-tile. **This is where the chip-wide roofline from Sprint 2a becomes the target:** measure multi-threaded GiB/s scaling toward the SoC ceiling and report both (single-core kernel quality vs threaded aggregate).
- [ ] Update the report with the integrated path and its GiB/s.

**Done when:** a model-style call reaches the norm kernel *through TEIR* and is verified correct, with threaded scaling measured against the chip ceiling.
**Tooling:** week7 TEIR runtime as the template.
**Learning focus:** primitive registration, loop-nest composition, stride/layout correctness, bandwidth scaling across cores.

---

## Sprint 6 — Optimization, roofline & the ablation study

**Goal:** the proposal's evaluation deliverable — close on the roofline and attribute every gain.

- [ ] Measure-driven optimization (decision E/F): identify the real bottleneck from the harness, then tune (resident rows, load/store scheduling, ZA tiling, fewer streaming transitions).
- [ ] **Ablation study**: GiB/s for scalar reference → naive SSVE → VLA SSVE → V4–V6 memory levers → ZA-tiled (hand-written) → JIT (per path) → TEIR+OpenMP, and **LayerNorm two-pass vs Welford vs RMSNorm single-pass** on the same harness (decision C). Each row attributable, negatives explained.
- [ ] Report GiB/s **vs both validated ceilings** (single-core and chip-wide) across shapes (small→large feature dims, varying row counts); state bytes counted and the convention used.
- [ ] Verify every optimized configuration still matches the reference (decision B).

**Done when:** an ablation table shows each optimization's contribution and how close to peak the best kernel gets — at both the single-core and the threaded level.
**Tooling:** none new.
**Learning focus:** roofline analysis, ablation methodology, attributing speed-ups.

---

## Sprint 7 — Numerical-stability & streaming-overhead depth (honest engineering)

**Goal:** the interview/report talking point — surface and explain the real tradeoffs (context.md §8).

- [ ] Characterize **numerical stability**: show where naive single-pass variance breaks (catastrophic cancellation) and why LayerNorm's two-pass and RMSNorm's mean-free reduction are stable; quantify error vs the reference on stress inputs — including the Welford variant's accuracy from Sprint 2c.
- [ ] Characterize **streaming-mode overhead**: measure the small-tensor regime where `SMSTART`/`SMSTOP` and fixed per-call costs dominate (the N=64 regime already visible in the Sprint-2 table) vs the large-tensor regime where they amortize; report *when* the SME kernel wins.
- [ ] Document both as deliberate tradeoffs (not accidents) in the report — this is a strength.

**Done when:** the report explains the accuracy/throughput and small/large-tensor tradeoffs with data.
**Tooling:** none new.
**Learning focus:** FP numerics, fixed-cost amortization — strong defense talking points.

---

## Sprint 8 — Report & ship

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