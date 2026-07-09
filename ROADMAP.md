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

- [x] **Check what "peak" measures.** VERDICT: disassembly shows the C++ probe autovectorized to **NEON** (`ldp q`/`fadd.4s`/`stp q`) and ran **single-threaded** — so 79.58 GiB/s was a *single-core NEON* figure. Single-core, yes; but the wrong *execution mode*: the kernels run streaming-mode `LD1W`/`ST1W`, which turned out to have a materially lower per-core ceiling.
- [x] **Measure the single-core roofline:** `bw_probe_ssve.S` (streaming-mode `LD1W`/`ST1W` scale-add, one streaming region, VLA tail, verified bit-exact; `QOS_CLASS_USER_INTERACTIVE` as the P-core hint — macOS has no pinning API). MEASURED (M4, ±0.3 % across runs): **single-core NEON 79.5 · single-core SSVE-streaming 59.5 (= the kernel roofline) · chip-wide 89.8 GiB/s** (10 threads). Streaming mode has ~25 % less single-core bandwidth than NEON; one core already sustains ~66 % of the chip aggregate, so Sprint-5 threading buys ≤1.5x. All three recorded in the report and printed by `main_norm`.
- [x] **Recompute every % -of-peak against the single-core ceiling.** Done in `main_norm` (all tables now state "% of 1-core SSVE peak"): large-N shapes are **42–46 %** of the validated roofline (was misstated as 31–34 % vs the NEON figure); best ~27 GiB/s.
- [x] **Validate the byte-counting convention.** DECIDED and printed in the benchmark header: GiB/s counts **useful bytes (1R+1W)** for kernels AND probes; the two-pass kernels physically move 2R+1W → moved-bytes figure = 1.5x printed (≈40 GiB/s ≈ 67 % of ceiling for the best variant). The useful-vs-moved gap is exactly what V6 attacks.
- [x] **Characterize the small-N regime separately:** N-sweep + linear fit `t(N) = t0 + b·N` in `main_norm`. M=128: fixed cost t0 ≈ 1 µs/call, overhead = streaming work at N ≈ 30–40 — the N=64 dip is fully explained. M=1024: fit invalid (negative t0) because the sweep crosses the **cache-capacity boundary** — at N=4096 (32 MiB working set) throughput falls 25 → 18.6 GiB/s as the normalize pass re-reads x from DRAM: the first direct measurement of the 2R+1W structural cost (evidence for V6). (Full streaming-overhead study stays in Sprint 7.)
- [x] **Decision gate: V4–V6 PROCEED.** V1 is at ~46 % useful-bytes / ~67 % moved-bytes of the single-core streaming roofline — nowhere near the 80–85 % stop threshold. Headroom is attributable to (a) memory-level parallelism in the strided two-pass loop (V4/V5) and (b) the 2R+1W → 1R+1W traffic reduction (V6), whose payoff the M=1024/N=4096 falloff bounds directly.

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

- [x] **V4 — multi-accumulator reduction ILP:** 4 independent `FMLA` accumulator chains (`rms_norm_ssve_v4.S`). **RESULT: +27–32 % vs V0 at N ≥ 2048 (54–56 % of roofline)** — the single-accumulator chain WAS the bottleneck, which also explains V3's zero (unrolling without breaking the chain). Numerics: tree-order combine deviates up to 1.1e-5 from the reference's sequential sum; V4/V5 verified at documented `kTolReassoc = 2e-5` (not a silent gate-widening).
- [x] **V5 — load software-pipelining:** rotating A/B load groups issued one group ahead (`rms_norm_ssve_v5.S`). **RESULT: ≈0 vs V4 — as pre-registered:** OOO rename already hoists the loads once the chains are broken. Closes the "more outstanding loads" family; `PRFW` gets a reasoned skip (same lever, one step earlier).
- [x] **V6 — redefined by measurement: contiguity, not residency.** Diagnostic first (footprint sweep): pass-2 residency is ALREADY satisfied (reuse distance 64·N ≈ 128–256 KB = L2-resident by construction); throughput actually collapses when the total footprint crosses the 16 MB L2 at fixed N (36.5 → 18.3 → 11.2 GiB/s), and at equal 32 MB footprint the access density (64 B per 4·M-byte stride) decides (31 vs 18 GiB/s). The 2a N=4096 falloff = true-DRAM regime, not pass-2 re-reads — 2a's provisional interpretation corrected in the report. **V6 = 4-row-block contiguity grouping** (`rms_norm_ssve_v6.S`): 256 B-contiguous column touches, one accumulator per block (keeps V4's ILP, restores strict-kTol sequential summation), shared gamma broadcasts. **RESULT: +131 % vs V0 in the DRAM regime (11.1 → 25.6 GiB/s), 63 % of roofline (~95 % moved-bytes) at M=1024/N=2048, never worse than V4 → final incumbent.**
- [x] **Vectorized `inv_rms` across row blocks (V6b):** folded into V6 — the four blocks' `FRSQRTE`+NR sequences are batched per group (already elementwise-vectorized per block; nothing left on the critical path after V4/V6).
- [x] Update the ablation table + report; **final Sprint-2 RMSNorm conclusion** (report §Sprint 2b): best = V6 at 63 % useful-bytes / ~95 % moved-bytes of the single-core streaming roofline (43 % in the 64 MB true-DRAM regime); exhausted: arithmetic (V1/V2), branches (V3), load scheduling (V5), accumulator ILP (V4), per-block residency (measured as already present); deferred: deeper/parameterized group depth as a shape-specialized JIT emission decision (Sprint 4), ZA staging with a hard target of beating V6 (Sprint 3), threading toward the 89.8 GiB/s chip ceiling (Sprint 5).

### 2c. LayerNorm (Mariza)

- [x] Hand-written SSVE kernel **V0**: two-pass (mean + variance in pass 1, normalize-scale-shift in pass 2), reduction is SSVE not ZA. Use the **stable two-pass variance** (centered second pass), never `E[x²]−E[x]²` (decision B / context.md §8).
- [x] VLA (decision D): predicated tail for any N.
- [x] Guard with `cpu_supports_sme()`; Catch2 tests skip on CI, run on M4.
- [x] Verified vs reference (incl. stress inputs: large-magnitude / shifted values — the cancellation cases); GiB/s measured against **both ceilings from 2a**; report updated. **RESULT: 12–13 GiB/s (20–22 % of 59.5 GiB/s SSVE roofline); 2.4–13.4× over reference. Plateau explained by 3R+1W structural cost vs RMSNorm's 2R+1W.**
- [x] **Ablation study (mirroring RMSNorm, plus the LayerNorm-specific axis):**
  - [x] Replay of the arithmetic variants — **the verdicts do NOT all transfer:** V1 (FRSQRTE+NR for `inv_std`) ≈0 (same as RMSNorm), but **V2 (pre-computed 1/N) = +9–12 %** vs RMSNorm's ≈0 — LayerNorm has TWO per-block FDIVs (mean, variance), both at serialization points between its three shorter passes, so hoisting them pays here.
  - [x] Memory-behaviour levers from 2b — **transfer cleanly:** V4 multi-accumulator +26–28 % (two accumulated statistics don't blunt it); V5 load pipelining ≈0 vs V4 (OOO rename covers it).
  - [x] **LayerNorm V6 = 4-row-block contiguity (three passes kept):** ties V4 at cache-assisted shapes, **+71 % vs V0 in the true-DRAM regime** (8.4 → 14.4 GiB/s) → **LayerNorm incumbent**. True 3R→2R pass fusion needs more resident state than 32 Z registers hold — recorded as the quantified Sprint-3 ZA hypothesis (+33 % traffic reduction available on LayerNorm, none on RMSNorm). Bytes stated under both conventions in the report.
  - [x] **Welford vs two-pass — two-pass dominates BOTH axes:** Welford is 25–30 % slower than V6 *despite moving 25 % fewer bytes* (per-element `delta/count` recurrence serializes on SIMD, as pre-registered), AND ~100× less accurate on shifted data in FP32 (max err 1.2e-3 vs 1.2e-5 at shift=1e4; the centered two-pass squares small values, FP32 Welford rounds `mean` at input magnitude every element). Stress tests added for V4/V5/V6/Welford (gap-fill); accuracy table in the report.
- [x] **LayerNorm vs RMSNorm on the same harness, same shapes** (decision C): **RMSNorm is 1.8–2.3× faster (LN/RMS = 0.43–0.56)** — well beyond the proposal's "10–40 %". Attribution: 1.33× structural traffic floor (3R+1W vs 2R+1W) plus per-byte efficiency (mean-subtract, β, two serialization points per block). Moved-bytes: LN V6 ~55 % of roofline vs RMS V6 ~95 %.

### 2d. Sprint close-out (docs + commits — same discipline as Sprints 0/1)

- [x] **Sphinx report updated:** `docs/source/weeks/project_norm.rst` now opens with a **Sprint-2 summary + reading guide** (headline results for both norms, consolidated ablation table against the validated roofline, per-lever verdicts, the build-order note explaining why 2b's first ablation precedes 2a), followed by the detailed 2a/2b/2c sections — byte-counting convention, Welford-vs-two-pass verdict, and small-N/large-N regime split all present. Rebuilt locally (`build succeeded`, 1 pre-existing cosmetic asm-lexer warning); `docs.yml` stays green.
- [x] **Commit discipline maintained throughout, not at the end:** every variant/test/bench landed as a small atomic conventional commit (`perf(norm): add rms_norm_ssve_v4 …`, `test(norm): …`, `bench(norm): …`) on the feature branch; the V-by-V history *is* the ablation narrative in git form. **Remaining: open the PR into `main` and let CI go green before merge** — a two-person review step (Ketsia/Mariza), not an automated one.

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
- [x] Hand-written ZA kernel (`rms_norm_za.S`): SVL×SVL FP32 tile staging via `mova`
      (granularity derived from `cntw`/SVL, decision D — no literal 16), reduction **stays
      SSVE** (context.md §5 — sum-of-squares via `fmla` in Z-regs, never through ZA); ZA used
      purely for tile staging/residency.
- [x] `SMSTART`/`SMSTOP` manage **PSTATE.ZA as well as PSTATE.SM** (bare `smstart`/`smstop`);
      whole norm in one streaming region; every ZA slice written before read (no `zero {za}`),
      ZA disabled before return (no AAPCS64 lazy-save hazard).
- [x] Tile-level **pass fusion / residency**: pass 1 stages `x` into ZA while reducing;
      pass 2 reads `x` back from ZA → **1R+1W** vs V6's 2R+1W where the row fits (`N ≤ 4·SVL`);
      streaming fallback for wider rows.
- [x] Verified vs reference (same Catch2 helpers, stress inputs, mismatched leading dims, tile
      boundaries, row tails, fallback); `cpu_supports_sme()` skip-on-CI guard. 86 test cases.
- [x] Benchmarked on the M×N grid vs V6 (both 2a ceilings printed). **VERDICT: ZA loses
      decisively — 39–65 % slower than V6 in the fast path; the kernel's own streaming fallback
      beats its ZA path.** Attributed: the two `mova` ops/element to stage `x` through ZA cost
      more than the DRAM read they save (memory isn't the binding constraint at these shapes).
      A pre-registered, valid "ZA adds nothing here, and here is why" outcome. **V6 stays the
      frozen RMSNorm incumbent for the JIT (Sprint 4).**

**LayerNorm ZA variant (Mariza):**
- [x] Gate resolved by **building the prototype** rather than taking the reasoned skip: the
      RMSNorm `mova`-throughput bottleneck is a hypothesis until measured on LayerNorm's own
      structure, and LayerNorm's larger headroom (3R+1W vs RMSNorm's 2R+1W) made it worth the
      direct test.
- [x] Hand-written ZA kernel (`layer_norm_za.S`): full **3-pass ZA residency** — `x` staged in
      ZA once during the mean pass, reused from ZA for BOTH the variance pass and the normalize
      pass (3R+1W → 1R+1W, a 50% traffic cut vs RMSNorm's 33%), going further than the Sprint-2c
      "3R→2R" partial-fusion sketch for the most decisive test available. Reduction stays SSVE
      (context.md §5); `smstart`/`smstop` manage PSTATE.SM + PSTATE.ZA correctly; VLA (tile
      geometry from `cntw`, no literal 16); streaming three-pass fallback for `N > 4*SVL`.
- [x] Verified vs `layer_norm_ref` (tile boundaries, row tails, mismatched leading dims, the
      fallback path, large-magnitude stress); 5 new test cases in `test_norm`, all pass on M4.
- [x] Benchmarked vs V6 on the same M×N grid as RMSNorm's ZA table. **VERDICT: ZA loses
      decisively — 51–68% slower than V6 in the fast path, a LARGER loss than RMSNorm's
      39–65%, exactly as the pre-registered hypothesis predicted** (proportionally more `mova`
      ops — 3/element vs RMSNorm's 2 — for a proportionally bigger traffic cut still nets a
      worse margin). Fallback (no ZA) is 11–12% slower than V6, expected since it's a plain
      single-block three-pass without V6's contiguity grouping/ILP. **V6 stays the frozen
      LayerNorm incumbent for the JIT (Sprint 4), for both norms.**

**Correctness fix found during this work (both kernels):** an eps register-stash bug affecting
13 of the 15 existing SSVE/ZA kernels (RMSNorm V0–V6 + `rms_norm_za`, LayerNorm V2/V4/V5/V6/
Welford) — eps was silently lost across `smstart` (a stack-reload aliasing bug in most, a false
assumption that D8 survives `smstart` in RMSNorm V0), invisible under every existing tolerance
but producing NaN whenever sumsq/variance is exactly zero. Fixed in all 13 kernels (dedicated
stack slot, stored before `smstart`, reloaded after — memory survives the PSTATE.SM transition,
no register does); 2 new regression tests added; full suite re-verified green (352,473
assertions, 93 cases).

**Sprint close-out (docs + commits):**
- [x] **Sphinx report updated:** Sprint-3 section — RMSNorm's original ZA writeup, PLUS the new
      LayerNorm ZA hypothesis/design/measured-verdict subsection, PLUS the eps register-stash
      correctness note. Heading underlines checked; docs build previously verified working on
      this file's structure (Sphinx itself unavailable in this environment to rebuild).
- [x] **Commit discipline throughout:** atomic conventional commits per file on
      `feat/sprint3-za-layernorm` (`fix(norm): …` for the eps bugfix, `feat(norm): add
      layer_norm_za …`, `test(norm): …`, `bench(norm): …`, `docs(norm): …`). **Remaining: open
      the PR into `main`, CI green before merge** (two-person review — Ketsia/Mariza).

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

- [x] Add `mini_jit::Norm` (`include/norm/jit_norm.hpp`, `src/norm/jit_norm.cpp` — APFS is case-insensitive, `Norm.hpp` would collide with `norm.hpp`): `generate(ntype)` emits via `InstGen` into a week5 `JitEngine` buffer; typed `get_rms_kernel()`/`get_layer_kernel()` return the function pointer (decision F). Parameterized on norm type; **ZA path deliberately not emitted** — Sprint 3's measured verdict ("loses decisively at every shape, mova-bound") means it did not earn its row; the reasoned skip is recorded in the report.
- [x] **Emit the winners, encoding-diff first:** both V6 winners transcribed 1:1; the diff compares the generator's buffer word-by-word against the **linked** hand-written kernel read from its function address at runtime (never stale). **GREEN: 143/143 (RMS) and 194/194 (LayerNorm) words identical** — the JIT inherits the trust of kernels that already passed the full suite.
- [x] Missing `InstGen` encoders: **30 added** (STP/LDP, scalar FMOV/SCVTF/FDIV, predicated FMLA/FMUL/FADD/FSUB, FRSQRTE/FRSQRTS, LD1RW, LD1W/ST1W + MUL VL, WHILELO, SEL, ADDVL, INCW, DUP, CBZ, B.cond, SM-only SMSTART/SMSTOP), each unit-tested against toolchain golden words + the Arm ARM. (`FADDV`/ZA encoders not needed: V6 has no horizontal reduction instruction and ZA is skipped.) **Bonus: the golden words exposed two latent week6 encoder bugs** — `sve_ptrue_all(fp32)` emitted `PTRUE P.B` not `P.S` (masked, fixed), and `sme_smstart_sm()` actually encodes SMSTART SM+ZA (kept for Gemm, correctly-named SM-only forms added).
- [x] Verify + parity: JIT kernels green vs reference on the V6 shape set incl. stress and eps-regression (M4); **parity within noise at all shapes (max |Δ| 0.6%)** after fixing a harness bug the comparison itself exposed — `cpu_supports_sme()` did an uncached sysctl syscall per wrapped call (~1 µs), which had also inflated the Sprint-2a small-N t0 (~70% of "t0 ≈ 1.4 µs" was syscall; restated in the report).
- [x] **Shape-dependent emission choices:** collapsed to the measured verdict — at SVL=512 no shape selects ZA (Sprint 3), so the JIT emits SSVE V6 for all shapes; documented as the emission decision rather than dead configurability. Parameterized group depth stays deferred (Sprint 6 optimization backlog).
- [x] Whole norm inside one streaming region (single SM-only `SMSTART`/`SMSTOP` pair, word-identical to the hand-written kernels).
- [x] Tests: `[sprint4][encoders]`/`[encoding-diff]` host-portable (run on CI), `[sprint4][jit]` execution SME-guarded; emission cost measured: **~4.7 µs one-time per kernel** vs ~0.4–135 µs/call execution — amortized after a handful of calls.

**Sprint close-out (docs + commits):**
- [x] **Sphinx report updated:** Sprint-4 section — generator design, encoding-diff methodology, latent-bug findings, the harness syscall correction (incl. Sprint-2a t0 restatement), parity table, emission cost. (`sprint4_errors.md` holds the full error log.)
- [x] **Commit discipline throughout:** atomic conventional commits on `feat/sprint4-jit-norm` (`feat(jit): …`, `test(jit): …`, `fix(bench): …`, `fix(build): enable_testing order — ctest discovered no tests before`). **Remaining: open the PR into `main`, CI green before merge** (two-person review — Ketsia/Mariza).

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