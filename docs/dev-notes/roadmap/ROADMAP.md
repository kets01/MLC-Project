# ROADMAP.md — MLC-Norm

> The path from today's lab foundation to the full vision in `../tooling/context.md`.
> Golden rule: get a **correct, measured** norm kernel working end-to-end before optimizing or integrating it.
> Each sprint ends with something that BUILDS, is VERIFIED (vs the C++ reference), and is COMMITTED.
>
> `../tooling/context.md` = the destination (JIT-generated SME/SSVE norm primitives, integrated into TEIR, measured at the roofline).
> `../../../CLAUDE.md` = working conventions (git, commits, tests, correctness gates, ISA rules) and the tooling list.
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
Stated as reduction stages and input traversals, because "two-pass / single-pass" conflates the two: **LayerNorm** has **two reduction stages** (mean, then variance) plus output generation → **three input traversals**, 3R+1W; **RMSNorm** has **one reduction stage** (sum of squares; no mean, no β) plus output → **two traversals**, 2R+1W. The 1.33× traffic ratio is the structural source of RMSNorm's advantage, and it is why the 2R+1W model must not be applied to LayerNorm. Three distinct claims, kept distinct: *semantics* (one fewer reduction), *literature* (Zhang & Sennrich: comparable task performance, 7–64 % runtime reduction across their experiments), and *our measurement* (1.8–2.3× on our shapes). Kernel numerical agreement is **not** model accuracy. The two reduction structures are deliberately different code and a measured ablation variable.

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
- **Arm Architecture Reference Manual (SME/SSVE)** + the lab's `tnzr.org/compile` material — the authority on instruction encodings and ZA/streaming semantics. Verify against the M4's *detected* SME level: `sysctl hw.optional.arm.FEAT_SME2` reports **1** on the target, so SME2 is available and the V7 kernels use it (guarded by `cpu_supports_sme2()`). The same material's *Hello SME* microbenchmarks independently corroborate our SVL, FP-issue-rate and cache-transition measurements — see the Sprint-6 external-validation section.
- **skill-creator** (optional, early): write a small MLC-Norm conventions skill as a learning step — no rush.
> Learning note: a generator can emit correct instructions you don't yet understand. Always have the agent explain the encoding/why — comprehension over an opaque build.

---

## MVP definition (Sprints 0–2)

A **correct, VLA SSVE kernel** for both norms — verified against the C++ reference and measured in GiB/s — driven from a benchmark app, no JIT and no TEIR yet. The JIT + SME-tiling + TEIR integration from `../tooling/context.md` arrives in Sprint 3+.

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
  - [x] **Welford vs two-pass — two-pass wins on SPEED; the accuracy half was overstated (corrected in Sprint 6 §1.3).** Welford is 25–30 % slower than V6 *despite moving 25 % fewer bytes* (per-element `delta/count` recurrence serializes on SIMD, as pre-registered) — **this is what decides the incumbent and it stands.** The original "~100× less accurate" claim does NOT: re-measured vs a float64 reference it holds at exactly one shift (1e4, and the factor is ~46×), is a wash at 0 and 1e2, and **reverses at 1e5** where Welford is 2.5× *better*. The stated mechanism was also wrong — Welford's *variance* is the more accurate of the two; the extra output error comes from the running **mean** updating at full input magnitude. Also newly documented: V6's 1.16e-5 accuracy floor is **FRSQRTE+NR, not the algorithm** (V0 with exact FSQRT reaches 3.33e-6 — a 3.5× accuracy cost that was undocumented). Stress tests added for V4/V5/V6/Welford; corrected accuracy table in the report.
- [x] **LayerNorm vs RMSNorm on the same harness, same shapes** (decision C): **RMSNorm is 1.8–2.3× faster (LN/RMS = 0.43–0.56)** — well beyond the proposal's "10–40 %". Attribution: 1.33× structural traffic floor (3R+1W vs 2R+1W) plus per-byte efficiency (mean-subtract, β, two serialization points per block). Moved-bytes: LN V6 ~55 % of roofline vs RMS V6 ~95 %.

### 2d. Sprint close-out (docs + commits — same discipline as Sprints 0/1)

- [x] **Sphinx report updated:** `docs/source/weeks/project_norm.rst` now opens with a **Sprint-2 summary + reading guide** (headline results for both norms, consolidated ablation table against the validated roofline, per-lever verdicts, the build-order note explaining why 2b's first ablation precedes 2a), followed by the detailed 2a/2b/2c sections — byte-counting convention, Welford-vs-two-pass verdict, and small-N/large-N regime split all present. Rebuilt locally (`build succeeded`, 1 pre-existing cosmetic asm-lexer warning); `docs.yml` stays green.
- [x] **Commit discipline maintained throughout, not at the end:** every variant/test/bench landed as a small atomic conventional commit (`perf(norm): add rms_norm_ssve_v4 …`, `test(norm): …`, `bench(norm): …`) on the feature branch; the V-by-V history *is* the ablation narrative in git form. **Remaining: open the PR into `main` and let CI go green before merge** — a two-person review step (Ketsia/Mariza), not an automated one.

**Done when:** both norms are correct and vectorized; every % -of-peak is stated against the **validated single-core roofline** (with the chip ceiling recorded for Sprint 5); the V4–V6 (and Welford) verdicts are in the ablation table with explanations; the Sprint-2 conclusion says explicitly which levers are exhausted and which are deliberately deferred to the ZA kernel (Sprint 3), the JIT (Sprint 4), and threading (Sprint 5); and the report section is merged with the sprint PR green.
**Tooling:** none (week6 `InstGen`/`Unary` as reference; `objdump` for spot-checking assembly).
**Learning focus:** SSVE reductions, predication/tails, VLA, roofline validation (peak of *what*), byte-counting honesty, memory-level parallelism vs arithmetic tuning, Welford-vs-two-pass on SIMD.

> 🎉 End of Sprint 2 = a working, correct, measured MVP norm kernel with a validated roofline and an exhausted hand-written **SSVE** optimization space. Good point to commit, refresh the report, and breathe. Everything below is enrichment toward the full `../tooling/context.md` vision — the ZA kernel (Sprint 3) now has a settled SSVE baseline to beat, and the JIT (Sprint 4) will have trusted hand-written counterparts for both architectures.

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
- [x] **DRAM-regime addendum (gap-fill, closes the ablation):** re-measured the ZA vs V6
      comparison at true-DRAM footprints (32–64 MB, past the 16 MB L2) in both the ZA fast
      path (N=64, huge M) and the streaming fallback — the regime where ZA's 1R+1W traffic
      saving *should* pay if it ever does. **It does not: ZA stays pinned at ~10 GiB/s
      regardless of footprint (mova-bound), while V6 delivers ~22–25 GiB/s → ZA still loses
      53–57 %.** V6's 4-row-block reuse distance keeps pass-2's re-read L1/L2-resident even at
      64 MB, so the DRAM read ZA "saves" was never a DRAM read. There is no footprint where
      residency staging wins. (This work also surfaced the d8–d15 ABI bug above — the fresh-build
      bench zeroed until all loop state was forced to memory; numbers taken with a robust
      standalone driver. Folded into the `main_norm` ZA table + report.)

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
Arm ARM for ZA load/store & tile-slice addressing (the M4 reports FEAT_SME2; Sprint 3 deliberately restricted itself to the SME1 subset, and Sprint 6 later rebuilt these kernels on SME2 multi-vector MOVA);
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
- [x] **Sphinx report updated:** Sprint-4 section — generator design, encoding-diff methodology, latent-bug findings, the harness syscall correction (incl. Sprint-2a t0 restatement), parity table, emission cost. (`../sprint-errors/sprint4_errors.md` holds the full error log.)
- [x] **Commit discipline throughout:** atomic conventional commits on `feat/sprint4-jit-norm` (`feat(jit): …`, `test(jit): …`, `fix(bench): …`, `fix(build): enable_testing order — ctest discovered no tests before`). **Remaining: open the PR into `main`, CI green before merge** (two-person review — Ketsia/Mariza).

**Done when:** both norm kernels are JIT-generated on the path(s) that earned it, verified, at least matching their hand-written counterparts' GiB/s, and the report section is merged with the sprint PR green.
**Tooling:** week5 JIT engine + week6 `InstGen`/`Unary` as the template; `objdump`/`llvm-mc` for the encoding diff; Arm ARM for the new encoders.
**Learning focus:** dynamic instruction emission, encoding verification, streaming-region management, emission-time specialization.

---

## Sprint 5 — TEIR integration (norm as a compiler primitive)

**Goal:** let the week7 TEIR runtime place the norm in a loop nest and invoke the JIT kernel — the `../tooling/context.md` flow, end to end.

- [x] **PREREQUISITE — fix the d8–d15 AAPCS64 violation in the frozen V6 kernels before TEIR calls them.** DONE: frames grown (RMS 80→128 B, LN 80→144 B) to save/restore all of `d8–d15`, mirrored in `mini_jit::Norm`, encoding-diff re-run green (157 / 208 words bit-identical). Two regression tests pin `d9–d15` via local register variables + an `asm` barrier — the only construct that observes the physical registers rather than a compiler-tracked copy. This is the **known** SMSTART/D-register hazard (first hit in Sprint 2, logged as `../sprint-errors/sprint4_errors.md` #6 and #8), escalated here from a caller-side `volatile` workaround to a *required fix*: TEIR's runtime calls the kernel from a loop nest that will **not** have that workaround, so the latent bug becomes a live one. AAPCS64 requires a callee to preserve the low 64 bits of `v8–v15` (`d8–d15`). Entering/leaving streaming mode (`smstart`/`smstop`) zeroes **all** of `d8–d15` (they alias the low 128 bits of `z8–z15`, which streaming-mode transitions clear), and `layer_norm_ssve_v6.S` additionally uses `z8–z15` directly as its mean/variance accumulators. Both `rms_norm_ssve_v6.S` and `layer_norm_ssve_v6.S` save/restore **only `d8`**, so a caller's `d9–d15` are silently clobbered across the call. This is invisible to the Catch2 tests (they hold nothing live in `d9–d15` across a single call) and was worked around caller-side in `main_norm`'s bench harness with `volatile`, but it **will corrupt TEIR's runtime state** when the loop nest calls the kernel without that workaround (it already zeroed the bench timing in a fresh Release build; see the Sprint-3 DRAM-regime addendum). **Fix:** save/restore `d8–d15` around the streaming region. This must be applied to **both** the hand-written `.S` *and* the `mini_jit::Norm` generator (Sprint 4 emits word-identical copies, so the JIT carries the same bug), then the Sprint-4 **encoding-diff must be re-run** so the two stay bit-identical. Verify with a caller that holds live FP values in `d9–d15` across the call.
- [x] Register the norm as a TEIR **primitive** (an `Invocation` the runtime resolves to the generated kernel), reconciling the canonical signature with TEIR's `CompiledKernel` convention. `CompiledKernel`/`RuntimeContext` widened to 4 pointers (LayerNorm needs β; the 3-pointer primitives ignore the 4th), and `Invocation` carries the per-invocation shape/ld/ε the norm ABI needs — the fixed-tile `(a,b,c,16,16)` dispatch could not express it.
- [x] Drive a multi-row tensor through TEIR `Iteration` → `Invocation` → generated kernel; honor strides and leading dimensions (decision F). **`load_teir()` is now a real parser** (`TeirParser`), not the filename-matching stub returning hardcoded trees: all five `.teir` files execute from disk, with layouts *derived* from each file's own strides. Needed a `Sequence` node (`contraction.teir` schedules two children) and `guard first(@axis)` support.
- [x] Verify the **TEIR-invoked** result against the C++ reference for the same shapes. `test_week7` rewritten: every parsed tree verified element-wise against a faithful-stride reference on **non-uniform** data (the old all-ones/two-element checks could not detect addressing bugs — which is exactly how the week-6 GEMM defect below stayed hidden).
- [x] Use `is_parallel` on the outer (row) axis for OpenMP across tiles; the kernel stays single-tile. **RESULT (M4, Release, 256 MiB working set, verified correct at every point): RMSNorm 20.28 → 42.80 GiB/s (2.11×, 49.8 % of the 85.8 GiB/s chip ceiling); LayerNorm 13.26 → 25.80 GiB/s (1.94×, 30.0 %).** More headroom than Sprint 2a's "≤1.5×" prediction, whose premise (single core at ~66 % of chip) held only at cache-assisted shapes — in the true-DRAM regime one core reaches 24 %.
- [x] Update the report with the integrated path and its GiB/s (`docs/source/weeks/norm_sprint5.rst`).

**Defects this integration exposed (all pre-existing, all invisible to the tests that should have caught them):**
- [x] **week-6 `Gemm` was wrong for non-uniform data.** `trans_b=0` advanced B by one element per k-step and loaded 16 contiguous floats — a sliding window over B's memory, not a row of B (FMOPA needs B's N-direction; column-major B requires a transpose it never did). It also computed `C = A*B` though documented as `C +=`, and used `W12` as a MOVA slice index while `X12` held a stride (`MOVZ W12` zeroes X12's upper half → every macro-tile after the first ran with a corrupted stride). Invisible to the existing test, which multiplies **all-ones** matrices and checks `MC[0] == K`. Redesigned: k-chunked, C loaded into ZA first, ZA-staged transposes for non-contiguous operands, all 8 `trans` combinations, loud failure on bad shapes. **Arbitrary K** now supported per the course spec (full chunks + predicated remainder). Verified against a scalar reference on distinct-valued data with non-zero initial C.
- [x] **week-6 `Unary::trans_b` was never implemented** (`[[maybe_unused]]`), though the course `Unary.h` has always specified it as the row-major-B — i.e. transposing — copy. `transposition.teir` needs exactly that; the old runtime served it with a contiguous copy that passed an all-fives, two-element check. Implemented as a ZA-staged transpose for `identity`/`relu`.
- [x] **The benchmark was measuring an unoptimized build.** `CMAKE_BUILD_TYPE` was empty → C++ probes/references at `-O0` while `.S` kernels were unaffected; single-core NEON read 10.8 GiB/s against SSVE's 59.4, though NEON is the faster of the two on this chip. Also fixed two real bugs in the chip-wide probe itself (equal static slices across heterogeneous P/E cores; then a same-chunk race in the first fix). Rebuilt Release, the three ceilings reproduce Sprint 2a and are mutually consistent: chip 85.79 > 1-core NEON 79.75 > 1-core SSVE 59.49.
- [x] **`test_week3` hung on SME hardware** (pre-existing week-3 code). Its "Identity Transpose Test" calls `identity_16_16_asm(..., trans_b=1)`, which used a **scatter store with vector offsets inside a streaming region** — unavailable without `FEAT_SME_FA64`, which Apple has not implemented (the constraint our own `rms_norm_ssve.S` documents). CI never saw it: no SME → the case is skipped. FIXED with the same ZA-staged transpose used in `Unary`'s `trans_b=1` path (A's columns into ZA horizontal slices, vertical slices back out = rows of A), verified against `identity_16_16_cpp`. The section now completes in well under a second instead of spinning indefinitely. This is the course's own Task 1 "permutation" kernel, and the SME chapter teaches ZA tiles/slices, so the fix stays inside the assignment's material.
- [x] **`test_week7` was never registered with CTest** — every other week calls `add_test(...)`, week7 did not, so the TEIR tests never ran under `ctest` or in CI. Registered as `Week7Tests`.
- [x] **A full `ctest` now completes on the M4 for the first time: 7/7 suites green** (it previously never finished, which is why both earlier full-suite attempts had to be killed).

**Done when:** a model-style call reaches the norm kernel *through TEIR* and is verified correct, with threaded scaling measured against the chip ceiling.
**Tooling:** week7 TEIR runtime as the template.
**Learning focus:** primitive registration, loop-nest composition, stride/layout correctness, bandwidth scaling across cores.

---

## Sprint 6 — Optimization, roofline & the ablation study

**Goal:** the proposal's evaluation deliverable — close on the roofline and attribute every gain.

- [x] **PREREQUISITE the baseline exposed — the Sprint-5 ABI fix was only half-applied.** The first baseline run printed **65 rows of `0.00 GiB/s`** (the known `d8–d15` clobber, `../sprint-errors/sprint4_errors.md` #6/#8). Sprint 5 fixed only the two V6 winners — the ones TEIR calls — so a hardware probe found **15 of 17 entry points still destroying the caller's `d9–d15`**, including `bw_probe_ssve`, the probe every "% of peak" is divided by. `main_norm`'s caller-side `volatile` is a workaround, and it had stopped holding. Fixed across all 15 kernels + the probe; **65 zero rows → 0**, every table reproducing its documented values. Two lessons recorded: the scripted fix introduced a `d9`/eps stack-slot collision that **every functional test passed over** and only the register probe caught; and the old ABI test covered just the two V6 kernels, which is exactly how the other 15 drifted — there is now one pinning all 17 (`[sprint6][abi]`).
- [x] Measure-driven optimization (decision E/F): the harness identified the real bottleneck, and the tuning that followed was **V7 = V6 + SME2 multi-vector accesses** (below). Levers left alone with reasons: ZA tiling (Sprint 3 measured it `mova`-bound at every footprint), streaming transitions (already one region per call), load scheduling (V5 ≈0).
- [x] **Ablation study**: one consolidated table in `main_norm` — scalar reference → V0 → V1/V2/V3 → V4/V5 → V6 → V7 → ZA → JIT, both norms, three regimes (per-call-overhead / cache-assisted / true-DRAM), same harness and same shapes throughout, with **LayerNorm two-pass vs Welford vs RMSNorm single-pass** side by side (decision C). Each row prints its measured max deviation from the reference, so accuracy is shown rather than asserted. The harness also caught itself: 20 reps made the small shape report hand-written V6 10 % below its own word-identical JIT twin — reps now scale with working set, after which they agree exactly.
- [x] Report GiB/s **vs both validated ceilings** (single-core SSVE and chip-wide) across shapes; byte convention (useful 1R+1W) restated in the section header.
- [x] **THE CEILING IS A CURVE, NOT A CONSTANT — every "% of peak" in the report since Sprint 2a was
      computed against the wrong denominator for cache-resident shapes.** 2a measured the streaming ceiling
      at a 256 MiB working set (59.5 GiB/s) and then applied that one number to every shape. It is the
      *DRAM* ceiling: the same probe reaches **115.6 GiB/s at 16 MiB** and ~117 across 512 KiB–16 MiB.
      Dividing a cache-resident kernel by the DRAM ceiling credits it for a constraint it never met and
      inflated those figures by ~2×. **The most-quoted number in the report — RMSNorm V6 at "~95 % of the
      moved-bytes roofline" — is really 50 %.** This is the same class of error 2a itself named ("peak of
      *what*"): 2a fixed the execution-mode half (NEON vs streaming) and left the footprint half unasked.
      FIXED: `main_norm` now sweeps the ceiling across footprints (64 KiB → 256 MiB, both modes) and every
      table divides by the ceiling at its own working set, so the numbers regenerate from the tree instead
      of depending on a constant that suits one regime. **All five report sections re-baselined**
      (Sprints 0/1, 2-RMS, 2-LN, 4, 6; Sprint 5 needed none — it already ran numerator and denominator both
      at 256 MiB). No kernel changed and no kernel-to-kernel ratio changed; what changed is the claim that
      the SSVE kernels were near saturation. Two further readings fell out of the sweep: streaming mode
      costs ~25 % vs NEON at DRAM but is nearly free in cache (0.93×), so "streaming is a structural
      handicap" is a DRAM-regime statement; and below ~512 KiB the SSVE figure stops being a bandwidth
      number at all (NEON climbs to 292 GiB/s at 64 KiB while SSVE falls to 105 — that is `SMSTART` cost
      becoming a visible share of a sub-µs pass, which is Sprint 7b's subject).
- [x] **Sprint 1's numbers were additionally `-O0`.** Its "roofline target" of 10.62 GiB/s was a debug-build
      probe (the Sprint-5 `CMAKE_BUILD_TYPE` defect), so its 6–29 % figures divided one unoptimized number
      by another. Re-measured in Release against footprint-matched ceilings: the scalar reference sits at
      **0.8–6.3 %**. Its "RMSNorm is consistently 1.3–1.6× faster" also does not survive — the Release
      ratio falls from 1.6× at M=128,N=64 to 1.08× at M=1024,N=2048.
- [x] Verify every optimized configuration still matches the reference (decision B) — verification runs *before* each row's GiB/s is reported, and the JIT/TEIR paths are verified on top of that.
- [x] **SME2 exploration — DONE, and the pre-registered expectation was WRONG (the sprint's headline result).**
      `FEAT_SME2` confirmed by `sysctl`, and the multi-vector forms assembled **and executed** in a
      streaming region on the M4 before any kernel was built on them. The lever taken was the one this
      bullet called "the more promising angle": V6's group loop already touches four consecutive VL-row
      blocks per column, which is exactly SME2's 4-vector operand shape, so **V7 = V6 with 4 accesses
      folded into 1** — same addresses, same traffic, same arithmetic, so the correctness gate is
      *bit-identity* with V6, not a tolerance. **RESULT: +17.2 % for RMSNorm in the true-DRAM regime**
      (order-alternating A/B, 24 samples each way), ≈0 cache-resident — against a pre-registration of
      "cannot help, SME2 does not raise the DRAM roofline". **LayerNorm V7 was then built as the
      discriminating experiment**, because two mechanisms explain the RMSNorm win and predict opposite
      LayerNorm outcomes: "fewer instructions retire faster" predicts a *bigger* gain (LayerNorm folds
      more accesses per element — three passes, not two), while "relieved memory-level parallelism"
      predicts a *smaller* one (LayerNorm does more FP work per byte and is less memory-starved).
      **Measured +2.7 % — the instruction-count explanation is falsified, MLP survives.** The threading
      data agrees: +17 % at one thread, +2.6 % at 16, where the memory system is already saturated.
      This refines Sprint 2b rather than contradicting it — V6 made the *addresses* contiguous, V7 makes
      the *request* singular; the roofline is unchanged, what changed is how much of it one core reaches.
      Guarded behind `cpu_supports_sme2()`, falling back to V6 on M1/M2.
- [x] **V7 promoted into the JIT and TEIR (the first feature-dependent emission decision).** Sprint 4
      left emission choices collapsed because no shape selected ZA; SME2 provides one that pays.
      `mini_jit::Norm::generate()` now takes an `isa_t` (`automatic` = V7 where `FEAT_SME2` is present,
      else V6). Three new `InstGen` encoders (`PTRUE PNn.S`, 4-vector `LD1W`/`ST1W`), field layouts
      derived from 13 toolchain golden words varying register/base/predicate independently and pinned by
      unit test. **Encoding diff green for both SME2 kernels**; emitted size 157→110 (RMS) and 208→144
      (LN) words — the folded accesses plus the predicated tail path the counter predicate removes. TEIR inherits it via the default: **single-thread RMSNorm
      21.04 → 24.68 GiB/s (+17.3 %)**, every configuration verified.
- [x] **SME2 multi-vector on the ZA path — the skip was RETRACTED and the experiment run.** The skip
      rested on an untested factor: `mova` had never been characterised as *issue*-bound vs *ZA-port*-bound,
      and the "2x" was assumed. Measured directly (no memory traffic, 16 vectors either way): single-vector
      **3.88 G vec/s**, 4-vector **15.50 G vec/s** = **4.00x** — issue-bound, and the fold is 4:1, not 2:1,
      so the skip's own arithmetic did not hold. Both kernels built (`rms_norm_za_sme2.S`,
      `layer_norm_za_sme2.S`), bit-identical to the Sprint-3 versions on every shape tested:
      **RMSNorm 10.07 → 16.35 (+62 %), LayerNorm 4.83 → 10.34 (+114 %)**. **Verdict unchanged — ZA still
      loses (−37 % / −24 % vs V7) — but now by measurement rather than extrapolation**, and the residual
      cause is different from the original diagnosis: the ZA kernels touch **one** SVL-row block per
      iteration (64 B per column) against V6/V7's **four** (256 B), so what binds them is the Sprint-2b
      access-density lever, not `mova`.

  <details><summary>Original pre-registration (kept verbatim — it is the thing the measurement overturned)</summary>

- [x] **SME2 exploration — SUPERSEDED pre-registration, kept verbatim for the record.** Hardware
      correction: `sysctl` on the target M4 reports `hw.optional.arm.FEAT_SME2: 1` — the machine
      supports **SME2**, not only SME1 as the docs conservatively assumed. SME2 does **not** raise
      the DRAM roofline, so it cannot help the bandwidth-bound large-N shapes; the one lever worth
      testing is **multi-vector `mova`/`ld1`/`st1` (2- and 4-vector forms)**, which could lift the
      ZA path's ~10 GiB/s `mova`-throughput ceiling *if* that ceiling is instruction-issue-bound
      rather than ZA-port-bound. Pre-registered expectation: even a 2× `mova` speedup only brings
      ZA to ~20 GiB/s — a tie-at-best with V6, and only in the N≤64 window that is cache-resident
      anyway, so it does not create a win; multi-vector SSVE `ld1w`/`st1w`/`fmla` on the V6
      streaming path is the more promising angle but V6 is already ~95 % of moved-bytes single-core.
      Measure it, keep each result (positive or explained-negative) as an ablation row, and guard
      the SME2 path behind a `FEAT_SME2` runtime check so it degrades to the SME1 kernel on M1/M2.

  </details>

**Done when:** an ablation table shows each optimization's contribution and how close to peak the best kernel gets — at both the single-core and the threaded level.
**Tooling:** none new (Arm ARM for the SME2 multi-vector encodings; verify against `FEAT_SME2`).
**Learning focus:** roofline analysis, ablation methodology, attributing speed-ups.

---

## Sprint 7 — Numerical-stability & streaming-overhead depth (honest engineering)

**Goal:** the interview/report talking point — surface and explain the real tradeoffs (context.md §8).

- [x] Characterize **numerical stability**. The project had never *implemented* the dangerous formulation, so the claim had no counterexample; `src/norm/stability.cpp` adds naive `E[x²]−mean²`, centred two-pass and Welford, all accumulating in **FP32** (the kernels' precision) against a float64 oracle. Stress axis = a shift, which leaves variance untouched but scales κ = 1 + mean²/var by ~s². **RESULTS: naive's error tracks κ** (each ×10 shift → ×~100 error, the classical result reproduced on our hardware) and **from shift 1e5 it returns a NEGATIVE variance → `sqrt` = NaN** — a qualitative failure, not a gradual one, which is why decision B counts stability as correctness. **Two-pass is flat across 12 orders of magnitude of κ** — that immunity is exactly what LayerNorm's 3R+1W buys. Welford sits between, its weakness being the running mean (per Sprint 6 §1.3), not the variance recurrence. Summation order is trustworthy because the build sets no `-ffast-math`, so IEEE forbids reassociating the reductions.
- [x] **The shipped kernels placed on the same axis** (scalar estimators prove nothing about SSVE kernels). Three findings: (a) V6's FRSQRTE accuracy cost vs V0's exact FSQRT is **real but narrow** — 24× at shift 0, *indistinguishable* from shift 1e2 up; (b) what limits LayerNorm on hard data is **not the variance at all** but the FP32 representation of `(x−μ)` in the output — predicted ULP 1.6e-2 vs measured 1.2e-2 at shift 1e5, and V0/V6 coinciding there is the evidence; (c) **RMSNorm is flat at ~5e-6 across the whole sweep, ~2300× better than LayerNorm at shift 1e5** — structural, since it never forms a mean, and the same property that makes it faster. Accuracy and throughput agree, which is unusual and worth saying. RMSNorm's own limit measured too: `Σx²` overflows FP32 at `|x| ~ sqrt(3.4e38/N)`, with a test straddling the boundary at ±10 %.
- [x] Characterize **streaming-mode overhead — and the headline is that it is NOT the small-tensor cost.** Every per-call figure so far was *inferred* from a fit intercept, an instrument that already failed once (Sprint 2a read t0 ≈ 1 µs; Sprint 4 found ~70 % of it was a sysctl). `smstart_probe.S` measures it directly against a matched transition-free control: **one `smstart`/`smstop` round trip = 9.07 ns**, and **SM+ZA ≡ SM-only**, so managing PSTATE.ZA is free (the ZA kernels' problem was never the transition). The **per-call floor is 46–53 ns measured** (batched, because one call is at clock resolution — the first attempt printed 0.0 ns), of which the transition is **~20 %**. A linear fit is deliberately not used: over these sizes it returns a *negative* intercept, the model announcing it does not apply.
- [x] **Report *when* the SME kernel wins — crossover at M ≈ 16 rows (N=512), and the cause is group granularity, not `SMSTART`.** V7's time is flat from M=1 to M=64 because that is **one group**: V6/V7 process 4 VL-row blocks at a time (VL=16 at SVL=512 → 64 rows, queried via a new `RDSVL`-based `svl_fp32_lanes()` rather than written as a literal, decision D), and a partial group costs a full one since unused lanes are predicated off, not skipped. At M=1 the kernel does 64 rows' work for 1 row of result; at M=256 the time is exactly 4× that of M=64. A 9 ns transition cannot explain a 6 µs plateau; the group explains it exactly. **Two consequences:** the fix is a narrower group for small M — the shape-specialized JIT emission decision deferred since Sprint 2b — and it is the *same* granularity behind Sprint 6's threaded chunk-alignment sensitivity (+42 % at 6 threads).
- [x] What the "one streaming region per call" decision (context.md §8) is worth, **quantified**: the alternative costs one transition per row = 1.16 µs at M=128, ~22× the entire per-call floor. The decision is right, now by a measured margin — but the constant is 9 ns, so *"streaming mode is expensive" is true per ROW and false per CALL*, and only the second is what these kernels do.
- [x] Document both as deliberate tradeoffs (not accidents) in the report — `docs/source/weeks/norm_sprint7.rst`, listed in the toctree. 9 new tests: the 7a stability cases are **host-portable and run on CI**, the 7b probe cases are SME-guarded, and the probes are pinned against the `d9–d15` clobber (a new SME entry point is a latent instance of the bug this project has hit five times). Suite green on M4: 133 cases, 749 715 assertions.
- [ ] **Left open, stated rather than glossed:** the 37–44 ns remainder of the per-call floor is attributed to prologue/epilogue + pointer setup + serialization as a group, not itemised component by component; and the narrow-group kernel for small M is identified but not built (it is a JIT emission change, and this sprint's scope was characterization).

**Done when:** the report explains the accuracy/throughput and small/large-tensor tradeoffs with data.
**Tooling:** none new.
**Learning focus:** FP numerics, fixed-cost amortization — strong defense talking points.

---

## Sprint 7.5 — External baselines (how do we compare to state of the art?)

**Goal:** place our kernels against two real, widely-used implementations, on the same machine, with a harness fair enough that the number means something.

> **Read `../sprint-errors/sprint6_errors.md` §2.7–2.8 first.** This project already attempted a vendor comparison once and it produced four wrong claims: an "Accelerate RMSNorm" that was our own `vDSP_svesq`+`vDSP_vsmul` composition (Accelerate has no RMSNorm), a "`vDSP_normalize` = LayerNorm" equivalence that is false (no eps, no γ, no β), and a headline "16–37× faster" that was **measuring a layout mismatch, not a kernel** — vDSP forced to walk our column-major matrix at `stride = ld`, one cache miss per element. Given its own contiguous layout, **vDSP beat us: 65.77 vs 38.47 GiB/s at 16 MiB.** That is the precedent, and the reason for every rule below.

**Chosen baselines: ExecuTorch + PyTorch ATen.** ExecuTorch is Meta's on-device runtime, which is the deployment story this project is about; PyTorch ATen is the most-cited CPU baseline and its LayerNorm uses Welford, tying directly into the Sprint 2c/6/7a Welford analysis.

**Harness rules (each one exists because of a specific past failure):**
- [ ] **Native layout for everyone.** Our kernels are column-major over a strided feature axis; every library normalizes the last, contiguous axis of a row-major tensor. Both are efficient *in their own layout* (V6's grouping makes each column touch 256 contiguous bytes). Forcing either into the other's layout measures the mismatch, not the kernel — the exact Sprint-6 error. Report the layout as a stated variable.
- [ ] **Correctness gates the comparison (decision B applies to baselines too).** Verify each library's output against our float64 reference *before* quoting its throughput. This is what catches semantic mismatches, which is where the last attempt went wrong: PyTorch LayerNorm uses biased variance; ExecuTorch may **decompose** RMSNorm into elementwise ops rather than running a fused kernel.
- [ ] **Say what is fused and what is not.** Comparing our fused kernel against an unfused op sequence and calling the difference "we are faster" is the §2.8 trap ("fused beats unfused" is not a kernel-quality claim).
- [ ] **One thread everywhere**, same shapes, same byte convention (useful 1R+1W), footprint-matched ceilings from Sprint 6 §1.2.
- [ ] **Record versions.** ExecuTorch 0.3.0 + torch 2.4.0 (isolated venv — the only combination available for this machine's Python 3.9; upstream is 0.5+, so the ExecuTorch figure is dated and must be labelled so). PyTorch eager/compile measured separately on 2.8.0 in `mlc_env`. **Do not install ExecuTorch into `mlc_env`** — it holds the Sphinx toolchain the docs workflow depends on.

**Pre-registered expectations (written BEFORE measuring; the practice `../sprint-errors/sprint6_errors.md` credits for every claim that survived) — and what the measurements said:**
- [x] **P1 — PyTorch eager slow, we win ~5–10×, from dispatch overhead and lack of fusion.** **PARTLY WRONG.** RMSNorm **4.2–5.8×** (just under the range), LayerNorm only **1.5–2.1×** (well under). The stated *reason* is also wrong for LayerNorm, which dispatches to a single fused, well-optimized `aten::native_layer_norm`. The attribution splits by norm, not by framework.
- [x] **P2 — ExecuTorch portable slower than PyTorch eager unless XNNPACK is active.** **CONFIRMED but narrowly**, and much closer than on the old stack: LayerNorm 5.92 vs 7.35; RMSNorm portable actually *wins* (3.26 vs 2.77), because eager RMSNorm decomposes.
- [x] **P3 — RMSNorm decomposes rather than running a fused kernel.** **CONFIRMED, and in PyTorch too — which was not predicted.** Profiler on CPU: `LayerNorm` → one fused `aten::native_layer_norm`; `RMSNorm` → `mul`, `pow`, `sum`, `_fused_rms_norm`, `div_`, with self time dominated by the elementwise ops. `_fused_rms_norm` exists at the dispatcher level but **CPU eager still decomposes**; `torch.compile` fuses it and roughly doubles throughput (2.77 → 5.90).
- [x] **P4 — we may lose.** **Not realized against these two** — see the honesty clause.

**RESULT — single-threaded, native layouts, current upstream (PyTorch 2.13.0, ExecuTorch 1.4.1 **including the XNNPACK delegate**), every output verified against our float64 reference first (max|diff| ≤ 3.1e-6 = same function):**

| shape | ours LN V7 | torch LN | ET LN (xnn) | ours RMS V7 | torch RMS (best) | ET RMS (xnn) |
|---|---|---|---|---|---|---|
| 1024×2048 (16 MiB) | **16.49** | 7.89 | 6.06 | **38.46** | 6.65 | 6.64 |
| 4096×8192 (256 MiB) | **13.54** | 7.35 | 5.93 | **24.65** | 5.90 | 4.19 |

- [x] **XNNPACK delegation was tested, not skipped.** The project's Python 3.9 env caps ExecuTorch at 0.3.0, whose XNNPACK partitioner cannot even import (`TypeError: 'staticmethod' object is not callable` — staticmethod objects only became callable in 3.10). Rather than report only the portable reference kernels — which would be the mirror image of the Sprint-6 error, handicapping the baseline and calling the gap a win — a **standalone Python 3.12 was installed** (via `uv`, isolated, no sudo, `mlc_env` untouched) so current ExecuTorch 1.4.1 and its deployed delegate path could run **unpatched**.
- [x] **Headline finding is an INVERSION, and it is about fusion rather than kernel quality.** Within our implementation RMSNorm is 1.8–2.3× faster than LayerNorm (decision C's premise: one pass, no mean, 2R+1W vs 3R+1W). Within PyTorch eager the ordering *flips* — its LayerNorm is 2.7× faster than its RMSNorm — purely because the cheaper operation is the one without a fused CPU kernel. **A fused implementation of the expensive norm beats a decomposed implementation of the cheap one**, which is the clearest argument in the report for why writing the kernel was worth doing.
- [x] **Baselines move between versions — margins are version statements, not facts.** Same harness, old stack (torch 2.4.0 / ET 0.3.0) vs current at 4096×8192: `torch.compile` RMSNorm 3.02 → **5.90**, ET portable RMSNorm 0.62 → **3.26**, ET portable LayerNorm 2.77 → **5.92**. Our RMSNorm margin would have read **8–13×** against the old stack and is **4.2–5.8×** against the current one. The kernels did not change; the baseline did. Every margin is quoted with versions attached.
- [x] **HONESTY CLAUSE — "faster than PyTorch/ExecuTorch" is NOT "state of the art", and the report says so.** (a) The fastest implementation ever measured on this machine is neither: Sprint 6's corrected vDSP figure is **65.77 GiB/s contiguous vs our 38.46 at 16 MiB — ~1.7× faster than us** (~57 % of the footprint-matched ceiling against our ~33 %). (b) These are general-purpose frameworks; one whose CPU RMSNorm decomposes in eager is a baseline for "what you get without a kernel", not a SOTA bar. (c) ExecuTorch times include per-invocation runtime dispatch — honest for on-device, but not a pure kernel number.
- [x] Report: `docs/source/weeks/norm_sprint7_5.rst`, in the toctree. Harness in `bench/baselines/` (PyTorch driver, ExecuTorch driver, C++ cross-verifier), reproducible from the tree.

**Still open:** a **vDSP re-measurement through this harness** — the 65.77 figure comes from the Sprint-6 error log, not from this correctness-gated harness, and since it is the only implementation known to beat us it is the most valuable baseline still missing; and a threaded comparison (everything here is single-threaded, while the frameworks parallelize by default).

**Done when:** both baselines are verified-correct against our reference, measured on the same machine and shapes with layouts stated, and the report carries the comparison — including whichever of P1–P4 the measurements refute. **DONE.**
**Tooling:** isolated venv for ExecuTorch; `mlc_env` for current PyTorch.
**Learning focus:** what "state of the art" means for a bandwidth-bound op, fair-comparison methodology, fused vs decomposed execution.

---

## Sprint 8 — Correctness, dispatch & provenance hardening

**Goal:** no new kernel and no new optimization — close the gap between "these numbers are true" and "these numbers are *demonstrably* true", and fix a real API defect found in review.

- [x] **FIXED A SILENT NO-OP IN THE PUBLIC API.** Every ISA-specific entry point began `if (!cpu_supports_sme()) return;`, so on a non-SME host `layer_norm_ssve(a,b,...); use(b);` produced **no computation, no status and no diagnostic** — the caller could not tell success from a no-op and would consume stale/uninitialized memory. The guard existed so CI tests could skip, which is a real need, but it solved a test-harness problem by degrading the library's contract, and it contradicted this project's own rule to fail fast and clearly (CLAUDE.md §4). **Two-layer API now:** public `layer_norm()`/`rms_norm()` dispatch `FEAT_SME2 → V7`, `FEAT_SME → V6`, else the **scalar reference**, and always compute a correct result on any CPU; the 25 named kernels keep a documented hard precondition and **abort with a diagnostic naming the function and the missing feature** rather than returning an uncomputed buffer. `norm_dispatch_target()` reports which path a host takes.
- [x] Dispatch tests are deliberately **not** SME-guarded — the fallback path only ever executes on a machine without SME, so guarding them would leave it untested on the one runner that exercises it. Includes a sentinel-based regression test for the original defect (fill output, call, require nothing survives).
- [x] **GATE BEFORE TIMING, EVERYWHERE.** The ablation previously verified each row but still printed GiB/s beside an `ok = NO`. Now: run the exact function about to be timed, on the exact shape, compare the full output to the float64 reference, and **refuse to produce a timing on disagreement**. Run reports **66 / 66 configurations verified before timing**.
- [x] **External baselines gated at every shape.** The drivers dumped one shape (128×64) while the table reported three — verifying one and publishing three is an inference, not a check. Both drivers now dump every benchmarked shape plus a manifest; the C++ checker walks it (**6/6 per baseline**, PyTorch 2.13.0 and ExecuTorch 1.4.1).
- [x] **Best-case is no longer the only statistic.** Min is kept and labelled best-case envelope, with **median and p10–p90** from the same samples beside it. Result: they agree to ~1 %, so the best-case figures reported throughout this project **were** typical — previously an assumption, now shown.
- [x] **Provenance printed with every run:** git SHA (**marked `-dirty` when the tree is modified**, since such a run is not reproducible from its SHA), build type, compiler, OS, CPU, `sysctl` FEAT_SME/FEAT_SME2, `cpu_supports_sme()/sme2()`, RDSVL streaming VL, dispatch target, thread count, QoS. The SME lines are **detected**, not quoted from a datasheet — these docs carried a stale "M4 is SME1" claim for several sprints while the hardware reported `FEAT_SME2 = 1`.
- [x] Report section `docs/source/weeks/norm_sprint8.rst`, in the toctree. Suite green on M4: **137 cases, 780 264 assertions**. No kernel changed, so no performance number moved — and the tables reproduce the Sprint 6/7 values.

**Done when:** the API cannot silently do nothing, every reported configuration is correctness-gated before timing, and every table carries the conditions that produced it. **DONE.**

---

## Sprint 9 — Reproducibility & polish

> **Scope decision: no further implementations will be tested.** The remaining
> work perfects what exists rather than adding baselines or kernels.

- [x] **Apple BNNS investigated and DROPPED as a baseline — recorded, not measured.** Full evidence in `../tooling/bnns_investigation.md` (kept out of the report as a developer note, per the `sprint*_errors.md` convention). What stands: BNNS reaches LayerNorm only through a **generic** `BNNSFilterCreateLayerNormalization(normType, …)` whose `normType` selects batch(2)/instance(3)/**layer(4)**/group(5) — a real mislabelling trap; and there is **no RMSNorm in BNNS on macOS 15.2** (every "RMS" symbol in the headers is `RMSProp`). Apple's *current* BNNSGraph builder does provide `layerNorm(axes:epsilon:)` and `rmsNorm(scale:epsilon:)`, but both are **absent from this SDK** — 0 occurrences in an `Accelerate.swiftinterface` carrying 644 BNNS and 117 BNNSGraph declarations — and post-date this OS, so a newer SDK would compile them but they would not run. We could create layer-normalization filters but not execute them (`BNNSFilterApply` → −1); the `M=1` minimal case fails identically at every axis, which **rules out tensor-dimension/axis interpretation as the cause**, and a ReLU control with the identical deprecation applies fine (rc=0), which rules out "deprecated hence stubbed". **No BNNS number is reported** — reporting one we could not obtain, or composing an RMSNorm and calling it a vendor kernel, would repeat the Sprint-6 §2.7 error.
- [x] **CMake presets added and verified:** `release` (→ `build/`), `debug` (→ `build-debug/`, Address + UB sanitizers), `release-submission` (→ `build-submission/`, its own directory so a submission run cannot pick up stale objects). All three configure; `ctest --preset release` discovers all 7 suites. The `cmake_minimum_required(3.10)` floor is unchanged for plain builds — presets need CMake ≥ 3.19 only to *use*.
- [x] **The whole norm suite runs clean under sanitizers: 780 264 assertions, 137 cases, ZERO ASan/UBSan findings** — including every SME-executing kernel. CLAUDE.md §7 has always called for sanitizer runs when a layout bug is suspected, but the project had never actually done a full one; it is now a one-line preset and the result is a real correctness datum, not just a convenience.
- [x] **Fixed two defects in the README while documenting the presets:** its build block was an unterminated code fence, and it instructed `cmake ..` with **no build type** — which is precisely the Sprint-5 defect (empty `CMAKE_BUILD_TYPE` compiles the C++ reference and probes at `-O0` while leaving the `.S` kernels optimized, making single-core NEON read 10.8 GiB/s against SSVE's 59.4 and invalidating a round of tables). The README documented the exact command that caused it. Now presets first, with the rule stated explicitly.
- [x] **Pinned the baseline environment and checked in the run manifests.** `bench/baselines/requirements-baselines.txt` records the exact versions behind the reported numbers (`torch==2.13.0`, `executorch==1.4.1`, `numpy==2.5.2`, Python ≥ 3.10 required for the XNNPACK partitioner); `bench/baselines/manifests/` holds the manifests the drivers wrote for those runs. The reproduction block in the report now installs from the pinned file instead of floating `executorch`. Rationale recorded in both places: the same harness on torch 2.4.0 / executorch 0.3.0 gave `torch.compile` RMSNorm 3.02 GiB/s against 5.90 here, which would have made our margin read 8–13× instead of 4.2–5.8× — the kernels did not change, the baseline did, so a margin is a statement about specific versions.
- [x] **Scoped the "native layout" question explicitly** in the report: it answers *what throughput can each implementation achieve given the layout it was designed for*, and **not** *what would it cost to substitute our kernel for a framework operator on the framework's own representation* — the second would have to include boundary conversion, which this comparison excludes from both sides. Added to the method section and to "what is left open".
- [x] **Withdrew the vDSP claim.** The Sprint 7.5 conclusion asserted that "a specialist vendor library still beats us", citing vDSP at 65.77 GiB/s. That figure came from the Sprint-6 error log, never went through the correctness-gated harness, and was never verified to compute the same function — and the composition it referred to is the very thing §2.7 identified as mislabelled. With vDSP withdrawn and BNNS unmeasured, **the report now makes no claim about how our kernels compare to a specialist vendor implementation, in either direction.** The §2.7–2.8 entries stay in `../sprint-errors/sprint6_errors.md`: that is the record of our own mistake.

## Sprint 10 — Claim corrections & external validation

- [x] **Traversal language corrected everywhere.** "Two-pass / single-pass" conflates *reduction stages* with *input traversals*; both are now stated explicitly — LayerNorm: 2 reduction stages + output → **3 traversals**, 3R+1W; RMSNorm: 1 reduction + output → **2 traversals**, 2R+1W. Updated in `../tooling/context.md` (decision C and §7), `project_norm.rst`, `norm_sprint0_1.rst` and this file. `main_norm`'s byte-convention header now gives the moved-bytes multiplier **per norm** (1.5× for RMSNorm, 2.0× for LayerNorm) — as written it quoted 2R+1W without naming a norm, which reads as if it covered both.
- [x] **RMSNorm claim split into three separate statements.** *Semantics* (one fewer reduction stage, one fewer traversal), *literature* (Zhang & Sennrich: comparable task performance, **7–64 %** runtime reduction across their experiments — not a single "10–40 %"), and *our measurement* (1.8–2.3× on the shapes evaluated). Also stated explicitly: **kernel numerical agreement is not model accuracy** — our tests show our RMSNorm matches an *RMSNorm reference*, which says nothing about substituting one norm for the other in a network.
- [x] **"Bandwidth-dominated for the measured shapes"** replaces the unqualified "memory-bandwidth bound", with the evidence: against footprint-matched ceilings the best kernel reaches ~33 % useful / ~50 % moved, and cache-resident RMSNorm V7 is **FP-issue-bound at 100 %** of the measured SSVE issue rate — which is exactly why SME2 buys it ~0 % there and +17 % at DRAM. `../tooling/context.md` §6 now carries an **operational-intensity table** (~0.25 flop/B for RMSNorm, ~0.31 for LayerNorm) and the ~31 flop/B that reaching FMOPA peak would require.
- [x] **Reconciled +17.2 % vs +21.5 %.** They are different measurement sets: the A/B study (20.96 → 24.56, order-alternating, 24 samples each way) and the consolidated ablation (20.29 → 24.65, each rung measured once in sequence). **The A/B study is authoritative** — alternating order within one run is what controls for drift in a paired comparison, which the sequential table does not do. Recorded in the Sprint-6 section so the two numbers cannot read as a contradiction.
- [x] **Stale SME1 assumption corrected** in `../tooling/context.md` (three sites), `../../../CLAUDE.md` and this file — replaced with *detect, don't assume, in either direction*, citing `hw.optional.arm.FEAT_SME2: 1` on the target and the fact that these docs asserted SME1 for several sprints while the hardware said otherwise. **`d8–d15` narrative anchored**: Sprint 5 is now marked as the single canonical account, with the ABI rule stated once there (AAPCS64 makes the low 64 bits of `v8–v15` callee-saved, so clobbering them is nonconforming even when a standalone benchmark appears to work).
- [x] **External validation added** (`norm_sprint6.rst`) against the Jena *Hello SME* M4 microbenchmarks: 512-bit SVL matches; **their SSVE FMLA 31 GFLOP/s against our independently measured 31.0 GFLOPS** — two harnesses, two purposes, same number; their sharp bandwidth reduction above ~8 MiB **independently corroborates the footprint correction**, turning "our probe produced these numbers" into "our numbers reproduce an independently observed architectural feature"; and their SME2 four-register `LD1W` at ~925 vs ~376 GiB/s **independently motivates V7** as a lever aimed at a documented feature rather than ISA trickery. Stated explicitly that absolute figures should *not* be expected to match (different instruction streams and traffic definitions — ours is 1R+1W scale-add, theirs read-only loads), and that their multicore finding (one P-core largely saturating the cluster's SME resource) is why our sub-linear thread scaling is expected rather than suspicious.
- [ ] **Rewrite the Sprint 7.5 "honest limit" section.** vDSP is out as a live claim (the 65.77 GiB/s figure came from the Sprint-6 error log, never through the correctness-gated harness), and BNNS produced no number. So the section can no longer say "a specialist vendor library still beats us" — that evidence is withdrawn. The replacement claim is narrower and fully supported: *we are faster than two general-purpose frameworks at current versions, verified to compute the same function; a specialist vendor kernel was **not measured**, and Apple's own RMSNorm API does not exist on this OS.* Keep the §2.7–2.8 entries in `../sprint-errors/sprint6_errors.md` intact — that is the record of our own mistake, and deleting it would undercut the point of having an error log.

## Sprint 11 — Report & ship

- [x] **Report restructured as synthesis, not chronology.** `project_norm.rst` is now: research question → system/hardware → algorithmic choices → performance methodology → variants/ablation → correctness & numerics → generation/integration → when the kernel is worth using → external comparison → external validation → **threats to validity** → reproducibility → conclusion. The sprint chronology, error logs and debug histories are demoted to an **Appendices — development record** toctree; the argument is meant to stand without them. ZA stays in the main text as the model negative result.
- [x] **Threats to validity written**, eight of them, including the ones that count against us: one machine/one OS; best-case reporting (now bounded by median/p10–p90, which agree to ~1 %); **no specialist vendor baseline, so no claim is made in either direction**; the native-layout comparison answering only one of two questions; framework margins being version statements; ExecuTorch numbers including runtime dispatch; single-threaded except where stated; and group granularity distorting small shapes.
- [x] **Benchmark tables refreshed** from the `release-submission` preset, with the full run captured at `docs/source/_static/results/main_norm_submission.txt` (provenance header included, 66/66 gated).
- [x] **Six figures**, generated by a dependency-free SVG helper (`docs/figures/`, plain `python3` — no matplotlib, keeping the lean-dependency rule): ceiling-vs-footprint curve, DRAM ablation bars, stability-vs-shift, scalar-crossover, external baselines, and a traversal-structure diagram. Data literals are transcribed from the captured run so a figure is always traceable to it.
- [x] **CI tidied**: configures via `cmake --preset release` (so an empty build type cannot slip through), host-portable vs SME-guarded groups labelled, a header stating plainly that **green CI cannot execute a single SME kernel** and that SME coverage is local-only on M4, plus a **second job running the norm suite under ASan+UBSan**.
- [x] **README fixed** — it had an unterminated code fence and instructed `cmake ..` with no build type, i.e. it documented the exact command behind the Sprint-5 `-O0` defect. Now presets first, with the rule stated.
- [x] **GenAI disclosure updated**: records Claude alongside Gemini 3, clarifies that "boilerplate" meant markup scaffolding (`.. toctree::`, list-table skeletons, heading underlines) rather than prose, and states that AI assisted implementation and drafting with the text rewritten by the authors. The synthesis page carries a draft banner until that rewrite is done.
- [x] **SME → SME2 framed as a beyond-coursework extension** in the report's conclusion: SME was taught, SME2-specific multi-vector optimization was not, and it was reached by detecting the hardware's actual feature set and then testing a hypothesis against it.
- [ ] **Remaining, for the authors:** rewrite the drafted prose in your own words and remove the draft banner; then push to `main` so `docs.yml` deploys to GitHub Pages.

**Done when:** the report tells the full story as a synthesis, the integrated kernel is verified + measured, and every claim is scoped to what was actually shown.

---

## How to use this file with Claude / Claude Code

- Work **one sprint at a time**. Don't pull tasks from a later sprint early.
- Session start: "Read context.md and CLAUDE.md, then let's do Sprint N."
- Tick boxes as we go; the checked history shows progress.
- Every kernel ships with a Catch2 test; build with `-Werror` clean and verify vs the reference before committing.
- SME kernels: build in CI, **run and benchmark on M4**; keep CI green on the M1/M2 runners by skipping SME tests when `cpu_supports_sme()` is false.
- If a sprint feels too big, split its first checkbox into a vertical slice (one norm, one shape, verified) and get THAT running before continuing.