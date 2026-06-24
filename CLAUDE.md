# CLAUDE.md — MLC-Norm working conventions

> How to work on MLC-Norm. Read alongside the other two files:
> - `context.md` = what the project is, target architecture, design decisions (A–F), the kernels, the stack.
> - `ROADMAP.md` = the route: sprint-by-sprint, reference + SSVE first, then SME, JIT, TEIR, evaluation.
> - **This file** = conventions: git, commits, tests, build, ISA rules, correctness gates, tooling, style.
>
> The norm work extends an existing, merged lab repo (**MLC-Project**, `github.com/kets01/MLC-Project`),
> weeks 1–7. Match the repo's established habits — don't invent parallel ones.

---

## 1. How to work with us (collaboration style)

- This is a **two-person learning project** (Ketsia & Mariza) for the ML Compilers Lab. Treat us as
  **pair-programming partners, not spectators** — the point is that we understand every part, including the
  assembly and the instruction encodings.
- Before a big step, briefly state the plan. After non-trivial code, explain the **why** — especially for
  anything in assembly or the JIT, where a working build can still hide a misunderstanding.
- Prefer the simplest thing that works; flag when we're overcomplicating.
- **One question at a time** when you need a decision from us.
- Work **one ROADMAP sprint at a time**; don't pull tasks from later sprints early.
- Session start: "Read context.md and CLAUDE.md, then let's do Sprint N."

---

## 2. Golden rules (carried from the design)

- **Learning over speed.** A kernel we fully understand and can defend beats a faster one we can't.
- **One new thing at a time.** Never debug two unfamiliar things at once (e.g. a new SME instruction *and* a
  new memory layout, or the JIT *and* TEIR integration).
- **Vertical slices.** Each step runs end-to-end: input → kernel → verified against the reference → GiB/s measured.
- **Correctness gates performance (decision B).** No GiB/s number is reported for a kernel that hasn't passed
  numerical verification against the C++ reference.
- **Measure before optimising (decisions E/F).** Establish the roofline and trace per-variant timing before
  tuning; fix the real bottleneck, not a guess.

---

## 3. Coding conventions

- **C++17** (the repo's `CMAKE_CXX_STANDARD`), AArch64 **assembly** (`.S`) for the kernels. Type-safe,
  documented public functions.
- **English** for all code, comments, docs, and commit messages.
- **Strict warnings are already on and `-Werror`** (`-Wall -Wextra -Werror -pedantic -g`). Keep the build
  warning-clean; do not relax the flags to get past a warning — fix the cause.
- **Follow the per-module pattern** the weeks use, so the norm slots in identically:
  - `src/norm/` — `CMakeLists.txt` + sources (`reference.cpp`, the `mini_jit::Norm` generator, kernel `.S` files),
    `add_library(norm_lib ...)` with `target_compile_options(norm_lib PRIVATE -march=armv9-a+sme)`, linking
    `week6_lib`/`week7_lib` as needed.
  - `include/norm/` — public headers (`.hpp`), including the **canonical kernel signature** (decision A).
  - `tests/test_norm.cpp` — Catch2 test, built as `test_norm`, linked to `norm_lib` + `Catch2::Catch2WithMain`,
    registered with `add_test(NAME NormTests COMMAND test_norm)`.
  - `apps/main_norm.cpp` — the `bench_norm` benchmark app (no Catch2).
  - Register `add_subdirectory(src/norm)` in the root `CMakeLists.txt`.
- **Reuse the lab code, don't reimplement it.** The SSVE machinery (`week6 Unary`/`InstGen`), the SME GEMM
  (week3/week6), the JIT engine (week5), and the TEIR runtime (week7) are the building blocks. Read them
  before extending.
- **Match the data conventions:** FP32, **column-major**, explicit **leading dimension** (`data[i + j*ld]`),
  as in the existing kernels.
- **Lean dependencies.** Catch2 (via FetchContent) is the only third-party dependency — keep it that way.
- **Explain as you go:** when you implement something non-trivial (a reduction, a predicated tail, an
  instruction encoding), say *why* in a comment and to us.

---

## 4. ISA & configuration rules (no magic numbers)

- **Never hard-code the streaming vector length (decision D).** Query SVL at runtime; write loops to any SVL;
  use the `InstGen` SVE predicates (`p0–p15`) for the tail. The ZA-tile granularity is **`SVL/32` FP32
  elements** (16 on the M4's 512-bit SVL) — *derived*, not a literal `16` baked into the kernel.
- **Guard every SME/streaming code path with `cpu_supports_sme()`** (from `include/week3/utility.hpp`) so it
  degrades cleanly on hardware without SME.
- **Fail fast and clearly.** If a kernel is asked for a shape it can't serve, or an emitted instruction word
  doesn't decode as intended, error loudly at the boundary — not with silent wrong numbers mid-benchmark.
- **Shapes, tolerances, and tile parameters are named values passed in**, not constants sprinkled through the
  code. The `-march` feature flags belong in the module's `CMakeLists.txt` (e.g. `+sme`, `+sme-f64f64`), as
  the weeks already do — not scattered.

---

## 5. Tests & CI (from day one — already the repo's habit)

- **Every kernel ships with at least one Catch2 test** that verifies it against the scalar C++ reference
  within an FP32 tolerance — including the **numerical-stability stress inputs** (large-magnitude / shifted
  values that expose single-pass cancellation; see `context.md §8`). Don't loosen the tolerance to pass —
  fix the kernel.
- **CI builds on every push and PR** (`.github/workflows/tests.yml`, on `macos-latest`). Add the norm test to
  it (`./build/src/norm/test_norm`) alongside the week runs.
- **SME runs on hardware, not CI.** The CI runner is Apple Silicon **without SME** (M1/M2). So:
  - SME/JIT kernels must still **build** in CI (they compile with `-march=armv9-a+sme`).
  - SME-**executing** assertions are guarded with `if (!cpu_supports_sme()) SKIP("SME required");` (the
    pattern already used in `test_week5`/`test_week6`) so `test_norm` runs and *skips gracefully* on the CI
    runner, and runs fully on **M4**. Prefer this `SKIP` over commenting tests out (as week3 currently does).
  - **Run the SME tests and the benchmark locally on M4** before committing — green CI alone does not prove
    the SME path works.
- The docs workflow (`docs.yml`) builds the Sphinx report and deploys to GitHub Pages on push to `main` —
  keep it green too.
- Locally before committing: configure + `cmake --build`, run `ctest` (or the test binaries directly), and
  the benchmark on M4.

---

## 6. Commit & branch conventions

- **Small, atomic commits**, one logical change each, with **conventional commit messages** (e.g.
  `feat(norm): add single-pass RMSNorm SSVE kernel`, `test(norm): add cancellation stress cases`). The git
  history should tell the story: reference → SSVE → VLA → SME ZA → JIT → TEIR → evaluation.
- **One file per commit** — stage and commit each file individually. Never bundle multiple files into one
  commit unless they are truly inseparable (e.g. a header and the `.cpp` that implements it).
- **Feature branches + PR into `main`** — never commit straight to `main`. CI runs on PRs, so open the PR and
  let it go green before merging. (Natural fit for a two-person team — review each other's kernels.)
- Commits are authored under **each author's own git identity**. **Never add a "Co-authored-by" line or any
  AI attribution.** Verify `git config user.name`/`user.email` match the author's GitHub account so commits
  show on their profile.
- Commit after each working step, not in one big dump at the end.

---

## 7. Correctness, safety & hygiene

- **JIT executable-buffer safety (decision F).** When emitting instruction words (extending the week5 engine),
  size and align the executable buffer correctly and respect W^X — a wrong buffer is a crash or silent
  corruption, not a compile error. Verify emitted words decode as intended.
- **Buffer discipline.** Aligned FP32 buffers, correct leading dimensions, no out-of-bounds on predicated
  tails. The `-Werror` build catches a lot; the rest is on the tests (run them with sanitizers locally when a
  layout bug is suspected).
- **No undefined behavior.** Keep the strict-warning build clean; don't silence warnings with casts you can't
  justify.
- **Numerical honesty.** Report the actual tolerance a kernel meets; if a "fast" variant only passes at a
  loosened tolerance, that is a result to document (§8 of context.md), not a knob to quietly turn.
- **Repo hygiene.** Don't commit build artifacts or environments — `build/`, `docs/_build/`, `mlc_env/`,
  `.vscode/` are already in `.gitignore`. No secrets in the tree (there are none today — keep it so).

---

## 8. Tooling (see context.md / ROADMAP.md for where each is used)

This is low-level C++/AArch64-assembly work, so the document/UI skills don't apply. The references that matter:

- **The existing week5–7 code is the primary template.** `mini_jit::Unary`/`InstGen` for the JIT step;
  the week7 TEIR runtime (`Axis`/`Iteration`/`Invocation`, `CompiledKernel = void(float*,float*,float*)`) for
  integration. Read them before extending.
- **Arm Architecture Reference Manual (SME/SSVE)** + the lab's `tnzr.org/compile` material — the authority on
  instruction encodings and ZA/streaming semantics. **Critical and exact** for the JIT encoding step. Verify
  against the M4's SME level (SME1) at coding time.
- **GDB** — stepping the hand-written assembly, inspecting registers / predicates.
- **skill-creator** (official, optional, early): write a small MLC-Norm conventions skill as a learning step.

> Learning note: a generator can emit correct instruction words we don't yet understand. Always trace the
> encoding back to the manual and have it explained — comprehension over an opaque build.

---

## 9. Reporting & reproducibility (the Sphinx report is a deliverable)

- **Update the Sphinx report at the END of every sprint**, the way the weekly reports already work — add the
  norm section (`docs/source/project_norm.rst` or a `weeks/`-style entry) and keep it listed in `index.rst`.
  It must always reflect the real current state and the latest numbers.
- **Match the repo's existing "GiB/s (Read + Write)" reporting** for the benchmark tables, and keep an
  **ablation table** (scalar → SSVE → VLA → ZA-tiled → JIT; RMSNorm vs LayerNorm) — that table is the core
  evaluation story (decisions C/E).
- **Reproducible by default:** `mkdir build && cd build && cmake .. && make`, fixed shapes, documented how to
  regenerate the GiB/s numbers (and on which machine — M4 for SME).

---

## 10. Verify against the hardware and the manual — not memory

"It assembles" does not mean "it's the right instruction." Confirm against ground truth, not recollection:
- **Instruction encodings** — against the Arm ARM, especially in the JIT step (a wrong encoding can assemble
  *and* run, just wrongly).
- **SME level / available instructions** — SME1 on the M4; don't assume SME2-only instructions exist.
- **Peak bandwidth (the roofline)** — measure it on the actual M4 (a STREAM-style probe); don't quote a
  datasheet figure.
