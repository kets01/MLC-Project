# Sprint 3 — errors encountered and how they were solved

An honest log of what actually went wrong while building the RMSNorm ZA
kernel (`rms_norm_za.S`), and how each was fixed. Kept truthful: the kernel
itself assembled and passed all tests on the **first** attempt, so most of
what follows is environmental/process, not kernel bugs. The genuinely
tricky SME hazards were designed around *before* coding (see the last
section) — they never became errors, and it would be dishonest to dress
them up as debugging war stories.

---

## 1. Documentation files "did not exist"

**What I saw**

```
Read context.md   → File does not exist. (cwd: /Users/mlc02/Documents/MLC)
Read ROADMAP.md   → File does not exist.
Read CLAUDE.md    → File does not exist.
```

**Cause.** The working directory was `/Users/mlc02/Documents/MLC`, but the
actual git repository is the `MLC-project/` **subdirectory** of it. The three
docs live at `MLC-project/context.md`, etc., not at the cwd root.

**Fix.** Listed the directory and searched for the files:

```bash
ls -la /Users/mlc02/Documents/MLC
find /Users/mlc02/Documents/MLC -maxdepth 2 -iname "*.md"
```

That located everything under `MLC-project/`, and I read them from there.
Lesson: reconcile the assumed layout against the real tree before trusting
any path (context.md itself warns about this).

---

## 2. `cmake: command not found`

**What I saw**

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
→ /bin/bash: cmake: command not found
```

**Cause.** The tool shell is non-interactive and does not inherit the
interactive login PATH, so Homebrew's `cmake` (installed under
`/opt/homebrew/Cellar/cmake/4.3.2/bin`, symlinked at `/opt/homebrew/bin`)
was not on PATH.

**Fix.** Located the binary and prefixed subsequent commands with an
explicit PATH export:

```bash
which cmake                       # nothing
ls /opt/homebrew/bin/cmake        # found it
export PATH="/opt/homebrew/bin:$PATH"   # prepended for every build/ctest call
```

Not a project bug — a shell-environment gap. Fixed once and reused.

---

## 3. Benchmark row printed the wrong baseline label ("vs V0")

**What I saw.** The new Sprint-3 ablation section compares the ZA kernel to
the SSVE winner **V6**, but the output read:

```
ZA residency   M=128  N=16   6.78 GiB/s  (11.4% peak)  -40.8% vs V0
                                                              ^^^  wrong — it's vs V6
```

The percentage was correct (computed against the V6 baseline I passed in);
only the printed label was wrong.

**Cause.** `print_ablation_row()` in `apps/main_norm.cpp` had a **hard-coded
string literal** `"% vs V0"` — fine for the Sprint-2 tables (all vs V0), but
my section uses V6 as the baseline.

**Fix.** Added an optional parameter with a backward-compatible default, so
every existing caller is unchanged and my section passes `"V6"`:

```cpp
static void print_ablation_row(..., double base_gibs,
                               const char* base_name = "V0") {
    ...
    std::cout << ... << delta_pct << "% vs " << base_name;
}
// ZA section:
print_ablation_row("ZA residency", vm, vn, gza, vpeak, g6, "V6");
```

Rebuilt and confirmed the rows now read `-40.8% vs V6`.

---

## 4. Commit rejected — messages too long

**What I saw.** The first attempt to commit the docs/roadmap used multi-line
commit bodies; the run was rejected with feedback: *"those messages are too
long."*

**Cause.** Process/style, not a code error — the multi-paragraph bodies were
more than wanted.

**Fix.** Re-ran with single-line conventional-commit subjects, e.g.:

```
docs(norm): Sprint 3 report — ZA residency design + verdict (39-65% slower than V6)
docs(roadmap): tick Sprint 3 RMSNorm ZA — V6 stays incumbent
```

---

## SME/ZA pitfalls I designed around (avoided, not encountered)

These are the traps that *would* have produced assemble-fine-but-run-wrong
bugs — the dangerous kind (CLAUDE.md §10: "it assembles" ≠ "it's correct").
I avoided them by studying `src/week3/gemm_32_32_1.S` and the Arm ARM first,
so they never became errors. Listed here because *why they didn't happen* is
the real learning content.

- **ZA tile number is an instruction immediate, not a register.** You cannot
  loop over `za{t}v` with a computed `t`. → Emitted the four tiles as four
  `.macro` expansions (`ZA_LOAD 0..3`), not a runtime loop.
- **The ZA slice-select register must be W12–W15.** Using any other W
  register would mis-assemble or mis-encode. → Used `w13` throughout for the
  `za{t}v.s[w13, 0]` slice index.
- **`smstart`/`smstop` must manage PSTATE.ZA, not just PSTATE.SM.** The
  Sprint-2 SSVE kernels use `smstart sm` (SM only); a ZA kernel needs bare
  `smstart`/`smstop` (both SM **and** ZA). → Used bare forms, and wrote every
  ZA slice before reading it so no `zero {za}` was required, and disabled ZA
  before `ret` so there is no AAPCS64 lazy-save hazard.
- **The reduction must stay SSVE.** Routing the horizontal sum-of-squares
  through the ZA matrix accumulator maps poorly (context.md §5). → ZA holds
  `x` only; the reduction is `fmla` in Z-registers.
- **VLA — no literal 16 (decision D).** → Tile geometry and the
  `N ≤ 4·SVL` path split are derived from `cntw` at runtime.
- **`SMSTART` zeroes D9–D15.** The benchmark harness already knew this (the
  Sprint-2 `volatile` discipline); I followed the same pattern so timing
  doubles were not silently zeroed.

**Net:** the kernel built clean (`-Werror`) and passed all 86 `test_norm`
cases (205 021 assertions) on the M4 on the first run — including tile
boundaries, row tails, mismatched leading dims, the streaming fallback, and
the large-magnitude stress case. The only real errors this sprint were the
environmental/process ones above.
