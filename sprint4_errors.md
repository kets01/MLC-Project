# Sprint 4 — errors encountered and how they were solved

An honest log of what actually went wrong while building the `mini_jit::Norm`
JIT generator, and how each was fixed.  Kept truthful, as in Sprint 3: the
generator itself passed the encoding-diff and all execution tests on the
**first** run — the encoding work went in without a single debugging session,
*because* every encoder was written against golden words extracted from the
toolchain-assembled kernels before any emission code existed.  The real
errors this sprint were found *by* that methodology in code that already
existed and was believed correct — which is precisely the argument for it.

---

## 1. `ctest` found no tests ("No tests were found!!!")

**What I saw**

```
$ ctest --test-dir build
Test project .../MLC-project/build
No tests were found!!!
```

despite every module registering with `add_test(...)`.

**Cause.** In the root `CMakeLists.txt`, `enable_testing()` sat at the
**bottom of the file — after all `add_subdirectory()` calls**.  CMake only
writes a subdirectory's tests into `CTestTestfile.cmake` if testing is
already enabled when that directory is processed, so every `add_test()` in
the week/norm modules was silently ignored.  Pre-existing on `main` since
the repo began; nobody noticed because everyone ran the test binaries
directly (`./build/src/norm/test_norm`), which CLAUDE.md lists as an
accepted alternative.

**Fix.** Moved `enable_testing()` above the `add_subdirectory()` block
(commit `fix(build)`).  `ctest --test-dir build` now discovers and runs all
seven suites.

---

## 2. `Norm.hpp` could not be created — APFS is case-insensitive

**What I saw**

```
Write include/norm/Norm.hpp
→ error: File has not been read yet.  (i.e. "this file already exists")
```

for a file that did not appear in `ls`.

**Cause.** macOS APFS is **case-insensitive by default**: `Norm.hpp` and the
existing `norm.hpp` are the *same directory entry*.  The tooling saw an
existing file where I intended a new one.  The `context.md` target layout
(`src/mini_jit/Norm.h`) silently assumed a case-sensitive filesystem.

**Fix.** Named the generator files `jit_norm.hpp` / `jit_norm.cpp` (also
more consistent with the repo's lowercase header style: `norm.hpp`,
`jit_engine.hpp`, `jit_kernel.hpp`).  Noted in the header comment why the
name deviates from context.md §12.

---

## 3. Latent InstGen bug: `sve_ptrue_all(fp32)` emitted `PTRUE P.B`, not `P.S`

**What I saw.** Preparing the encoding diff, the golden word for
`ptrue p0.s` in both assembled V6 kernels is `0x2598e3e0`, but the week6
encoder returned `0x2518e3e0`.  Decoding the difference: bits 23:22
(the size field) were `00` (= `.B`) instead of `10` (= `.S`) — the
encoder's own doc comment says "size=10 (bits 23:22)" but the constant
never matched it.

**Cause.** Wrong base constant in `Instgen.cpp`, present since week 6 and
functionally **masked in every existing caller**: an ALL-true `.B`
predicate governs `.S` operations identically to an ALL-true `.S`
predicate, so Unary and Gemm still computed correct results.  A textbook
"it assembles and runs, but is not the instruction you meant"
(CLAUDE.md §10) — invisible to behavioral tests, caught the moment an
exact word comparison existed.

**Fix.** Corrected the constant to `0x2598e3e0` (now matching the comment),
added a pinning unit test, and re-ran the week5/week6 suites (green — the
change is behavior-preserving for them by the masking argument above).

---

## 4. Latent InstGen naming bug: `sme_smstart_sm()` actually enables ZA too

**What I saw.** The golden word for the norm kernels' `smstart sm` is
`0xd503437f` (`MSR SVCRSM, #1` — streaming mode only).  The week6
`sme_smstart_sm()` returns `0xd503477f`, which is `MSR SVCRSMZA, #1` —
`SMSTART` with **both** PSTATE.SM and PSTATE.ZA.  Same for the stop pair
(`0xd503427f` vs `0xd503467f`).

**Cause.** Misnamed encoder.  Harmless-to-necessary for its existing users
(Gemm genuinely needs ZA for `fmopa`; Unary merely enables ZA needlessly),
but wrong for the norm kernels, which deliberately run SM-only — and it
would have silently introduced the AAPCS64 lazy-save/ZA-state hazards the
Sprint-3 kernels were carefully designed around.

**Fix.** Did **not** change the existing encoder (Gemm depends on its actual
behavior); added correctly-named `sme_smstart_sm_only()` /
`sme_smstop_sm_only()` with a doc comment spelling out the distinction, and
improved the old pair's comments.  Pinned both words in the encoder tests.

---

## 5. JIT kernel benchmarked FASTER than the word-identical hand-written one

**What I saw.** First run of the Sprint-4 parity benchmark:

```
RMS V6 hand-written   M=128  N=64    22.54 GiB/s
RMS V6 JIT-emitted    M=128  N=64    39.61 GiB/s   +75.7% vs hand
...
RMS V6 hand-written   M=1024 N=2048  37.98 GiB/s
RMS V6 JIT-emitted    M=1024 N=2048  38.47 GiB/s   +1.3%  vs hand
```

The encoding diff proves the two kernels are **word-for-word identical**, so
+75.7% at the small shape is physically impossible for the code itself —
the discrepancy had to be in the *call path*, and it shrank with shape size
exactly like a fixed per-call cost.

**Cause.** The hand-written kernels are benchmarked through their
`mini_jit::norm` guard wrappers, and `cpu_supports_sme()` performed an
**uncached `sysctlbyname` syscall on every single call** (~1 µs).  The JIT
function pointer is called directly, no wrapper.  At M=128/N=64 the kernel
itself does ~1.3 µs of work, so the syscall was ~75% overhead; at
M=1024/N=2048 it amortizes into the noise.

Ripple effect: every small-shape number in the Sprint-2/3 tables — and in
particular the Sprint-2a small-N fit's headline "fixed cost t0 ≈ 1 µs/call,
overhead = streaming work at N ≈ 30–40" — was measured **through the same
wrapper** and therefore includes ~1 µs of *syscall*, not kernel overhead.
The qualitative conclusions survive (there IS a fixed per-call cost; the
N=64 dip is real), but t0 was roughly doubled by the harness, and the
"overhead = work" crossover N was overstated.  Corrected numbers after the
fix: see the re-measured Section 5/6 tables in the report.

**Fix.** Cached the sysctl result in a function-local `static` (CPU features
cannot change at runtime; static-local init is thread-safe).  Re-ran the
full suite and the benchmark: JIT and hand-written are now at parity within
noise at **all** shapes, and the small-N sweep's t0 dropped accordingly.
The Sprint-4 report section documents both the before/after and the
correction to the Sprint-2a interpretation.

---

## 6. Guard fix fallout: small-N sweep printed `GiB/s 0.00` (D-register clobber, again)

**What I saw.** Immediately after fix #5, re-running the benchmark:

```
--- Small-N regime ---
M=128:
  N        us/call    GiB/s (% 1-core SSVE peak)
  16        0.416      0.00     0.0 %      <- times right, GiB/s zero
```

The µs/call column showed the expected improvement (N=16 dropped from
1.375 µs to 0.416 µs — the syscall gone), but every GiB/s printed 0.00.
The same computation had printed correctly for weeks.

**Cause.** The known SMSTART hazard, resurfaced by an unrelated change.
`SMSTART` zeroes the callee-saved FP registers D9–D15 behind the compiler's
back (the kernels save only D8), so the harness keeps timing values in
`volatile` stack slots.  `small_n_sweep()` was the one section computing
`to_gibs(norm_bytes(m, n), secs[k])` in a single expression without a
volatile intermediate — it had merely been *lucky* in register allocation.
Making `cpu_supports_sme()` trivially inlinable (fix #5) changed the
allocator's choices, and part of the GiB/s computation landed in a
D9–D15 register that the kernel call zeroes.

**Fix.** The same two-hop volatile discipline used everywhere else in the
file (`volatile double vbytes = ...; volatile double vgibs = to_gibs(...)`),
with a comment naming the trigger.  Lesson repeated from Sprint 2: with an
ABI-violating callee, *every* FP value computed near the call must be
forced to the stack — "it printed fine last week" is register-allocator
luck, not correctness.

**Corrected result worth recording:** with the syscall removed, the N=16
call drops from 1.375 µs to ~0.42 µs, throughput is flat (~29 GiB/s) from
N=32 upward, and the fixed per-call cost is too small for the linear fit to
resolve (intercept within noise of zero, ≲0.4 µs) — the Sprint-2a headline
"t0 ≈ 1.4 µs/call, overhead = streaming work at N ≈ 41" was ~70% syscall.
The report's Sprint-4 section restates the small-N regime accordingly.

---

## 7. Environment: `cmake` not on PATH (recurring from Sprint 3)

Same as Sprint 3 error #2: the tool shell doesn't inherit the interactive
PATH, so every cmake/ctest invocation is prefixed with
`export PATH="/opt/homebrew/bin:$PATH"`.  Known, worked around immediately,
listed only for completeness.

---

## 8. DRAM-regime ZA addendum: the D9–D15 clobber, a third time — now a TEIR blocker

**What I saw.** Adding four true-DRAM shapes (32–64 MB) to the RMSNorm ZA
ablation and running `main_norm` on a **fresh Release build**, every GiB/s
column printed `0.00` — not just the new rows, but *every* row after the
first SME kernel call, including the scalar-reference rows that touch no SME
at all. The ceiling probes (run before any norm kernel) measured fine
(59.5 GiB/s), and the very first row (`layer_norm_ref`, scalar) printed 4.09
before the first streaming call poisoned everything after it.

**Cause.** The same ABI-violating callee from #6 and Sprint 2:
`smstart`/`smstop` zero `d8–d15` and the V6 kernels save/restore only `d8`,
so a caller's `d9–d15` are destroyed. This build's register allocator placed
timing state in `d9–d15`; the `volatile double best` in `bench()` was not
enough because the *timestamp* values (not just the accumulator) landed
there too. Confirmed by writing a standalone driver: per-call timing and
loop-outer timing both failed until **all** loop state — counter, bound,
both timestamps — was forced to `volatile` memory, after which V6 measured
consistently (the re-measured V6 matched the first within noise).

**Fix (for the measurement).** A standalone driver with fully volatile loop
control (`scratchpad`, not committed); numbers folded into the report's
DRAM-addendum table and the `main_norm` ZA shape list.

**Escalation (the real lesson).** Twice now this was patched caller-side with
`volatile`. That works only because *we* control the caller. **TEIR's runtime
(Sprint 5) calls the kernel from a generated loop nest with no such
discipline**, so the latent bug becomes a live one that will corrupt runtime
state. Recorded as a Sprint-5 **prerequisite** in `ROADMAP.md`: fix the
kernels properly (save/restore `d8–d15` around the streaming region) in both
the `.S` and the `mini_jit::Norm` generator, then re-run the Sprint-4
encoding-diff so the two stay bit-identical. "Force every nearby FP value to
the stack" is a workaround; preserving the callee-saved registers is the fix.

---

## 9. clang backend crash compiling a C++ caller with `-march=…+sme`

**What I saw.** Building the standalone measurement driver (plain C++ that
just *calls* the prebuilt kernels) with `-march=armv9-a+sme`:

```
fatal error: error in backend: Cannot select: … AArch64ISD::UUNPKLO …
clang frontend command failed with exit code 70
```

**Cause.** Passing `+sme` to a C++ translation unit lets the compiler try to
lower `std::vector`/lambda code through scalable-vector codegen, and Apple
clang 16's backend hits an unimplemented selection path. The project never
does this: the kernels are hand-written `.S`, and only the assembler sees
`+sme` (`target_compile_options(norm_lib …)` applies to the `.S` files); the
C++ callers are compiled with plain `-march`.

**Fix.** Compiled the driver without `+sme` — it only needs to *link* the
`.a`, not generate SME. Noted for anyone writing a new C++ harness: do **not**
add `+sme` to C++ TUs; it belongs on the assembly sources only.

---

## What did NOT go wrong — and why (the methodology content)

The generator emitted **337 instruction words across two kernels — plus 30
new instruction encoders — and every test passed on the first run**:
encoding diff green for both norms (143 + 194 words), all execution tests
green on M4, 34,006 assertions.  That is not luck; it is the order of
operations:

- **Golden words before any encoder.**  Both winning `.S` kernels were
  assembled by the toolchain and `objdump`'d *first*; every encoder was
  written against the Arm ARM field layout and immediately pinned to a
  golden word from that disassembly.  No encoder was ever "probably right".
- **The diff reads the linked kernel, not a copy.**  The encoding-diff test
  takes the address of the linked `rms/layer_norm_ssve_v6` functions and
  compares the JIT buffer against the actual code bytes in the test binary —
  so the reference can never go stale if the `.S` changes.
- **Transcription, not re-derivation.**  The emission functions are a 1:1
  transcription of the hand-written kernels (same registers, same order),
  so the JIT inherits the trust of a kernel that already passed 350k+
  assertions — including the Sprint-3 eps-stash fix — instead of
  re-earning it.
- **Branch offsets computed, then verified by the same diff.**  Forward
  branches are backpatched from recorded label indices; a single off-by-one
  anywhere would break the word-for-word diff loudly at that index.

The two latent week6 bugs (#3, #4) and the harness bug (#5) were all found
*by* this exactness, in code that behavioral tests had passed for weeks.
That is the sprint's real lesson: behavioral tests verify what code does;
an encoding diff verifies what code *is* — and the second catches classes
of error the first structurally cannot.
