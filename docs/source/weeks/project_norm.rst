MLC-Norm Project
================

Sprint 0 — Scaffold
-------------------

**Goal:** prove the norm module is wired into the existing build/test/CI/Sphinx
pipeline with placeholder logic — no real kernel yet.

Module layout
~~~~~~~~~~~~~

.. code-block:: text

    include/norm/norm.hpp       — canonical kernel signatures (pinned in Sprint 1)
    src/norm/reference.cpp      — scalar C++ reference for LayerNorm and RMSNorm
    src/norm/CMakeLists.txt     — build wiring
    tests/test_norm.cpp         — Catch2 verification harness (normal + stress cases)
    apps/main_norm.cpp          — GiB/s benchmark + STREAM-style peak-bandwidth probe

Sprint 0 status
~~~~~~~~~~~~~~~

- **Build:** ``cmake --build`` produces ``test_norm`` and ``main_norm``
- **Test:** placeholder identity copy verified; test runs in CI on M1/M2 (no SME required)
- **CI:** ``test_norm`` in the host-portable group alongside week1/week2
- **Docs:** this section

Background
~~~~~~~~~~

MLC-Norm adds LayerNorm and RMSNorm kernels for AArch64 SME/SSVE on top of
the existing lab foundation (weeks 1–7). The full stack target is:

- **C++ reference** → hand-written SSVE kernel → JIT-generated via ``mini_jit::Norm``
  → composed into a TEIR loop nest via the week7 runtime

The two norms differ in reduction structure:

- **LayerNorm** — two-pass (mean, then variance + normalize-scale-shift); more
  arithmetic intensity, numerically stable
- **RMSNorm** — single-pass (sum of squares, no mean, no β); ~10–40% faster

Both are bandwidth-bound; the evaluation metric is **effective GiB/s** vs the
measured peak.

Sprint 1 — C++ reference, correctness harness, and bandwidth baseline
----------------------------------------------------------------------

**Goal:** an obviously-correct scalar reference for both norms, a Catch2
verification harness (including numerical-stability stress cases), and a
reproducible GiB/s measurement that sets the roofline target for all future
kernels.

Canonical kernel signature (decision A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both norms share a single interface defined in ``include/norm/norm.hpp`` under
``namespace mini_jit::norm``.  The same signatures are used by the C++
reference, the future ``mini_jit::Norm`` JIT generator, and the TEIR
registration — no layer invents its own.

**Layout convention:** column-major, explicit leading dimension
(``data[row + col * ld]``), matching the rest of the lab code.

.. code-block:: cpp

    // LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    void layer_norm_ref(const float* a, float* b,
                        const float* gamma, const float* beta,
                        int64_t m, int64_t n,
                        int64_t ld_a, int64_t ld_b,
                        float epsilon);

    // RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
    void rms_norm_ref(const float* a, float* b,
                      const float* gamma,
                      int64_t m, int64_t n,
                      int64_t ld_a, int64_t ld_b,
                      float epsilon);

Reference implementations (decision B/C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``src/norm/reference.cpp`` implements both norms as deliberately simple scalar
C++ using ``double`` accumulation throughout — this makes it the trusted oracle
against which every future kernel is verified.

**LayerNorm — two-pass per row:**

1. Pass 1: accumulate mean (``sum / N``), then variance ``E[(x − mean)²]``.
   The two-pass structure avoids the catastrophic cancellation that a naive
   single-pass ``E[x²] − mean²`` formula produces on large-magnitude inputs.
2. Pass 2: normalize, scale by γ, shift by β.

**RMSNorm — single-pass per row:**

Sum of squares ``Σx²``, divide by N, add ε, take the inverse square root, then
scale by γ.  No mean subtraction and no β — the simpler reduction is what makes
RMSNorm ~10–40% faster at equal accuracy relative to LayerNorm.

Correctness harness (decision B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_norm.cpp`` covers four cases for each norm:

- **Identity / unit-vector input** — analytically checkable output (e.g. all-equal
  input → LayerNorm output equals β; unit-vector input → RMSNorm output scales
  by ``sqrt(N)``).
- **Normal random-ish input** — verified against an independent scalar ``double``
  computation in the same test file.
- **Non-square matrix with ``ld_a ≠ ld_b``** — exercises leading-dimension
  handling.
- **Large-magnitude stress input** (``SHIFT = 1e4f`` plus small variation) —
  documents that the two-pass LayerNorm and the mean-free RMSNorm both remain
  accurate on inputs where a naive single-pass variance would lose bits.

All tests run on the CI runner (M1/M2, no SME required) with tolerance
``epsilon = 1e-5``.

Bandwidth harness and roofline (decision E)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``apps/main_norm.cpp`` measures:

1. **Peak bandwidth** — a STREAM-style ``dst[i] = src[i] + 1.0f`` loop over
   128 MiB per array (beyond M-series L3), best-of-10 runs, to establish the
   roofline target.
2. **Norm effective bandwidth** — ``bytes = 2 × M × N × 4`` (one FP32 read +
   one FP32 write; γ/β assumed L1-resident), best-of-50 runs, for six shapes:

**Roofline target (STREAM-style probe, single-threaded scalar):** 10.62 GiB/s.
This is the single-threaded scalar ceiling; the SSVE and JIT kernels will
target the vectorised peak, which is significantly higher.

.. list-table:: Sprint 1 — reference GiB/s (scalar C++, Apple M4)
   :header-rows: 1
   :widths: 10 10 20 20 20

   * - M (rows)
     - N (features)
     - ``layer_norm_ref`` GiB/s
     - ``rms_norm_ref`` GiB/s
     - Notes
   * - 128
     - 64
     - 1.91 (18% of peak)
     - 3.10 (29% of peak)
     - Small working set, fits in L1/L2
   * - 128
     - 512
     - 1.66 (16% of peak)
     - 2.47 (23% of peak)
     -
   * - 128
     - 2048
     - 1.60 (15% of peak)
     - 2.38 (22% of peak)
     - Row resident across passes
   * - 1024
     - 64
     - 0.80 (8% of peak)
     - 1.29 (12% of peak)
     - More rows → more scalar loop overhead
   * - 1024
     - 512
     - 0.78 (7% of peak)
     - 1.14 (11% of peak)
     -
   * - 1024
     - 2048
     - 0.68 (6% of peak)
     - 0.89 (8% of peak)
     - Large tensor, pressure on caches

Two patterns visible in the data:

- **RMSNorm is consistently 1.3–1.6× faster than LayerNorm** at the same
  shape — the single-pass structure eliminates the second loop over the feature
  axis and the mean subtraction, halving the arithmetic work.
- **Performance degrades as M grows** — the scalar loop overhead (branch,
  pointer arithmetic, FP divide) accumulates per row.  The SSVE kernel will
  amortise this by processing multiple elements per cycle.

The numbers sit at 6–29 % of the single-threaded STREAM ceiling.  The gap is
the target for Sprint 2's SSVE kernel.

Sprint 1 status
~~~~~~~~~~~~~~~

- **Reference:** both ``layer_norm_ref`` and ``rms_norm_ref`` implemented and
  merged (PR #24).
- **Signature:** canonical interface pinned in ``include/norm/norm.hpp``.
- **Tests:** 8 Catch2 cases (4 per norm), all green on CI, including
  large-magnitude stability-stress inputs.
- **Benchmark:** GiB/s harness + STREAM peak-bandwidth probe in place;
  numbers recorded from M4 (see table above).

Sprint 2 — Summary and reading guide
-------------------------------------

Sprint 2 delivers the MVP: a correct, VLA Streaming-SVE kernel for **both**
norms, verified against the C++ reference and measured against a **validated**
roofline, with the hand-written optimization space explored to its verdict.
Both norms converge on the same winning structure (**V6**, 4-row-block
contiguity grouping); the sprint's headline result is *why*, and what each
lever did or did not buy.

Headline results (Apple M4, useful-bytes GiB/s, M=1024 × N=2048 = 16 MB)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - stage
     - RMSNorm
     - LayerNorm
   * - scalar C++ reference
     - 1.0 GiB/s
     - 1.0 GiB/s
   * - V0 (naive VLA SSVE)
     - 22–25
     - 12–13
   * - **V6 (incumbent)**
     - **37.8 (64 % of roofline)**
     - **16.3 (27 % of roofline)**
   * - V6 in moved-bytes terms
     - ~57 (95 % of roofline)
     - ~33 (55 % of roofline)

**The three validated ceilings** (Sprint 2a): single-core NEON **79.4**,
single-core SSVE-streaming **59.4** (← the kernel roofline, same execution
mode as the kernels), chip-wide 10-thread **89.9** GiB/s (← the Sprint-5
threading target). All "% of roofline" above is against the 59.4 figure.

**RMSNorm is 1.8–2.3× faster than LayerNorm** on identical shapes (LN/RMS =
0.43–0.56) — beyond the proposal's "10–40 %". The gap is structural: a 1.33×
traffic floor (LayerNorm's three passes move 3R+1W vs RMSNorm's 2R+1W) plus
LayerNorm's extra per-byte work (mean-subtract, β, two serialization points
per block).

What each lever bought (both norms, detail in the sub-sections below)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **V1** (FRSQRTE+NR inv_rms/inv_std): ≈0 on both — reciprocal-sqrt latency is
  off the critical path.
- **V2** (pre-computed 1/N): ≈0 on RMSNorm, **+9–12 % on LayerNorm** — the one
  verdict that does *not* transfer, because LayerNorm has two per-block FDIVs
  at serialization points.
- **V3** (×2 unroll): ≈0 — the OOO core already overlaps the loads.
- **V4** (4-accumulator ILP): **+27–33 % on both** — the single-accumulator
  dependency chain was the real bottleneck (this is why V3 was flat).
- **V5** (load pipelining): ≈0 vs V4 — OOO rename already hoists the loads.
- **V6** (4-row-block contiguity): ties V4 at cache-resident shapes and wins
  big in the true-DRAM regime (**RMSNorm +131 %, LayerNorm +71 % vs V0** at
  64 MB) — the winning lever is *access density*, not residency (residency was
  measured to be already satisfied). **Incumbent for both norms.**
- **Welford** (LayerNorm single-pass): a documented **double negative** —
  25–30 % slower than V6 *and* ~100× less accurate on shifted FP32 data. On
  SIMD the stable two-pass wins on both speed and accuracy.

Reading guide (the sections are in build order, not sprint-letter order)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detailed sections below appear in the order the work was actually done,
which is the honest "measure before optimize" narrative — so **Sprint 2b's
first ablation (V0–V3) appears before Sprint 2a (roofline validation)**: the
early V0–V3 percentages were quoted against a peak that 2a later proved was
the wrong ceiling, and 2a restates them. The logical order is:

1. **Sprint 2a — roofline validation** — what "peak" really measures; the
   three ceilings; the byte-counting convention.
2. **Sprint 2b — RMSNorm** — the V0 kernel, the V0–V3 ablation (with its
   interim numbers), then round two (V4–V6) against the validated roofline.
3. **Sprint 2c — LayerNorm** — the V0 kernel and the full V1–V6 + Welford
   ablation, plus the LayerNorm-vs-RMSNorm comparison.

Frozen for Sprint 3+: **RMSNorm V6 and LayerNorm V6** are the SSVE baselines
the ZA kernel (Sprint 3) must beat and the JIT (Sprint 4) must emit. The
quantified hand-off: true 3R→2R pass fusion needs more resident state than the
32 Z-registers hold, so it is the Sprint-3 ZA hypothesis — and LayerNorm is
where it has structural headroom (+33 % traffic reduction available; none on
RMSNorm).

Sprint 2 — Hand-written SSVE RMSNorm kernel
--------------------------------------------

**Goal:** a correct, vectorised, VLA RMSNorm kernel using Streaming SVE
that beats the scalar reference by at least 4×.

Why column-major changes the loop order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data layout is **column-major** (``a[row + col * ld]``), so elements
within a *row* are strided ``ld`` apart in memory, while elements within a
*column* are contiguous.

The naive approach — outer loop over rows, inner reduction over columns per
row — would require loading non-contiguous row elements, normally done with
SVE gather loads (``LD1W`` with a vector offset register).  However, on
Apple M4's SME1 implementation, **gather loads are restricted in Streaming
SVE mode** unless the ``SMEFA64`` feature is present (which Apple has not
exposed).  Attempting to use them causes ``SIGILL``.

The fix: **swap the loop order**.  For each block of ``VL`` rows, walk all N
columns; each column is contiguous in memory, so a plain predicated ``LD1W``
loads ``VL`` row-values in one instruction.

Kernel structure (``src/norm/rms_norm_ssve.S``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel uses three SSVE-available instruction groups:

- ``LD1W {Zt.S}, Pg/Z, [Xn]`` — load ``VL`` contiguous FP32s (one column
  slice across ``VL`` rows)
- ``LD1RW {Zt.S}, Pg/Z, [Xn]`` — broadcast one scalar ``gamma[col]`` to all
  active lanes
- ``ST1W {Zt.S}, Pg, [Xn]`` — store ``VL`` contiguous FP32s

The outer loop is VLA: ``WHILELO`` sets the row-block predicate,
``ADDVL``/``INCW`` advance the pointer and index by one vector length.

**For each block of VL rows:**

1. **Pass 1 — sum of squares:**
   iterate over N columns; for each column load ``VL`` values with ``LD1W``,
   accumulate into a ``VL``-wide accumulator using ``FMLA Z2.S, P1/M, Z0.S, Z0.S``.
   After N iterations: ``Z2[i] = Σ_c a[r_i + c*ld_a]²`` for rows ``r_0 … r_{VL-1}``.

2. **Compute ``inv_rms``:**
   ``FDIV Z2.S, P1/M, Z2.S, Z_N`` (divide by N), ``FADD Z2.S, P1/M, Z2.S, Z_eps``
   (add ε), ``FSQRT Z2.S, P1/M, Z2.S`` (rms), ``FMOV Z4.S, #1.0`` +
   ``FDIV Z4.S, P1/M, Z4.S, Z2.S`` (reciprocal).
   Result: ``Z4[i] = inv_rms[r_i]`` for all active rows.

3. **Pass 2 — normalize and scale:**
   iterate over N columns again; ``LD1W`` loads the column slice of A,
   ``LD1RW`` broadcasts ``gamma[col]``,
   two ``FMUL`` instructions compute ``x * gamma * inv_rms``,
   ``ST1W`` writes the result to B.

VLA correctness
~~~~~~~~~~~~~~~

The tail (when M is not a multiple of VL) is handled automatically by
the ``WHILELO P1.S, X0, X22`` predicate: lane ``i`` is active only when
``X0 + i < M``.  All arithmetic and memory operations use ``P1/M``
(merge-predication) or ``P1/Z`` (zero-predication), so inactive lanes
neither contribute to the sum-of-squares nor write to B.

``ADDVL X19, X19, #1`` advances the row-pointer by ``SVL`` bytes (one full
vector of FP32 rows) and ``INCW X0`` advances the scalar index by VL
elements — both are VLA-correct regardless of the actual SVL at runtime.

Benchmarking note: SMSTART/SMSTOP and caller-saved FP registers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the kernel enters streaming mode (``SMSTART SM``), the V registers
(V0–V31 = the lower 128 bits of Z0–Z31) become undefined in the non-streaming
view.  d0–d7 are caller-saved FP registers, so this is ABI-legal — but the
compiler at ``-O2`` can keep benchmark locals (elapsed time, byte count) in
those registers and silently lose them on every kernel call.

The fix in ``apps/main_norm.cpp``:

- ``bench()`` accumulator declared ``volatile double``.
- SSVE timing wrapped in a ``__attribute__((noinline))`` helper
  (``bench_ssve``), creating a proper ABI call boundary so the compiler
  saves all live locals before entering the function.
- Post-call values reloaded via ``volatile`` to bypass register allocation.

Correctness tests (decision B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_norm.cpp`` adds five Sprint-2 test cases, all guarded with
``if (!cpu_supports_sme()) SKIP(…)`` so they skip cleanly on CI:

1. Small square (M=32, N=32, no tail).
2. Tail exerciser (M=8, N=50 → tail of 2 elements in the last column block).
3. Mismatched leading dims (M=7, N=40, ld_a=8, ld_b=16).
4. Large shape (M=64, N=2048) — spot-checked every 8th element.
5. Stress input (values shifted by ``1e4f``, tolerance widened to ``5e-4f``).

Total: 20 356 assertions, all green on M4.  Sprint-1 tests unaffected.

GiB/s results (Apple M4, Streaming SVE, SVL = 512 bits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Roofline: 79.28 GiB/s** (measured STREAM-style peak on M4 with our harness).

.. list-table:: Sprint 2 — rms_norm_ssve GiB/s vs reference (Apple M4)
   :header-rows: 1
   :widths: 10 10 18 18 18 14

   * - M (rows)
     - N (features)
     - ``rms_norm_ref`` GiB/s
     - ``rms_norm_ssve`` GiB/s
     - % of peak
     - Speedup
   * - 128
     - 64
     - 6.60
     - 18.31
     - 23.1 %
     - 2.8×
   * - 128
     - 512
     - 3.41
     - 25.15
     - 31.7 %
     - 7.4×
   * - 128
     - 2048
     - 3.24
     - 25.01
     - 31.5 %
     - 7.7×
   * - 1024
     - 64
     - 1.40
     - 25.15
     - 31.7 %
     - 18.0×
   * - 1024
     - 512
     - 1.62
     - 24.36
     - 30.7 %
     - 15.0×
   * - 1024
     - 2048
     - 1.03
     - 24.75
     - 31.2 %
     - 24.0×

The SSVE kernel achieves **23–32 % of the 79 GiB/s scalar-STREAM peak** and
delivers a **4–24× speedup** over the C++ reference.  The gap between scalar
STREAM (10.62 GiB/s, Sprint 1) and the M4's true vectorised peak (79 GiB/s)
shows the Sprint-1 roofline was the scalar ceiling, not the hardware ceiling.

The remaining headroom to the 79 GiB/s peak is primarily:

- **Two passes over the data per row** (sum-of-squares + normalize) — the
  future JIT/ZA-tiled kernel can fuse both passes using the ZA accumulator.
- **FDIV / FSQRT latency** — replaceable with a two-step
  ``FRSQRTE`` + ``FRSQRTS`` Newton-Raphson reciprocal square root in Sprint 3.

Sprint 2 status
~~~~~~~~~~~~~~~

- **Kernel:** ``src/norm/rms_norm_ssve.S`` + ``src/norm/norm_ssve.cpp``
  implementing ``mini_jit::norm::rms_norm_ssve``.
- **Build:** ``-march=armv9-a+sme`` scoped only to the SSVE files via
  ``set_source_files_properties``; ``reference.cpp`` compiled without the flag
  (prevents auto-vectorisation with non-streaming SVE, which is illegal on M4).
- **Tests:** 5 Sprint-2 cases + 8 Sprint-1 cases = 13 cases, 20 356 assertions,
  all green on M4.
- **Benchmark:** numbers in table above, measured on M4.
- **LayerNorm SSVE:** pending (to be contributed by colleague).

Sprint 2b — RMSNorm SSVE optimization ablation
-----------------------------------------------

**Goal:** identify the performance ceiling of the hand-written SSVE kernel and
attribute each potential gain to one isolated change.  Three variants (V1–V3)
are evaluated against the V0 baseline using a controlled ablation — one change
per variant — so every delta is attributable.

Process
~~~~~~~

All four kernels share the same two-pass column-major structure.  Each variant
introduces exactly one modification relative to the previous:

.. list-table:: Ablation variant definitions
   :header-rows: 1
   :widths: 12 30 40

   * - Variant
     - Change
     - Motivation
   * - **V0** (baseline)
     - ``FSQRT Z2.S`` + ``FMOV Z4.S, #1.0`` + ``FDIV Z4.S, P1/M, Z4.S, Z2.S``
     - Reference hand-written kernel; inv_rms costs ~24 cycles (sqrt + reciprocal).
   * - **V1**
     - Replace inv_rms with ``FRSQRTE`` + one Newton-Raphson step (``FRSQRTS``)
     - ``FRSQRTE`` estimates ``1/√x`` in one cycle; one NR refinement reaches
       full FP32 accuracy in ~12 cycles total — half the V0 latency.
   * - **V2**
     - Pre-compute ``1/N`` as a scalar ``FDIV`` once before the outer loop;
       broadcast to Z5; replace the per-block vector ``FDIV`` with ``FMUL Z2.S, P1/M, Z2.S, Z5.S``
     - The vector FDIV in V0/V1 runs once per block of VL rows, not once per
       element.  Moving it outside amortises a ~12-cycle divide over all M/VL
       iterations.
   * - **V3**
     - ×2 unroll of both column loops (peel one column when N is odd; main loop
       processes pairs with two back-to-back ``LD1W``/``FMLA`` or ``FMUL``/``ST1W``)
     - Halves the branch count; lets the out-of-order core overlap two
       independent memory streams per iteration.

The Newton-Raphson sequence for ``1/√a`` (V1 onward):

.. code-block:: asm

    frsqrte z4.s, z2.s          // estimate: e ≈ 1/√a  (one cycle)
    fmul    z3.s, z4.s, z2.s   // t = e·a
    frsqrts z3.s, z4.s, z3.s   // correction: c = (3 − e·t) / 2
    fmul    z4.s, p1/m, z4.s, z3.s  // refined: inv_rms = e·c

``FRSQRTE`` and ``FRSQRTS`` are unpredicated; the ``SEL`` before them fills
inactive lanes with ``eps`` so no ``+inf`` or NaN propagates to active output
lanes.

Expectations
~~~~~~~~~~~~

Before measuring, the predicted gains were:

- **V1:** 15–25 % — inv_rms sits on the critical path between pass 1 and pass 2;
  halving its latency should be visible for small M where few outer iterations
  amortise the latency.
- **V2:** 5–10 % — FDIV fires once per block, not per element; moving it out of
  the loop saves a small but real cost, especially at large M.
- **V3:** 5–10 % — branch overhead is measurable for small N; unrolling should
  help the prefetcher overlap two column streams.

Findings
~~~~~~~~

**Roofline: 79.58 GiB/s** (STREAM-style probe, re-measured for this run).

.. list-table:: Sprint 2b — RMSNorm SSVE ablation (Apple M4, SVL = 512 bits)
   :header-rows: 1
   :widths: 22 10 10 14 12 14

   * - Variant
     - M
     - N
     - GiB/s
     - % of peak
     - vs V0
   * - V0 (FSQRT+FDIV)
     - 128
     - 64
     - 17.77
     - 22.3 %
     - baseline
   * - V1 (FRSQRTE+NR)
     - 128
     - 64
     - 17.95
     - 22.6 %
     - +1.0 %
   * - V2 (V1 + inv_N)
     - 128
     - 64
     - 17.70
     - 22.2 %
     - −0.4 %
   * - V3 (V2 + unroll-2)
     - 128
     - 64
     - 17.78
     - 22.3 %
     - +0.1 %
   * - V0 (FSQRT+FDIV)
     - 128
     - 2048
     - 25.11
     - 31.6 %
     - baseline
   * - V1 (FRSQRTE+NR)
     - 128
     - 2048
     - 26.74
     - 33.6 %
     - **+6.5 %**
   * - V2 (V1 + inv_N)
     - 128
     - 2048
     - 25.12
     - 31.6 %
     - +0.1 %
   * - V3 (V2 + unroll-2)
     - 128
     - 2048
     - 24.96
     - 31.4 %
     - −0.6 %
   * - V0 (FSQRT+FDIV)
     - 1024
     - 2048
     - 24.75
     - 31.1 %
     - baseline
   * - V1 (FRSQRTE+NR)
     - 1024
     - 2048
     - 24.98
     - 31.4 %
     - +0.9 %
   * - V2 (V1 + inv_N)
     - 1024
     - 2048
     - 25.01
     - 31.4 %
     - +1.0 %
   * - V3 (V2 + unroll-2)
     - 1024
     - 2048
     - 25.10
     - 31.6 %
     - +1.4 %

**Interpretation:**

- **V1 wins only at M=128, N=2048 (+6.5 %).** The inv_rms computation is on the
  critical path between pass 1 and pass 2.  At M=128 there are only 8 outer
  iterations (128 / VL=16), so the ~12-cycle latency saving is a meaningful
  fraction of total time.  At M=1024 (64 outer iterations) the same saving
  amortises to noise.

- **V1 shows no gain for small N (N=64).** With only 64 columns the column loops
  are very short; the kernel spends a larger fraction of time in setup and
  ``SMSTART``/``SMSTOP`` overhead.  inv_rms latency is a smaller share of total
  time.

- **V2 shows near-zero gain everywhere.** The vector FDIV in V0/V1 fires once per
  outer block (once per VL rows), not once per element.  For M=128, N=2048 it
  fires 8 times across 2048 column iterations — already well amortised.  Moving
  it out of the loop saves almost nothing; the added setup (SMSTART, eps reload,
  1/N computation) partially cancels the saving.

- **V3 shows no gain.** The M4's hardware prefetcher and out-of-order execution
  already overlap sequential column loads in the non-unrolled loops.  The extra
  branch logic for the odd-N peel and the more complex loop body add marginal
  overhead.

- **Structural ceiling:** all variants plateau at **~31–34 % of the 79.58 GiB/s
  vectorised peak**.  The bottleneck is not inv_rms computation, nor branch
  overhead — it is the **two-pass column-major loop pattern** itself.  Both
  passes traverse all N columns sequentially; the memory system cannot hide the
  second read of the A matrix.  Closing the gap requires pass fusion (keeping the
  row resident in registers or the ZA accumulator between the two passes), which
  is the target of Sprint 3's JIT/ZA-tiled kernel.

**Key implementation lesson — SMSTART zeroes all Z registers:**
``SMSTART SM`` on the Apple M4 zeroes Z0–Z31 (and therefore D0–D15) on entry to
streaming mode.  Any value held in a Z/D register before ``SMSTART`` is lost.
This had two consequences during development (see the debug log):

1. ``eps`` was saved in S8 (a scalar alias of Z8) before ``SMSTART``; after
   ``SMSTART`` it read as 0.0 → the inv_rms denominator became 0 → ``FRSQRTE(0) = +∞``
   → NaN output.  Fix: reload ``eps`` from its D8 stack slot immediately after
   ``SMSTART`` and re-broadcast into Z8.
2. The ``1/N`` pre-computation in V2/V3 was done before ``SMSTART`` into S0 (Z0.S[0]);
   after ``SMSTART``, the ``DUP Z5.S, Z0.S[0]`` broadcast 0.  Fix: move the scalar
   ``FDIV`` and ``DUP`` to after ``SMSTART``.

Sprint 2b status
~~~~~~~~~~~~~~~~

- **Kernels:** ``src/norm/rms_norm_ssve_v1.S``, ``v2.S``, ``v3.S`` — all
  correct, verified, and committed.
- **Tests:** 10 ablation Catch2 cases (tagged ``[sprint2][ablation]``), all green
  on M4; skip on CI.
- **Benchmark:** ablation table printed by ``./build/apps/main_norm``; all four
  variants run in sequence under a single streaming region per shape.
- **Best result:** 26.74 GiB/s (V1, M=128, N=2048) = 33.6 % of vectorised peak.
- **LayerNorm SSVE ablation:** to be done after the LayerNorm V0 kernel is written.

Sprint 2a — Roofline validation
--------------------------------

**Goal:** validate the peak the V0–V3 percentages were judged against.  Every
"% of peak" so far used a single 79.58 GiB/s figure; before optimizing
further, establish *what* that number measures — and what the right ceiling
for a single-threaded streaming-mode kernel actually is.

What the old "peak" measured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disassembly of the benchmark binary answers the first question: the C++
STREAM probe (``bw_scale_add``) autovectorizes to **NEON** (``ldp q`` pairs +
``fadd.4s`` + ``stp q``, 64 bytes per iteration) and runs **single-threaded**.
So 79.58 GiB/s was already a *single-core* figure — but a single-core **NEON**
figure.  The norm kernels execute in a different mode entirely: streaming SVE
inside ``SMSTART``/``SMSTOP``, with ``LD1W``/``ST1W``.  Nothing guarantees
those two modes see the same per-core bandwidth, so a second probe was needed.

The streaming-mode probe
~~~~~~~~~~~~~~~~~~~~~~~~

``src/norm/bw_probe_ssve.S`` re-runs the same scale-add
(``d[i] = s[i] + 1.0f``, 128 MiB arrays) the way the kernels access memory:
one streaming region, contiguous ``LD1W``/``ST1W``, four independent
load/add/store chains per iteration, predicated VLA tail (no hard-coded SVL).
It is verified bit-exactly against the scalar loop by two Catch2 cases
(``[sprint2][roofline]``) before any bandwidth number derived from it is
trusted.  The benchmark process requests ``QOS_CLASS_USER_INTERACTIVE``
(macOS has no core-pinning API; this is the strongest P-core scheduling
hint), and a third, multi-threaded probe measures the chip-wide aggregate.

The three ceilings (Apple M4, best-of-10, 128 MiB arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Ceiling
     - GiB/s
     - Role
   * - single-core NEON (compiler-vectorized)
     - 79.5
     - what the old "peak" was
   * - **single-core SSVE streaming (LD1W/ST1W)**
     - **59.5**
     - **the kernel roofline**
   * - chip-wide (10 threads, NEON)
     - 89.8
     - Sprint-5 threading target

Two structural facts fall out:

- **Streaming mode has ~25 % less single-core bandwidth than NEON mode** on
  the M4 (59.5 vs 79.5 GiB/s).  The kernels were being judged against a
  ceiling they cannot physically reach in their execution mode.
- **One core already sustains ~66 % of the chip-wide aggregate** (59.5 of
  89.8 GiB/s).  Threading across rows (Sprint 5) can buy at most ~1.5x, not
  10x — the M4's memory system saturates with very few cores.

Byte-counting convention (now stated explicitly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All GiB/s figures — kernels *and* probes — count **useful bytes**:
1 read + 1 write per element, the algorithm's minimum traffic.  The V0–V3
kernels as implemented are two-pass over memory (reduction reads x, normalize
reads x again, then writes y = 2R+1W), so their *moved*-bytes figure is 1.5x
the printed one.  Keeping the useful-bytes convention makes the two-pass cost
visible as a lower % of peak — which is exactly the gap the residency lever
(V6) attacks, rather than a number to hide.

V0–V3 restated against the validated roofline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Against the 59.5 GiB/s single-core streaming ceiling, the picture improves
from the interim "31–34 %" (which used the NEON figure):

- large-N shapes (N ≥ 512): **42–46 % of the kernel roofline** (best ~27 GiB/s);
  in moved-bytes terms the kernel sustains ~40 GiB/s ≈ **67 % of the ceiling**
- small-N shapes (N = 64): 31–43 % depending on M (see below)

Small-N regime: overhead quantified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeping N at fixed M with the V1 kernel and fitting
:math:`t(N) = t_0 + b \cdot N`:

- **M=128:** the fit is clean — fixed cost :math:`t_0 \approx 1` µs/call
  (streaming entry, prologue, 8 per-row-block setups + ``inv_rms``
  serializations), equal to the streaming work at **N ≈ 30–40**, asymptote
  ~27.5 GiB/s.  The N=64 dip is fully explained by fixed overhead, not by a
  different memory regime.  (The deeper ``SMSTART``/``SMSTOP`` cost study
  stays in Sprint 7.)
- **M=1024:** the linear model is *invalid* (negative intercept) — throughput
  is not constant in N.  At N=4096 the working set (x plus y, 32 MiB) no
  longer fits in cache and throughput falls from ~25 to ~18.6 GiB/s.  *The
  provisional interpretation here ("the normalize pass re-reads x from DRAM")
  was tested and corrected by the Sprint-2b diagnostic below: pass-2
  residency was never the problem — the falloff marks the true-DRAM regime,
  where access density is the binding constraint.*

Decision gate: V4–V6 proceed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gate asked: is V1 already ≳80–85 % of the single-core roofline?  **No —
best case is ~46 % (useful bytes) / ~67 % (moved bytes).**  Real single-core
headroom remains, from two attributable sources: memory-level parallelism in
the strided two-pass loop (V4 multi-accumulator, V5 load pipelining) and the
2R+1W → 1R+1W traffic reduction (V6 residency), whose payoff the M=1024/N=4096
falloff bounds directly.  V4–V6 are therefore worth pursuing before the ZA
prototype (Sprint 3).

Sprint 2a status
~~~~~~~~~~~~~~~~

- **Probe:** ``bw_probe_ssve.S`` written, verified bit-exact, committed.
- **Benchmark:** ``main_norm`` reports all three ceilings, states the byte
  convention in its header, computes every % against the single-core SSVE
  ceiling, and prints the small-N sweep with the overhead fit (flagging the
  cache-boundary regime where the fit is invalid).
- **Reproducibility:** ceilings stable to ±0.3 % across runs; the
  M=1024/N=4096 falloff reproduces.

Sprint 2b round two — the memory-behaviour levers (V4, V5, V6)
---------------------------------------------------------------

**Goal:** with the 2a gate passed (~46 % of the validated roofline), test the
three levers that change memory behaviour rather than arithmetic.  Each
hypothesis was written down before measuring; each variant is verified
against the reference before any number is trusted.

V4 — multi-accumulator reduction ILP: **the ILP win**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Hypothesis:* V0–V3 accumulate the sum of squares into a single Z register,
so every ``FMLA`` waits ~3–4 cycles on the previous one, capping how many
column loads can be in flight.  Four independent accumulator chains
(``Z2/Z16/Z17/Z18``, four columns per iteration, combined with two ``FADD``
at the end) should unblock the load stream.

*Result:* **+27–32 % vs V0** at N ≥ 2048 (best 33 GiB/s ≈ 54–56 % of the
roofline), +18 % even at N=64.  This also explains V3's zero: unrolling
without breaking the chain does nothing — the chain *was* the bottleneck.

*Numerics note:* four partial sums combine in tree order, not the
reference's sequential order.  Worst observed deviation 1.1e-5 relative (at
N=4, where each accumulator holds one term), marginally over the 1e-5 gate.
V4/V5 are verified at the honest tolerance they meet (``kTolReassoc =
2e-5``), documented in the test file rather than silently widening the gate
for all kernels.  V6 removes the issue entirely (below).

V5 — explicit load software-pipelining: **≈ 0, as predicted**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Hypothesis (pre-registered):* ~0 gain.  The M4's out-of-order rename should
already hoist loads past the FMLAs once V4 broke the dependency chain.

*Result:* within noise of V4 on every shape (rotating A/B load groups,
loads issued one 4-column group ahead in program order).  The ≈0 closes the
"more outstanding loads" family: explicit ``PRFW`` prefetch works the same
lever one step earlier and gets a reasoned skip rather than its own kernel.

The diagnostic that redefined V6: density, not residency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROADMAP's V6 premise was "make pass 2 re-read x from cache, not DRAM."
Before building it, the premise was checked (measure before optimizing):
the reuse distance between pass 1 and pass 2 of one VL-row block is only
``64*N`` bytes (~128–256 KB at bench shapes) — **already L2-resident by
construction**.  A footprint sweep with V4 found what actually binds:

.. list-table:: V4, fixed N=2048, growing footprint (Apple M4)
   :header-rows: 1

   * - M
     - footprint (x+y)
     - GiB/s (useful bytes)
   * - 512
     - 8 MB
     - 36.5
   * - 1024
     - 16 MB
     - 31.1
   * - 2048
     - 32 MB
     - 18.3
   * - 4096
     - 64 MB
     - 11.2

Throughput collapses once the footprint crosses the ~16 MB L2 — at *fixed*
N — so the Sprint-2a N=4096 falloff was the **true-DRAM regime** being
reached (smaller shapes stay partially cache-resident *across benchmark
repetitions*), not pass-2 re-reads.  And at equal 32 MB footprint the aspect
ratio decides: M=512/N=8192 runs 31 GiB/s while M=2048/N=2048 runs 18 —
each column touch reads 64 B out of every ``4*M`` bytes, and the sparser
that strided walk, the worse the DRAM/prefetch efficiency.  **The remaining
lever is access density, not residency.**

V6 — 4-row-block contiguity grouping: **the DRAM-regime win**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Design:* process four consecutive VL-row blocks per outer iteration, so
every column touch is four back-to-back ``LD1W`` (``[x8, #k, MUL VL]``) =
**256 B contiguous**, 4x denser.  The four accumulators become one *per row
block* — still four independent FMLA chains (keeps V4's ILP win), but each
row now sums sequentially like the scalar reference, so V6 is verified at
the strict 1e-5 tolerance.  ``inv_rms`` is computed for four blocks per
group (the planned V6b batching, folded in), and pass 2 shares each
``LD1RW`` gamma broadcast across the four blocks.  Rows beyond the last
full group take a predicated single-block tail.

*Result (vs the 59.5 GiB/s single-core streaming roofline):*

.. list-table:: RMSNorm SSVE ablation, key rows (Apple M4)
   :header-rows: 1

   * - shape
     - V0
     - V4
     - **V6**
     - V6 % roofline
   * - M=128, N=2048 (2 MB)
     - 25.1
     - 33.1
     - 32.1
     - 54 %
   * - M=1024, N=2048 (16 MB)
     - 24.6
     - 30.3
     - **37.7**
     - **63 %**
   * - M=4096, N=2048 (64 MB, true DRAM)
     - 11.1
     - 11.3
     - **25.6**
     - 43 %

In the true-DRAM regime V6 is **+131 % vs V0 / +126 % vs V4** — the access-
density mechanism confirmed.  At M=128 it ties V4 (stride is only 512 B
there; density was never the constraint), and it is never worse anywhere →
**V6 is the final Sprint-2 incumbent**.  In moved-bytes terms (x1.5) the
M=1024 figure is ~57 GiB/s ≈ **95 % of the streaming ceiling**: for
cache-assisted shapes the two-pass SSVE kernel is essentially saturating
its execution mode.

Final Sprint-2b RMSNorm conclusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Best variant: V6** — 63 % of the single-core streaming roofline (useful
  bytes) at 16 MB shapes, ~95 % in moved bytes; 43 % in the 64 MB true-DRAM
  regime.
- **Exhausted levers:** ``inv_rms`` arithmetic (V1/V2: ≤ +6 %), branch
  overhead (V3: 0), load scheduling (V5: 0, OOO covers it), accumulator ILP
  (V4: +27–32 %, absorbed into V6), per-block residency (already present by
  construction — measured, not assumed).
- **Deliberately deferred:** deeper/parameterized group depth (8–16 blocks
  would further densify huge-M shapes — a natural *shape-specialized JIT
  emission decision* for Sprint 4); ZA-tile staging (Sprint 3) now has a
  hard target: beat V6's 256 B-contiguity with 2D tile movement, in the
  regimes where V6 still trails the roofline; threading across rows
  (Sprint 5) toward the 89.8 GiB/s chip ceiling.

Sprint 2b status
~~~~~~~~~~~~~~~~~

- **Kernels:** ``rms_norm_ssve_v4.S``, ``v5.S``, ``v6.S`` — verified (V4/V5
  at documented ``kTolReassoc``, V6 at strict ``kTol``), committed.
- **Tests:** 40 Catch2 cases, 50 846 assertions, all green on M4; skip on CI.
- **Benchmark:** V0–V6 ablation incl. a 64 MB DRAM-regime shape; footprint
  diagnostic recorded above.

Sprint 2c — Hand-Written SSVE LayerNorm Kernel
----------------------------------------------

**Goal:** implement LayerNorm as a hand-written AArch64 SSVE kernel in
``src/norm/layer_norm_ssve.S``, verify it numerically against the Sprint 1
reference across 6 test cases, and integrate into the shared build on branch
``feat/sprint2-layernorm-ssve``.

This sprint is done by **Mariza**; RMSNorm SSVE and the Sprint-2 ablation are
handled separately by **Ketsia** (see sections above).

Key design decisions
~~~~~~~~~~~~~~~~~~~~

**SME1 mandatory streaming SVE subset.**
Not all SVE instructions are available in streaming mode (``PSTATE.SM = 1``).
The most important restriction: *scatter-gather loads*
(``LD1W {z.s}, p, [Xn, Zm.S, UXTW #2]``) and integer vector multiply
(``MUL Zdn.S, Pg/M, ...``) are absent from the SME1 mandatory subset and
raise SIGILL at runtime on the M4.

**Vectorise across rows, loop over columns.**
In column-major layout a *column* is contiguous in memory.  Rather than
loading a row (which needs a gather), the kernel loads one column slice of
SVL rows at a time using a plain contiguous ``LD1W``:

.. code-block:: text

    outer loop: row chunks of SVL rows
      pass 1: for each column → LD1W SVL rows → accumulate per-row sum z8
              → FDIV z8/n → per-row mean z8
      pass 2: for each column → LD1W, FSUB mean, FMUL square, accumulate z9
              → FDIV z9/n, FADD eps, FSQRT, FDIV 1/sqrt → per-row inv_std z5
      pass 3: for each column → LD1W, FSUB, FMUL inv_std
              → LD1RW gamma[col], FMUL; LD1RW beta[col], FADD; ST1W

This is a *three-pass* implementation in terms of memory reads (the
literature's "two-pass" label counts passes over the row, not reads of the A
matrix).  Pass 1 reads x for the mean; pass 2 reads x again for variance;
pass 3 reads x a third time to normalise.  The additional gamma/beta reads
are L1-resident (N-element vectors, typically ≤ 8 KiB).

**Vector length agnostic (VLA).**
``CNTW x8`` queries SVL at runtime.  ``WHILELO p1.s, x_rbase, x_m``
generates the predicate for the tail row chunk (active lane j active iff
r\_base + j < m), so the kernel is correct for any SVL — not just the
16-lane 512-bit SVL of the M4.

**Streaming mode boundary.**
Epsilon arrives in ``s0`` (the first FP argument register) before
``SMSTART SM`` would alter FP state; it is spilled to the stack frame and
reloaded after entering streaming mode.  The callee-saved registers x19/x20
hold the chunk base pointers and are saved/restored around the streaming
region.

``-march=armv9-a+sme`` scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SME march flag is applied only to the assembly source (and to
``norm_ssve.cpp``, which has no auto-vectorizable loops), NOT to
``reference.cpp``:

.. code-block:: cmake

    set_source_files_properties(
        norm_ssve.cpp layer_norm_ssve.S rms_norm_ssve.S ...
        PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme")

Applying the flag to ``reference.cpp`` would cause the compiler to
auto-vectorize with regular (non-streaming) SVE, which SIGILLs on Apple
Silicon because non-streaming SVE is absent on M-series chips.

C++ namespace bridge
~~~~~~~~~~~~~~~~~~~~

The assembly exports a plain C symbol ``layer_norm_ssve``.
``src/norm/norm_ssve.cpp`` (the shared SSVE bridge) declares it with
``extern "C"``, wraps it in the ``mini_jit::norm`` namespace, and guards
the call with ``if (!cpu_supports_sme()) return;`` so the function is a
safe no-op on CI runners without SME, matching the pattern used for all
RMSNorm variants.

Verification test cases
~~~~~~~~~~~~~~~~~~~~~~~

Six new Catch2 test cases in ``tests/test_norm.cpp``, all tagged
``[sprint2c][ssve][layernorm]`` and guarded with
``if (!cpu_supports_sme()) SKIP("SME required")``:

.. list-table::
   :header-rows: 1

   * - Case
     - Shape / condition
     - What it exercises
   * - 1
     - M=8, N=4 (N < SVL)
     - tail-only path
   * - 2
     - M=8, N=16 (N = SVL)
     - single full vector, no tail
   * - 3
     - M=6, N=19 (N = SVL + 3)
     - full vector + small tail
   * - 4
     - M=8, N=37 (N = 2·SVL + 5)
     - multiple vectors + tail, outer row loop
   * - 5
     - M=5, N=19, ld_a=ld_b=8 (ld > M)
     - padded matrix, stride test
   * - 6
     - M=4, N=37, SHIFT=1e4
     - large-magnitude stability

Cases 1–5 use relative tolerance ``kTol = 1e-5``.
Case 6 uses an absolute margin of ``1e-3``: with a DC offset of 1×10⁴ and
differences of ±5, catastrophic cancellation limits FP32 accuracy to roughly
``ε_float × SHIFT × inv_std ≈ 5×10⁻⁴``; a relative tolerance would be
unfairly strict for elements whose normalised value is near zero.

GiB/s results — V0 baseline (Apple M4, SVL = 512 bits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Roofline:** 59.45 GiB/s single-core streaming (LD1W/ST1W probe).

.. list-table:: Sprint 2c — layer_norm_ssve V0 vs reference (Apple M4)
   :header-rows: 1
   :widths: 10 10 20 20 16 12

   * - M (rows)
     - N (features)
     - ``layer_norm_ref`` GiB/s
     - ``layer_norm_ssve`` GiB/s
     - % of SSVE peak
     - Speedup
   * - 128
     - 64
     - 4.09
     - 9.64
     - 16.2 %
     - 2.4×
   * - 128
     - 512
     - 2.41
     - 12.37
     - 20.8 %
     - 5.1×
   * - 128
     - 2048
     - 2.24
     - 12.39
     - 20.8 %
     - 5.5×
   * - 1024
     - 64
     - 1.23
     - 12.03
     - 20.2 %
     - 9.8×
   * - 1024
     - 512
     - 1.22
     - 12.38
     - 20.8 %
     - 10.1×
   * - 1024
     - 2048
     - 0.95
     - 12.76
     - **21.5 %**
     - **13.4×**

The kernel plateaus at **~20–21 % of the 59.5 GiB/s SSVE roofline** (useful
bytes, 1R+1W convention).  Compare to RMSNorm V0's ~42 %.  The factor of ~2
is fully explained by the three-pass memory access pattern: LayerNorm reads x
three times (mean, variance, normalize), so its moved-bytes traffic is
**3R+1W** vs RMSNorm's **2R+1W**.  Restated in moved-bytes terms:

- LayerNorm V0: 12–13 GiB/s useful → ×2 = **~25 GiB/s moved** ≈ **42 % of roofline**
- RMSNorm V0:   25 GiB/s useful → ×1.5 = **~37 GiB/s moved** ≈ **63 % of roofline**

The kernel quality is similar; the structural cost is the extra read pass.
This is exactly what the residency lever (the LayerNorm-specific V6) targets:
fusing passes 1 and 2 drops the traffic from 3R+1W to 2R+1W — a potential
**+50 % in useful-bytes throughput** before any other optimisation.

Sprint 2c status
~~~~~~~~~~~~~~~~

- **Build:** clean; ``layer_norm_ssve.S`` assembles under ``-march=armv9-a+sme``
- **Test:** 6 Sprint-2c cases pass on M4 (46 total with RMSNorm suite); CI skips gracefully
- **Kernel:** ``src/norm/layer_norm_ssve.S`` — VLA, SME1 mandatory subset only,
  three passes (mean / variance / normalise), column-major contiguous loads
- **Benchmark:** V0 baseline measured; 12–13 GiB/s (20–22 % of SSVE roofline)
- **Ablation:** V1–V6 + Welford — see the next section

Sprint 2c ablation — LayerNorm V1–V6 and the Welford verdict
-------------------------------------------------------------

**Goal:** replay the RMSNorm levers on LayerNorm's three-pass structure
(characterized separately, not assumed to transfer) and run the algorithmic
ablation the ROADMAP called the headline row: vectorized Welford single-pass
vs the stable two-pass.  All variants verified on M4 first — including
large-magnitude stress inputs for V4/V5/V6/Welford (added as gap-fill; a
Welford speed verdict is only meaningful if its accuracy actually holds).

Ablation results (59.4 GiB/s single-core streaming roofline, useful bytes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: LayerNorm SSVE ablation, key rows (Apple M4)
   :header-rows: 1

   * - shape
     - V0
     - V2
     - V4
     - **V6**
     - Welford
   * - M=128, N=2048 (2 MB)
     - 12.0
     - 13.4
     - 15.2
     - **15.2**
     - 11.7
   * - M=1024, N=2048 (16 MB)
     - 12.7
     - 14.3
     - 16.3
     - **16.3**
     - 12.1
   * - M=4096, N=2048 (64 MB, true DRAM)
     - 8.4
     - 8.3
     - 8.4
     - **14.4**
     - 8.1

Per-lever verdicts (vs the RMSNorm outcomes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **V1 (FRSQRTE+NR for inv_std): ≈0** (−2.6 % to +2.9 %).  Same verdict as
  RMSNorm: the reciprocal-sqrt latency is not on the critical path.
- **V2 (pre-computed 1/N): +9–12 %** — *much bigger than RMSNorm's ≈0*, and
  the clearest structure-dependent difference.  LayerNorm has **two**
  per-block vector FDIVs (mean = Σx/N and variance = M2/N), and both sit at
  serialization points *between* passes: mean gates pass 2, variance gates
  ``inv_std`` and pass 3.  With three shorter passes over the same N, these
  serial bubbles are a ~3x larger share of runtime than RMSNorm's single
  once-per-block FDIV.  Hoisting them pays here; it didn't there.
- **V4 (4-accumulator ILP): +26–28 %** at N=2048 — transfers cleanly from
  RMSNorm (which saw +27–32 %).  Accumulating two statistics does not blunt
  the lever.
- **V5 (load pipelining): ≈0 vs V4** — transfers: OOO rename already covers
  it once the chains are broken.
- **V6 (4-row-block contiguity, three passes kept): ties V4** at
  cache-assisted shapes and **+71 % vs V0 in the true-DRAM regime**
  (8.4 → 14.4 GiB/s) — the density mechanism from Sprint 2b transfers, with
  a smaller DRAM-regime multiple than RMSNorm's +131 % because LayerNorm
  spends relatively more time in FP work per byte.  **V6 is the LayerNorm
  incumbent** (never worse than V4 anywhere).

Welford vs two-pass: two-pass dominates on BOTH axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The comparison production libraries actually face, measured:

- **Speed: Welford loses everywhere** — −2.5 % to −8.5 % vs V0, and
  25–30 % below V6 — *despite moving 25 % fewer bytes* (2R+1W vs 3R+1W).
  The pre-registered hypothesis held: the per-element recurrence
  (``mean += delta/count``) serializes on SIMD, and the FDIV-per-column cost
  swamps the traffic saving.
- **Accuracy: Welford is ~100x worse on shifted data in FP32.**  Max
  absolute error vs the double reference (M=16, N=512):

  .. list-table::
     :header-rows: 1

     * - input shift
       - two-pass V6
       - Welford (FP32)
     * - 0
       - 1.2e-5
       - 1.1e-5
     * - 1e4
       - **1.2e-5**
       - **1.2e-3**
     * - 1e5
       - 2.5e-3
       - 2.5e-3

  The centered two-pass subtracts the mean *before* squaring, so its
  variance arithmetic works on small values; FP32 Welford updates
  ``mean`` at full input magnitude every element, accumulating rounding at
  the scale of the shift.  (Welford is still far better than the
  catastrophic naive ``E[x²]−µ²`` — but it is not a free replacement for
  two-pass in FP32.)  At shift 1e5 both degrade equally: the input itself
  has lost the bits.

**Verdict:** on SIMD, the stable two-pass formulation wins on speed *and*
FP32 accuracy.  Welford's ablation row is a strong negative — kept in the
table with both measurements, not just the throughput number.

LayerNorm vs RMSNorm on the same harness (decision C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Best variant vs best variant (both V6), same shapes:

.. list-table::
   :header-rows: 1

   * - shape
     - LN V6 GiB/s
     - RMS V6 GiB/s
     - LN/RMS
   * - M=128, N=64
     - 11.4
     - 22.2
     - 0.52
   * - M=128, N=2048
     - 15.2
     - 31.8
     - 0.48
   * - M=1024, N=2048
     - 16.3
     - 37.7
     - 0.43
   * - M=4096, N=2048
     - 14.4
     - 25.5
     - 0.56

**RMSNorm is 1.8–2.3x faster — well beyond the proposal's "10–40 %".**
Two attributable causes: (1) structural traffic — LayerNorm's three passes
move 4 units (3R+1W) per element vs RMSNorm's 3 (2R+1W), a 1.33x floor;
(2) per-byte efficiency — LayerNorm adds the mean-subtract and β, and its
two serialization points per block (mean, then variance) cost more than
RMSNorm's one.  In moved-bytes terms LN V6 sustains ~33 GiB/s (55 % of the
roofline) vs RMS V6's ~57 (95 %): the remaining LayerNorm-specific gap is
the pass structure itself.

What is deliberately left on the table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

True 3R→2R pass fusion needs the variance pass's data kept resident until
``inv_std`` is known — more state than 32 Z registers hold for real N.
That is exactly tile-level staging, i.e. the **Sprint-3 ZA hypothesis**,
now with a quantified target: +33 % traffic reduction available on
LayerNorm (vs none on RMSNorm), so LayerNorm is where ZA staging has
structural headroom.  Deeper group depth (8–16 blocks) remains a
shape-specialized JIT decision (Sprint 4), threading toward the 88–90
GiB/s chip ceiling is Sprint 5.

Sprint 2c ablation status
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Kernels:** ``layer_norm_ssve_v1/v2/v4/v5/v6/welford.S`` — all verified
  on M4 (81 test cases, 64 885 assertions in ``test_norm``), stress
  coverage complete for every variant.
- **Incumbents frozen for Sprint 3/4:** RMSNorm V6 and LayerNorm V6.
- **Decision-C numbers recorded:** LN/RMS = 0.43–0.56 on identical shapes.

Sprint 3 — Hand-written SME/ZA RMSNorm kernel (``rms_norm_za``)
---------------------------------------------------------------

The second kernel *architecture* for RMSNorm: the same math as the SSVE
winner V6, but staged through the SME **ZA** array instead of streamed
twice. The goal is not a new speed record — Sprint 2a already put V6 near
the single-core streaming ceiling — but to attribute exactly what the ZA
tile does, and does not, buy a bandwidth-bound horizontal reduction
(context.md §5; ROADMAP Sprint 3 "honest expectation-setting").

The hypothesis, written before measuring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZA does not raise DRAM bandwidth. RMSNorm must see a row's **entire**
sum-of-squares before it can normalize, so V6 re-reads ``x`` in pass 2
(2R+1W). That second read is only avoidable if the whole row-block stays
resident on core between the two passes. ZA gives ~4 KB of extra on-core
storage — four 32-bit tiles ZA0..ZA3, each ``SVL_S × SVL_S`` = 16×16 FP32
on the M4 — beyond the 32 Z-registers, exactly enough to hold one
``SVL``-row × (≤ 4·SVL)-col block. So the *only* lever available is:
stage ``x`` in ZA during the reduction, read it back from ZA in pass 2,
and turn 2R+1W into **1R+1W** — but only where the row fits
(``N ≤ 4·SVL`` = 64 on the M4). That is precisely the small-N,
streaming-overhead-dominated regime, so the pre-registered expectation was:
**a possible small-N win if the saved read outweighs the cost of shuffling
``x`` through ZA, and no win at all for the large-N bandwidth-bound shapes
(row does not fit → fall back to streaming → tie/lose V6).**

Kernel design (``rms_norm_za.S``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reduction stays SSVE.** The per-row sum-of-squares is a horizontal
  reduction (``fmla`` accumulating ``x*x`` in a Z-register), never routed
  through ZA — a matrix accumulator maps poorly onto a horizontal collapse
  (context.md §5). ZA is used as **pure residency staging**, not compute.
- **Pass 1 (1 memory read):** for each column, ``ld1w`` the ``x`` vector
  (SVL consecutive rows, column-major), ``fmla`` it into the accumulator,
  and ``mova`` it into a ZA tile slice.
- **inv_rms:** ``1/√(sumsq/N + ε)`` per lane, via ``FRSQRTE`` + one
  Newton-Raphson step (same sequence as V1/V6).
- **Pass 2 (1 memory write):** for each column, ``mova`` ``x`` back **from
  ZA**, apply ``γ[col]`` and ``inv_rms``, ``st1w``.
- **VLA (decision D):** tile geometry is derived from ``cntw`` (SVL), not a
  literal 16; the path split is ``N`` vs ``4·SVL`` computed at runtime. The
  four ZA tiles are emitted as four ``.macro`` expansions (the tile number
  is an instruction immediate, not a register, so it cannot be looped).
- **PSTATE handling:** bare ``smstart``/``smstop`` enable and disable
  **both** PSTATE.SM and PSTATE.ZA for the whole norm (one streaming
  region); every ZA slice is written before it is read, so no ``zero {za}``
  is needed, and ZA is disabled before return (no lazy-save/AAPCS64 hazard).
- **Fallback (``N > 4·SVL``):** a correct streaming two-pass (2R+1W, no ZA)
  so the kernel is right for every shape. This is not the performance
  story — V6 remains the large-N incumbent.
- A single accumulator sums in strict column order (reference order), so no
  reassociation tolerance is needed: verified at ``kTol = 1e-5`` (86 test
  cases in ``test_norm``, tile-boundary / row-tail / mismatched-ld /
  fallback / large-magnitude-stress cases all pass on M4).

Measured: ZA residency vs V6 (Apple M4, useful bytes = 1R+1W)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Sprint 3 — ``rms_norm_za`` vs ``rms_norm_ssve_v6``
   :header-rows: 1

   * - shape
     - path
     - V6 GiB/s
     - ZA GiB/s
     - ZA vs V6
   * - M=128, N=16
     - ZA (1 tile)
     - 11.5
     - 6.8
     - **−41 %**
   * - M=128, N=32
     - ZA (2 tiles)
     - 16.3
     - 8.6
     - **−47 %**
   * - M=128, N=64
     - ZA (4 tiles, max residency)
     - 21.5
     - 8.4
     - **−61 %**
   * - M=1024, N=64
     - ZA (4 tiles)
     - 30.2
     - 10.9
     - **−64 %**
   * - M=4096, N=64
     - ZA (4 tiles)
     - 31.1
     - 11.0
     - **−65 %**
   * - M=128, N=2048
     - fallback (streaming)
     - 34.8
     - 25.1
     - −28 %
   * - M=1024, N=2048
     - fallback (streaming)
     - 38.0
     - 24.7
     - −35 %

Verdict: ZA loses decisively, and the reason is instructive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZA-resident fast path is **39–65 % slower than V6**, even though it
moves strictly *less* memory (1R+1W vs 2R+1W — a real 33 % traffic
reduction on this regime). The clinching data point: the ZA kernel's own
streaming **fallback** at N=2048 (≈25 GiB/s) is *faster* than its
**ZA-resident** path at N=64 (8–11 GiB/s). Staging through ZA is the
bottleneck, not memory.

The mechanism: each element now costs two extra ``mova`` ops — one
``Z→ZA`` in pass 1, one ``ZA→Z`` in pass 2 — and ``mova`` to/from the ZA
array is throughput-limited on the M4. At the modest bandwidth these
shapes sustain (10–35 GiB/s), the memory system was never the binding
constraint that ZA relieves, so trading a DRAM read for two ``mova`` ops is
a net loss. This is the same lesson as context.md §5's reduction warning,
now confirmed for **residency staging** too: routing a bandwidth-bound
horizontal-reduction norm through the matrix array costs more than it saves.

Reconciling with the Sprint-2c note: the "+33 % traffic reduction
available" framing referred to the *fusion arithmetic*; the measurement
shows the reduction is only reachable in the ``N ≤ 4·SVL`` regime (the row
must fit in ZA), and even there the ``mova`` cost dominates, so the traffic
saving never converts into a time saving. For the large-N shapes that are
actually bandwidth-bound, ZA cannot hold the row at all — structurally out
of reach. "ZA adds nothing here, and here is why" is the pre-registered,
fully valid ablation outcome (ROADMAP Sprint 3).

- **RMSNorm architecture verdict:** SSVE **V6 stays the frozen incumbent**
  for Sprint 4's JIT. ``rms_norm_za`` is kept in the tree and the ablation
  table as a *measured, explained* negative — the honest-engineering
  deliverable, not a failure.
- **LayerNorm ZA (Mariza):** gate resolved by building the prototype rather
  than taking the reasoned skip — see below. Verdict: also negative, and by
  a *larger* margin than RMSNorm's.

Sprint 3 — Hand-written SME/ZA LayerNorm kernel (``layer_norm_za``)
---------------------------------------------------------------------

The gated second half of Sprint 3. RMSNorm's ZA verdict was negative, but
LayerNorm's 3R+1W structure has more theoretical residency headroom than
RMSNorm's 2R+1W (its *two* extra reads, not one, are what ZA staging could
eliminate), so the gate was resolved by measuring rather than skipping.

The hypothesis, written before measuring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RMSNorm's ``mova``-throughput finding (above) is norm-agnostic in principle,
but LayerNorm's larger headroom makes it worth a direct test rather than an
inference. Rather than the partial "3R→2R" fusion sketched in Sprint 2c
(fuse only the mean+variance sub-passes, still re-read ``x`` once for
normalize), this prototype goes further: ``x`` is staged in ZA **once**
during the mean pass and reused from ZA for **both** the variance pass and
the normalize pass — a true 3R+1W → 1R+1W fusion, a 50 % traffic cut versus
RMSNorm's 33 %. This is the most decisive version of the test available: it
costs *three* ``mova`` ops per element (one store into ZA, two loads back
out) against RMSNorm's two (one store, one load). Pre-registered expectation:
if a 33 % traffic cut already lost 39–65 % to ``mova`` throughput, a 50 % cut
needs a proportionally *bigger* saving to break even, and instead pays a
proportionally bigger ``mova`` tax. Expected outcome: also a loss, likely by
a *larger* margin than RMSNorm's — which would confirm ``mova`` throughput,
not DRAM bandwidth, as the real, norm-agnostic ceiling. A win would be the
surprising result.

Kernel design (``layer_norm_za.S``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reduction stays SSVE** (context.md §5): both the mean sum and the
  sum-of-squared-deviations are horizontal reductions (``fadd`` / ``fmla``
  accumulating in Z-registers), never routed through the ZA matrix
  accumulator. ZA is pure residency staging for ``x``.
- **Pass 1 (1 memory read):** ``ld1w`` each column's ``x``, accumulate into
  the mean sum, ``mova`` it into a ZA tile slice.
- **Pass 2 (0 memory reads):** ``mova`` ``x`` back **from ZA**, center on the
  mean, accumulate the squared deviation.
- **inv_std:** ``FRSQRTE`` + one Newton-Raphson step (same sequence as V1/V6).
- **Pass 3 (1 memory write):** ``mova`` ``x`` back **from ZA a second time**,
  apply ``(x-mean)*inv_std*gamma[col]+beta[col]``, ``st1w``.
- **VLA (decision D):** tile geometry derived from ``cntw`` (SVL); the path
  split is ``N`` vs ``4·SVL`` computed at runtime, same four-tile ZA
  layout as ``rms_norm_za``.
- **PSTATE handling:** bare ``smstart``/``smstop`` for the whole norm (one
  streaming region), every ZA slice written before read, ZA disabled before
  return.
- **Fallback (``N > 4·SVL``):** a correct streaming three-pass (3R+1W, no
  ZA, single-block predicated — not the 4-block-contiguity-grouped structure
  V6 uses, since this fallback is a correctness net, not the performance
  story).
- Single accumulators (mean, variance) run in strict column order across all
  four tile sections, matching the reference's summation order exactly —
  verified at ``kTol``/``kAbsMarginNR``, no reassociation widening needed.
  Verified vs ``layer_norm_ref`` (tile boundaries, row tails, mismatched
  leading dims, the ``N > 4·SVL`` fallback, and the large-magnitude stress
  case; 5 new test cases in ``test_norm``, all passing on M4).

Measured: ZA residency vs V6 (Apple M4, useful bytes = 1R+1W)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Sprint 3 — ``layer_norm_za`` vs ``layer_norm_ssve_v6``
   :header-rows: 1

   * - shape
     - path
     - V6 GiB/s
     - ZA GiB/s
     - ZA vs V6
   * - M=128, N=16
     - ZA (1 tile)
     - 7.5
     - 3.7
     - **−51 %**
   * - M=128, N=32
     - ZA (2 tiles)
     - 9.9
     - 4.1
     - **−59 %**
   * - M=128, N=64
     - ZA (4 tiles, max residency)
     - 11.8
     - 4.6
     - **−61 %**
   * - M=1024, N=64
     - ZA (4 tiles)
     - 14.7
     - 4.8
     - **−68 %**
   * - M=4096, N=64
     - ZA (4 tiles)
     - 15.0
     - 5.1
     - **−66 %**
   * - M=128, N=2048
     - fallback (streaming)
     - 15.1
     - 13.4
     - −11 %
   * - M=1024, N=2048
     - fallback (streaming)
     - 16.3
     - 14.3
     - −12 %

Verdict: ZA loses decisively — and by more than RMSNorm's loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZA-resident fast path is **51–68 % slower than V6** — a bigger loss
than RMSNorm's 39–65 %, exactly as the pre-registered hypothesis predicted.
The mechanism is the same one RMSNorm exposed: ``mova`` throughput, not DRAM
bandwidth, is the binding constraint at these shapes, and LayerNorm's
prototype pays it three times per element instead of twice. The 50 % traffic
cut (vs RMSNorm's 33 %) is real, but it converts into *more* saved DRAM
time than RMSNorm's cut did — and that extra saving still isn't enough to
cover the extra ``mova`` op's cost, so the margin gets *worse*, not better.
This is the cleanest confirmation available that the RMSNorm finding
generalizes: trading DRAM reads for ZA residency is a structural loss on
this hardware for a bandwidth-bound horizontal-reduction norm, regardless of
how many reads are being traded away.

The fallback path (no ZA, N > 4·SVL) is 11–12 % slower than V6 — expected,
since this fallback is a plain single-block three-pass without V6's
4-row-block contiguity grouping or multi-accumulator ILP (the same pattern
``rms_norm_za``'s own fallback shows against V6, there by 28–35 %); it exists
for correctness on every shape, not to compete with V6.

- **LayerNorm architecture verdict:** SSVE **V6 stays the frozen incumbent**
  for Sprint 4's JIT, for both norms. ``layer_norm_za`` is kept in the tree
  and the ablation table as a *measured, explained* negative, and its
  bigger-than-RMSNorm loss margin is itself evidence: it closes the question
  Sprint 2c left open (whether LayerNorm's larger headroom might tip ZA
  staging into a win) with a direct, decisive answer rather than an
  inference from RMSNorm alone.

Correctness note: an eps register-stash bug, found and fixed during this work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While building ``layer_norm_za``, an empirical check (an all-zero RMSNorm row,
sumsq = 0 exactly) surfaced a real correctness bug affecting **every**
existing SSVE/ZA kernel except the two original LayerNorm baselines
(``layer_norm_ssve``/``_v1``): each stashed ``eps`` in the callee-saved S8/D8
register before ``smstart``, intending to read it back afterwards, either via
a stack slot that actually still held the *caller's* old ``d8`` (saved by
``str d8`` *before* the eps ``fmov`` overwrote the register), or by reading
S8/D8 directly post-``smstart`` — which also fails, because streaming-mode
entry clobbers D8 too, not just D9–D15 as had been assumed. Every existing
tolerance-based test passed regardless, since eps's effect on non-degenerate
data is far below the FP32 comparison tolerance; the bug is only observable
when sumsq/variance is exactly zero, where eps is the only thing standing
between a valid reciprocal and ``1/0 = inf`` (then ``inf * 0 = NaN``
propagates through the normalize step). Fixed in all 13 affected kernels by
stashing eps to a **dedicated stack slot before smstart** and reloading from
that same address afterwards — memory, unlike any register, is unaffected by
a PSTATE.SM transition. Two new regression tests lock this in (all-zero
RMSNorm row, all-constant LayerNorm row, both required to be finite). Full
test suite re-verified green after the fix (352 473 assertions, 93 cases).

Sprint 3 status
~~~~~~~~~~~~~~~

- **RMSNorm kernel:** ``rms_norm_za.S`` — verified on M4 (86 test cases in
  ``test_norm``), guarded by ``cpu_supports_sme()``, VLA, one streaming
  region with correct PSTATE.SM/ZA handling. Measured verdict: ZA residency
  is 39–65 % slower than V6 in its fast path.
- **LayerNorm kernel:** ``layer_norm_za.S`` — verified on M4 (5 test cases,
  full 3-pass ZA residency), same guards and VLA discipline. Measured
  verdict: ZA residency is 51–68 % slower than V6 in its fast path — a
  larger loss than RMSNorm's, confirming the mechanism generalizes.
- **Lever attributed for both:** the loss is the ``mova`` staging cost,
  quantified against V6 and against each kernel's own non-ZA fallback.
- **V6 (both norms) frozen as the incumbent for Sprint 4's JIT.**
- **Correctness fix:** the eps register-stash bug above, fixed across all
  13 affected kernels with two new regression tests.

Sprint 4 — JIT generation (mini_jit::Norm)
-------------------------------------------

Goal: generate the norm kernels **at runtime** instead of hand-writing them —
a ``mini_jit::Norm`` generator following the week-6 ``Unary`` pattern:
``generate(ntype)`` emits instruction words through ``InstGen`` into a week-5
``JitEngine`` executable buffer; ``get_rms_kernel()`` / ``get_layer_kernel()``
return a reusable function pointer.  Emission is a one-time cost.

What gets emitted — and what deliberately does not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generator emits the **measured Sprint-2 winners**: ``rms_norm_ssve_v6``
and ``layer_norm_ssve_v6``, transcribed 1:1 (same registers, same
instruction order).  The ZA path is **not** emitted: the ROADMAP gated it on
"if it earned its row", and Sprint 3's verdict was a decisive loss for both
norms (39–68 % slower than V6, ``mova``-throughput-bound).  Emitting a
kernel architecture that measurement rejected would add encoder surface and
maintenance for a path no shape selects — the reasoned skip *is* the
shape-dependent emission decision at SVL = 512.

Verification: the encoding diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JIT is verified by three layers, weakest to strongest
(``tests/test_norm.cpp``, tags ``[sprint4][encoders]``,
``[sprint4][encoding-diff]``, ``[sprint4][jit]``):

1. **Per-encoder golden words.**  Every new ``InstGen`` encoder is pinned to
   the exact word the toolchain assembled for the same instruction, taken
   from ``objdump`` of the two ``.S`` objects, cross-checked against the Arm
   ARM field layouts.  30 encoders were added (STP/LDP pairs, scalar
   FMOV/SCVTF/FDIV, predicated FMLA/FMUL/FADD/FSUB, FRSQRTE/FRSQRTS, LD1RW,
   LD1W/ST1W with ``MUL VL`` offsets, WHILELO, SEL, ADDVL, INCW, DUP, CBZ,
   B.cond, and the SM-only SMSTART/SMSTOP forms).
2. **Whole-kernel encoding diff.**  The generator's buffer is compared
   word-for-word against the **linked** hand-written kernel, read straight
   from its function address at runtime — the reference is what the
   toolchain actually assembled, and can never go stale.  Green for both
   norms: 143/143 words (RMSNorm) and 194/194 words (LayerNorm) identical.
   A green diff means the JIT kernel *inherits* the trust of a kernel that
   already passed the full Sprint-2/3 suite (including the eps-stash fix),
   instead of re-earning it.
3. **Execution anyway.**  The emitted kernels are run against the C++
   reference on the same shape set as the V6 tests (groups, tails,
   mismatched leading dimensions, N = 1, stress inputs, the zero-variance
   eps regression) — this also exercises the ``JitEngine`` buffer path
   (``MAP_JIT``, W^X toggling, icache invalidation).  All green on M4.

Layers 1–2 are host-portable and run on CI; layer 3 skips without SME.

What the exactness found: two latent InstGen bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparing the golden words surfaced two week-6 encoder bugs that every
behavioral test had passed over (full detail in ``sprint4_errors.md``):

- ``sve_ptrue_all(fp32)`` emitted ``PTRUE P.B`` (size bits 00), not the
  ``PTRUE P.S`` its own comment promised — functionally masked because an
  ALL-true ``.B`` predicate governs ``.S`` elements identically.  Fixed and
  pinned by test; week-5/6 suites re-verified green.
- ``sme_smstart_sm()`` actually encodes ``SMSTART`` (SM **and** ZA,
  ``MSR SVCRSMZA``), not ``SMSTART SM``.  Correct for Gemm (which needs ZA),
  wrong by name — and wrong for the norm kernels, which run SM-only to avoid
  the ZA lazy-save hazards.  Left in place for its dependents; correctly
  named ``sme_smstart_sm_only()`` / ``sme_smstop_sm_only()`` added alongside.

"It assembles and runs" is not "it is the instruction you meant"
(CLAUDE.md §10) — an encoding diff checks what the code *is*, and caught in
minutes what weeks of behavioral tests structurally could not.

A harness bug found by the parity run — and a Sprint-2a correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first parity benchmark showed the word-identical JIT kernel **+76 %
faster** than the hand-written one at M=128/N=64 — physically impossible for
identical code, so the gap had to be the call path.  Cause: the hand-written
kernels are benchmarked through their ``mini_jit::norm`` guard wrappers, and
``cpu_supports_sme()`` performed an uncached ``sysctlbyname`` **syscall on
every call** (~1 µs); the JIT pointer is called directly.  Fix: cache the
result in a static (CPU features cannot change at runtime).

Consequence for earlier numbers: the Sprint-2a small-N sweep ran through the
same wrapper, so its headline "fixed cost t0 ≈ 1.4 µs/call, overhead =
streaming work at N ≈ 41" was **~70 % syscall**.  With the guard fixed, the
N=16 call drops from 1.375 µs to ~0.42 µs, throughput is flat (~29 GiB/s)
from N=32 upward, and the true fixed per-call cost is too small for the
linear fit to resolve (intercept within noise of zero, ≲0.4 µs).  The qualitative Sprint-2a conclusions stand (a
fixed per-call cost exists; the N=64 dip is real), but the *magnitude* was
harness, not hardware — recorded here as a correction, in the same spirit as
Sprint 2b's correction of the 2a residency interpretation.

Fixing the fix: the sweep's GiB/s column then printed 0.00 — the now
trivially-inlinable guard shifted register allocation so the ``1/2^30``
constant inside ``to_gibs`` landed in a callee-saved FP register that
``SMSTART`` zeroes.  ``volatile`` inputs cannot protect a hoisted
*constant*; the robust fix separates the bench loop from the print loop so
no SME call sits between computation and output.  (Third appearance of the
D9–D15 hazard in this project; each time in a new disguise.)

Parity and emission cost (M4, corrected harness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Emission (one-time): **~4.7 µs** for RMSNorm (143 words), **~4.8 µs** for
LayerNorm (194 words) — recouped after a handful of small-shape calls, and
the pointer is reused thereafter.

.. list-table:: JIT-emitted V6 vs hand-written V6 (GiB/s, useful bytes, % of 59.5 GiB/s 1-core SSVE roofline)
   :header-rows: 1

   * - Shape (M×N)
     - RMS hand
     - RMS JIT
     - Δ
     - LN hand
     - LN JIT
     - Δ
   * - 128×64
     - 38.56 (65.2 %)
     - 38.56 (65.2 %)
     - +0.0 %
     - 16.46 (27.8 %)
     - 16.65 (28.1 %)
     - +1.1 %
   * - 128×2048
     - 38.52 (65.1 %)
     - 38.52 (65.1 %)
     - +0.0 %
     - 16.51 (27.9 %)
     - 16.51 (27.9 %)
     - −0.0 %
   * - 1024×2048
     - 38.44 (65.0 %)
     - 38.45 (65.0 %)
     - +0.0 %
     - 16.49 (27.9 %)
     - 16.48 (27.9 %)
     - −0.0 %
   * - 4096×2048 (DRAM)
     - 25.84 (43.7 %)
     - 26.21 (44.3 %)
     - +1.5 %
     - 14.42 (24.4 %)
     - 14.44 (24.4 %)
     - +0.1 %

Parity within noise at every shape — exactly what a word-identical kernel
must show once the call paths are equalized.  The mmap'd JIT page vs the
executable's text section makes no measurable difference.

Sprint 4 status
~~~~~~~~~~~~~~~

- ``mini_jit::Norm`` emits both SSVE V6 winners; ZA path skipped with its
  measured rationale (Sprint 3).
- 30 new ``InstGen`` encoders, each pinned to a toolchain golden word;
  two latent week-6 encoder bugs found and resolved.
- Encoding diff green for both norms (143 + 194 words); JIT kernels verified
  against the reference on M4; full suite green.
- Benchmark parity within noise at all shapes; emission ~5 µs one-time.
- Harness corrections: ``cpu_supports_sme()`` syscall caching (with the
  Sprint-2a t0 restatement) and the bench/print loop separation.
- ``ctest`` discovery fixed repo-wide (``enable_testing()`` ordering).
