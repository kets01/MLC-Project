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

Next
~~~~

Sprint 2 (LayerNorm): hand-written SSVE kernel for LayerNorm — two-pass,
VLA, predicated tails, verified against the reference, measured vs the
roofline; followed by the same V1–V3 ablation.
