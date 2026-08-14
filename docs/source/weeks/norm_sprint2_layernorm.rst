MLC-Norm Sprint 2 — LayerNorm: SSVE Kernel and Ablation
=======================================================

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

**Roofline:** the single-core streaming ceiling (``LD1W``/``ST1W`` probe) at
each shape's own footprint — 104.7–117.2 GiB/s for the six cache-resident
shapes below.  Originally quoted as one 59.45 GiB/s figure, which is the
ceiling at a 256 MiB working set (Sprint 6 §1.2).

.. list-table:: Sprint 2c — layer_norm_ssve V0 vs reference (Apple M4)
   :header-rows: 1
   :widths: 10 10 20 20 16 12

   * - M (rows)
     - N (features)
     - ``layer_norm_ref`` GiB/s
     - ``layer_norm_ssve`` GiB/s
     - ceiling
     - % of ceiling
     - Speedup
   * - 128
     - 64
     - 4.09
     - 9.64
     - 104.7
     - 9.2 %
     - 2.4×
   * - 128
     - 512
     - 2.41
     - 12.37
     - 117.2
     - 10.6 %
     - 5.1×
   * - 128
     - 2048
     - 2.24
     - 12.39
     - 116.0
     - 10.7 %
     - 5.5×
   * - 1024
     - 64
     - 1.23
     - 12.03
     - 117.2
     - 10.3 %
     - 9.8×
   * - 1024
     - 512
     - 1.22
     - 12.38
     - 115.7
     - 10.7 %
     - 10.1×
   * - 1024
     - 2048
     - 0.95
     - 12.76
     - 115.6
     - **11.0 %**
     - **13.4×**

.. note::

   Percentages restated in Sprint 6 (§1.2) against the streaming ceiling at
   each shape's own footprint.  The original column divided every row by
   59.5 GiB/s — the ceiling at a 256 MiB working set — while all six shapes
   here are cache-resident, so it read ~2× too high (16–21 % instead of
   9–11 %).  The GiB/s and the speedups are unchanged.

The kernel plateaus at **~10–11 % of the ceiling at its own footprint** (useful
bytes, 1R+1W convention).  Compare to RMSNorm V0's ~21 %.  The factor of ~2
is fully explained by the three-pass memory access pattern: LayerNorm reads x
three times (mean, variance, normalize), so its moved-bytes traffic is
**3R+1W** vs RMSNorm's **2R+1W**.  Restated in moved-bytes terms (M=1024,
N=2048, ceiling 115.6):

- LayerNorm V0: 12.76 GiB/s useful → ×2 = **~25.5 GiB/s moved** ≈ **22 % of ceiling**
- RMSNorm V0:   25 GiB/s useful → ×1.5 = **~37.5 GiB/s moved** ≈ **32 % of ceiling**

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
- **Benchmark:** V0 baseline measured; 12–13 GiB/s (10–11 % of the ceiling at
  each shape's footprint; stated at the time as 20–22 %, against the DRAM
  ceiling — corrected in Sprint 6 §1.2)
- **Ablation:** V1–V6 + Welford — see the next section

Sprint 2c ablation — LayerNorm V1–V6 and the Welford verdict
-------------------------------------------------------------

**Goal:** replay the RMSNorm levers on LayerNorm's three-pass structure
(characterized separately, not assumed to transfer) and run the algorithmic
ablation the ROADMAP called the headline row: vectorized Welford single-pass
vs the stable two-pass.  All variants verified on M4 first — including
large-magnitude stress inputs for V4/V5/V6/Welford (added as gap-fill; a
Welford speed verdict is only meaningful if its accuracy actually holds).

Ablation results (useful bytes; % against the ceiling at each shape's footprint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: LayerNorm SSVE ablation, key rows (Apple M4)
   :header-rows: 1

   * - shape
     - V0
     - V2
     - V4
     - **V6**
     - Welford
     - ceiling
     - **V6 %**
   * - M=128, N=2048 (2 MiB)
     - 12.0
     - 13.4
     - 15.2
     - **15.2**
     - 11.7
     - 116.0
     - **13.1 %**
   * - M=1024, N=2048 (16 MiB)
     - 12.7
     - 14.3
     - 16.3
     - **16.3**
     - 12.1
     - 115.6
     - **14.1 %**
   * - M=4096, N=2048 (64 MiB, true DRAM)
     - 8.4
     - 8.3
     - 8.4
     - **14.4**
     - 8.1
     - 63.3
     - **22.7 %**

The heading originally read "59.4 GiB/s single-core streaming roofline" and
applied that one number to all three rows; only the 64 MiB row is anywhere near
DRAM-resident (Sprint 6 §1.2).  As with RMSNorm, the per-lever deltas below are
kernel-to-kernel ratios and are unaffected.

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

Welford vs two-pass: two-pass wins on speed; the accuracy claim was overstated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The comparison production libraries actually face, measured:

- **Speed: Welford loses everywhere** — −2.5 % to −8.5 % vs V0, and
  25–30 % below V6 — *despite moving 25 % fewer bytes* (2R+1W vs 3R+1W).
  The pre-registered hypothesis held: the per-element recurrence
  (``mean += delta/count``) serializes on SIMD, and the FDIV-per-column cost
  swamps the traffic saving.  **This is the argument that decides the
  incumbent, and it is unaffected by anything below.**

.. admonition:: Corrected in Sprint 6 (§1.3) — "~100× less accurate" does not hold
   :class: warning

   This section originally claimed Welford is *~100× worse on shifted data*
   and that two-pass therefore "dominates on BOTH axes".  Re-measured against
   a float64 reference at M=16, N=512, across more shift magnitudes:

   .. list-table::
      :header-rows: 1

      * - input shift
        - V6 (two-pass)
        - Welford
        - V0 (two-pass, exact ``FSQRT``)
      * - 0
        - 1.16e-5
        - 1.09e-5
        - 3.33e-6
      * - 1e2
        - 1.16e-5
        - 1.49e-5
        - 3.33e-6
      * - 1e4
        - **1.16e-5**
        - **5.37e-4**
        - 3.33e-6
      * - 1e5
        - 4.30e-3
        - **1.69e-3**
        - 4.30e-3

   The claim holds at exactly one shift (1e4, and the factor is ~46×, not
   ~100×), is a wash at 0 and 1e2, and **reverses at 1e5**, where Welford is
   2.5× *better*.  "Dominates on both axes" is not supported.

   **The stated mechanism was also wrong.**  The original text blames
   Welford's variance recurrence.  Isolating the variance alone in FP32 shows
   the opposite — Welford's variance is the *more* accurate of the two
   (3.4e-7 vs two-pass's 3.0e-6 at shift 1e2 and 1e4).  The extra output error
   comes from the running **mean**: ``mean += delta/count`` updates at full
   input magnitude every element, while two-pass forms the mean once and then
   squares already-centred values.  LayerNorm's output is
   ``(x − mean)·inv_std``, so mean error is amplified by ``inv_std``.

   **Also undocumented until now:** V6's accuracy floor of 1.16e-5 is not the
   algorithm, it is ``FRSQRTE``+Newton-Raphson.  V0, which uses exact
   ``FSQRT``+``FDIV``, reaches 3.33e-6 — so V6 trades **3.5× accuracy** for
   that substitution.  That is a real cost of the ablation's winning variant
   and belongs in the table.

   **What does not change:** the decision.  Two-pass stays the incumbent
   because Welford is 25–30 % slower for a measured reason.

  The centered two-pass subtracts the mean *before* squaring, so its
  variance arithmetic works on small values; FP32 Welford updates
  ``mean`` at full input magnitude every element, accumulating rounding at
  the scale of the shift.  Both are far better than the catastrophic naive
  ``E[x²]−µ²`` — which is the row that makes this comparison meaningful, and
  which Sprint 7a adds.  Worth noting that PyTorch's CPU LayerNorm moved *to*
  Welford precisely to escape naive; "Welford ≫ naive" and "Welford ≈
  two-pass" are both true, and the original text compared against the stronger
  baseline without saying so.

**Verdict:** on SIMD the stable two-pass formulation wins on **speed**, which
is why it is the incumbent.  On FP32 accuracy the two are comparable, with the
ordering depending on the shift magnitude.  Welford's ablation row is kept with
both measurements, not just the throughput number.

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
RMSNorm's one.  In moved-bytes terms at M=1024, N=2048 (16 MiB footprint,
ceiling 115.6 GiB/s) LN V6 sustains ~32.6 GiB/s (**28 %** of that ceiling) vs
RMS V6's ~56.6 (**49 %**): the remaining LayerNorm-specific gap is the pass
structure itself.  (Originally quoted as 55 % and 95 %, against the DRAM
ceiling — corrected in Sprint 6 §1.2.  The *ratio* between the two norms, which
is what this section is about, is unchanged.)

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

