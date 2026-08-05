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

