MLC-Norm Sprint 6 — Optimization, Roofline & the Ablation Study
================================================================

Sprint 6 — the evaluation deliverable
--------------------------------------

Goal: consolidate five sprints of scattered measurements into one attributable
ablation, state every number against both validated ceilings, and test the one
instruction-set lever left untried — SME2.

Two things came out of it that were not on the plan: a correctness defect that
made most of the existing benchmark tables unreadable, and an SME2 result that
**contradicted the pre-registered expectation** and ended up changing what the
JIT emits.

The blocker: the Sprint-5 ABI fix was only half-applied
---------------------------------------------------------

The sprint opened by re-running ``main_norm`` for a baseline. **65 rows printed
0.00 GiB/s** — Section 1 after the first SME call, and both ZA tables in their
entirety. That is the ``d8-d15`` clobber signature already logged twice
(``sprint4_errors.md`` #6 and #8).

Sprint 5 fixed the two V6 winners, because those are what TEIR calls, and left
everything else. A hardware probe over every entry point found **15 of 17
destroying the caller's d9-d15**, including ``bw_probe_ssve`` — the roofline
probe that every "% of peak" figure in this report is divided by.

The reason it resurfaced now rather than then: ``main_norm`` defends itself with
caller-side ``volatile``, which the Sprint-2 debug log already described as a
workaround rather than a fix. A workaround holds only while register allocation
cooperates, and it had quietly stopped.

Fixed across all 15 kernels plus the probe, using the frame layout Sprint 5
established. Two notes on how it went, because both are the point:

* The transformation was applied to the regular kernel families by a script
  that asserts each expected prologue/epilogue block appears exactly once and
  refuses to touch a file otherwise. It still introduced a bug: the eps-slot
  rewrite also matched the freshly-inserted ``d9`` line, so ``d9`` and eps
  collided in one stack slot. **Every functional test still passed** — eps
  remained self-consistent — and only the register probe caught it. A
  behavioural test cannot see a register it does not hold a value in.
* The pre-existing ABI test covered only the two V6 kernels, which is precisely
  how the other 15 drifted. There is now a test pinning all 17 at once
  (``[sprint6][abi]``, 119 assertions), so a new variant cannot ship without it.

Result: **65 zero rows → 0**, and every table reproduces its historically
documented values.

The consolidated ablation
--------------------------

``main_norm`` now opens with a single table covering the whole ladder — scalar
reference → V0 → V1/V2/V3 → V4/V5 → V6 → V7 → ZA → JIT — for both norms, on
three shapes chosen to separate the regimes, with each row **verified against
the C++ reference before its GiB/s is printed** (decision B) and the measured
maximum deviation printed beside it, against **both** ceilings (decision E).

It is written deliberately without the ``volatile`` discipline the older
sections use: that workaround is no longer load-bearing, and if the ABI fix ever
regresses this section shows it first, as 0.00 rows.

The harness also caught a flaw in itself. At the small shape the hand-written V6
first measured 10 % *below* its own word-identical JIT twin — impossible for
identical code, so it was sampling noise, not signal. Repetitions now scale with
working-set size instead of being a fixed count, after which the two agree to
the last printed digit (38.56 / 38.56).

.. list-table:: Consolidated ablation, useful bytes (1R+1W), % of the ceiling at each footprint
   :header-rows: 1

   * - stage
     - RMS 16 MiB
     - RMS 256 MiB (DRAM)
     - LN 16 MiB
     - LN 256 MiB (DRAM)
   * - *ceiling at this footprint*
     - *115.56*
     - *59.54*
     - *115.56*
     - *59.54*
   * - scalar reference
     - 1.03 (0.9 %)
     - 0.68 (1.1 %)
     - 0.95 (0.8 %)
     - 0.54 (0.9 %)
   * - V0 SSVE
     - 25.02 (21.7 %)
     - 10.09 (17.0 %)
     - 12.82 (11.1 %)
     - 7.27 (12.2 %)
   * - V4 (accumulator ILP)
     - 31.68 (27.4 %)
     - 10.52 (17.7 %)
     - 16.43 (14.2 %)
     - 7.75 (13.0 %)
   * - V6 (Sprint-2 incumbent)
     - 38.46 (33.3 %)
     - 20.29 (34.1 %)
     - 16.48 (14.3 %)
     - 13.11 (22.0 %)
   * - **V7 (SME2)**
     - 38.46 (33.3 %)
     - **24.65 (41.4 %)**
     - 16.49 (14.3 %)
     - 13.54 (22.7 %)
   * - ZA residency
     - 25.04 (21.7 %)
     - 10.11 (17.0 %)
     - 14.42 (12.5 %)
     - 7.45 (12.5 %)
   * - Welford
     - —
     - —
     - 12.00 (10.4 %)
     - 6.81 (11.4 %)
   * - JIT (auto ISA)
     - 38.47 (33.3 %)
     - 24.64 (41.4 %)
     - 16.49 (14.3 %)
     - 13.55 (22.8 %)

Every row verified; maximum deviation from the reference across the whole table
is 1.9e-5 absolute, which is the accuracy FRSQRTE+NR actually delivers rather
than a tolerance chosen to pass.

.. admonition:: Denominator corrected after this table was first published (§1.2)
   :class: warning

   As originally printed, every column divided by **59.5 GiB/s** — the ceiling
   the Sprint-2a probe measured at a 256 MiB working set.  The 16 MiB columns
   are cache-resident and their real ceiling is **115.56 GiB/s**, so those
   percentages were inflated ~2× (V6 at 16 MiB read "65 %"; it is 33.3 %).
   The DRAM columns were already correct.

   ``main_norm`` now measures the ceiling as a **curve** across footprints
   (64 KiB → 256 MiB) and each row divides by the ceiling at its own working
   set, so the table regenerates correctly rather than depending on a constant
   that happens to suit one regime.

   The correction inverts one reading of this table.  At 16 MiB RMSNorm looked
   like it was close to saturating its execution mode; it is at a third of the
   available bandwidth.  What binds there is not memory at all — see the FP
   issue-rate measurement below — which is also why V7 buys nothing at that
   footprint and +17 % at DRAM.

SME2: the lever, and a wrong prediction
-----------------------------------------

The ROADMAP recorded a hardware correction — ``sysctl`` reports
``FEAT_SME2 = 1`` on the target M4, so the machine is not SME1-only as the docs
assumed — and pre-registered the expectation that SME2 **cannot help**, since it
does not raise the DRAM roofline.

V6's group loop already touches four consecutive VL-row blocks per column at
``[x8]``, ``[x8,#1,MUL VL]``, ``[x8,#2,MUL VL]``, ``[x8,#3,MUL VL]``. That is
exactly the operand shape of SME2's 4-vector contiguous access, so **V7 = V6
with four accesses folded into one**:

.. code-block:: text

    4x  ld1w {zN.s}, p0/z, [x8, #k, mul vl]   ->  ld1w {z0.s-z3.s}, pn8/z, [x8]
    4x  st1w {zN.s}, p0,   [x9, #k, mul vl]   ->  st1w {z0.s-z3.s}, pn8, [x9]

Same addresses, same traffic, same arithmetic, same summation order. The only
variable is instruction count — which makes it a clean test of whether V6 is
issue-bound or memory-bound. Because no value changes, the correctness gate is
**bit-identity with V6**, not a tolerance.

Before writing either kernel, the multi-vector forms were assembled *and
executed* inside a streaming region on the M4, so the work rests on a verified
capability rather than a datasheet reading (CLAUDE.md §10).

Measured (order-alternating A/B, 24 samples each way, medians):

.. list-table::
   :header-rows: 1

   * - norm
     - V6
     - V7 (SME2)
     - Δ
   * - RMSNorm, 256 MiB
     - 20.96
     - 24.56
     - **+17.2 %**
   * - LayerNorm, 256 MiB
     - 13.20
     - 13.56
     - +2.7 %
   * - either norm, cache-resident
     - —
     - —
     - ≈0

**The prediction was wrong**, and the interesting part is why. Two mechanisms
could explain an RMSNorm win, and they make opposite predictions for LayerNorm:

* *Fewer instructions retire faster.* LayerNorm folds **more** accesses per
  element (three passes, not two), so it should gain **more**.
* *Relieved memory-level parallelism* — one instruction requesting 256
  contiguous bytes occupies less load-tracking capacity than four requesting 64,
  which only matters where outstanding-request capacity is the constraint.
  LayerNorm does far more FP work per byte and is the less memory-starved of the
  two, so it should gain **less**.

LayerNorm gained +2.7 % against RMSNorm's +17.2 %. The instruction-count
explanation is falsified. The threaded path points the same way: the win is
+17 % single-threaded through TEIR but only +2.6 % at 16 threads, where the
memory system is already saturated by other cores and there is no per-core
request capacity left to relieve.

The 2-vector control: both hypotheses die
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory-level-parallelism story was *pre-registered with a quantitative
prediction*, which is what made it testable: if the win comes from bytes in
flight per load-queue entry, then a **2-vector** variant at 128 B per
instruction should land halfway between V6 and V7, around +9 %.
``rms_norm_ssve_v7x2.S`` was built as exactly that control.

.. list-table:: RMSNorm, 256 MiB, bytes per load instruction
   :header-rows: 1

   * - variant
     - load insns/column
     - GiB/s
     - vs V6
   * - V6 (4 × 64 B)
     - 4
     - 20.94
     - —
   * - **V7x2 (2 × 128 B)**
     - 2
     - **24.31**
     - **+16.1 %**
   * - V7 (1 × 256 B)
     - 1
     - 24.67
     - +17.9 %

**The curve saturates**: going 4 → 2 captures 90 % of the win, and 2 → 1 adds
only 1.5 %. That is inconsistent with a proportional bytes-per-slot model — the
memory-level-parallelism hypothesis — *and* with a "256 B burst matches the
prefetcher" model. Both pre-registered explanations are dead.

So the honest statement is narrower than the one this section originally made
("the memory-level-parallelism one survives"): what is supported is a
**threshold effect on load-instruction count**, in load-dominated loops, in the
latency-exposed regime only. Why the threshold sits between four and two
instructions per column is not established here.

This still refines Sprint 2b rather than contradicting it: V6 made the
*addresses* contiguous, V7 makes the *request* singular. SME2 did not raise the
roofline — the ROADMAP was right about that — it changed how much of the
existing roofline one core can reach.

Promotion into the JIT and TEIR
---------------------------------

Because V7 earned its row, it became the first genuinely *feature*-dependent
emission decision in the project. ``mini_jit::Norm::generate()`` now takes an
``isa_t``: ``automatic`` emits V7 where ``FEAT_SME2`` is present and V6
otherwise, and the explicit values let the tests pin either path regardless of
build host.

Three new ``InstGen`` encoders were needed (``PTRUE PNn.S``, 4-vector ``LD1W``
and ``ST1W``). Field layouts were derived from 13 toolchain golden words varying
register, base and predicate independently, then pinned by unit test — the
Sprint-4 methodology, which is what keeps the encoding diff from being circular.

Both SME2 kernels pass the whole-buffer encoding diff against their linked
hand-written counterparts, so the emitted code inherits the trust of kernels
that already passed bit-identity and reference verification. Emitted sizes drop
from 157 → 148 words (RMSNorm) and 208 → 196 (LayerNorm), exactly the 9 and 12
folded accesses.

TEIR picks this up for free — its runtime constructs the generators with the
default — so the integrated path improves without any change to the runtime:

.. list-table:: TEIR + OpenMP, 256 MiB working set, before/after V7 emission
   :header-rows: 1

   * - configuration
     - V6 emitted
     - V7 emitted
     - Δ
   * - RMSNorm, 1 thread
     - 21.04
     - 24.68
     - +17.3 %
   * - RMSNorm, 16 threads
     - 42.77
     - 43.88
     - +2.6 %
   * - LayerNorm, 1 thread
     - 13.24
     - 13.63
     - +2.9 %
   * - LayerNorm, 16 threads
     - 26.11
     - 26.15
     - +0.2 %

Every configuration is verified against the C++ reference before its number is
reported.

SME2 on the ZA path: the skip was retracted and the experiment run
--------------------------------------------------------------------

This section previously recorded SME2-on-ZA as a *reasoned skip*, on Sprint 3's
finding that ZA is ``mova``-bound at ~10 GiB/s and the arithmetic that "even a
2× ``mova`` speed-up only reaches ~20 GiB/s, a tie at best with V6".

That skip rested on an untested factor. ``mova`` had never been characterised
as **issue**-bound versus **ZA-port**-bound, and the assumed 2× was a guess.
Measured directly, with no memory traffic and 16 vectors moved either way:

.. list-table:: ``mova`` throughput, single- vs multi-vector
   :header-rows: 1

   * - form
     - G vectors/s
     - ratio
   * - single-vector
     - 3.88
     - —
   * - 4-vector
     - **15.50**
     - **4.00×**

``mova`` is issue-bound, and the fold is 4:1, not the assumed 2:1 — so the
skip's own arithmetic did not hold, and the kernels were built:
``rms_norm_za_sme2.S`` and ``layer_norm_za_sme2.S``, bit-identical to their
Sprint-3 counterparts on every shape tested.

.. list-table:: ZA rebuilt on multi-vector MOVA (GiB/s)
   :header-rows: 1

   * - norm
     - Sprint-3 ZA
     - Sprint-6 ZA-SME2
     - gain
     - vs V7
   * - RMSNorm
     - 10.07
     - 16.35
     - **+62 %**
     - −37 %
   * - LayerNorm
     - 4.83
     - 10.34
     - **+114 %**
     - −24 %

**The verdict is unchanged — ZA still loses — but it is now a measured verdict
rather than an extrapolation from an assumed factor.** And the residual cause
is different from the original diagnosis: the ZA kernels process **one** SVL-row
block per iteration (64 B per column touch) against V6/V7's **four** (256 B).
What limits them is the Sprint-2b access-density lever, not ``mova`` throughput
— which is why making ``mova`` 4× faster still left them behind.

Instrument readings from this sprint
--------------------------------------

Several measurements taken to settle the arguments above are worth recording in
their own right, independently of the conclusions they supported.

* **The ceiling is footprint-dependent in both execution modes** — the curve in
  §1.2 above. Streaming mode costs ~25 % against NEON at DRAM (0.75×) but is
  nearly free in cache (0.93×), so "streaming is a structural handicap" is a
  DRAM-regime statement, not a property of the mode.
* **SSVE FP issue rate:** ``FMLA`` 0.97 G instructions/s (= 31.0 GFLOPS);
  ``FMUL`` 0.97 G instructions/s. The ratio is **1.00×** — an ``FMUL`` costs
  exactly what an ``FMLA`` costs, so instructions, not flops, are the unit that
  makes these comparable.
* **Cache-resident RMSNorm V7 is FP-issue-bound, not memory-bound.** At 16 MiB
  it sustains 0.97 G FP vector instructions/s — **100 % of the measured SSVE
  issue ceiling** — against 0.65 G (67 %) in the true-DRAM regime. This is the
  explanation for something the ablation shows but does not account for: V7
  gains ~0 % at cache-resident footprints because bandwidth was never the
  binding constraint there.
* **SME FMOPA issue rate:** 3.87 G instructions/s (1983.9 GFLOPS) — 3.99× the
  ``FMLA`` *issue* rate and 64× its flop rate, reproducing the Jena "Hello SME"
  figure.
* **FMOPA and SSVE FP are disjoint resources.** 8 ``FMLA`` = 8.26 ns, 8
  ``FMOPA`` = 2.07 ns, 8 of each interleaved = **8.26 ns** — exactly ``max``,
  not ``sum``. They overlap completely.
* **Multi-vector MOVA round-trips exactly:** two groups, all Z registers
  clobbered between write and read-back, 0 of 128 elements wrong.

.. note::

   A guard against a tempting misreading of the FMOPA numbers: normalization
   has an arithmetic intensity of ~0.33 flops/byte, and reaching a 1984 GFLOPS
   matrix-unit peak would need ~31 flops/byte — about 100× more. "We are at
   1 % of FMOPA peak" is not a defect to optimise away; it is what the roofline
   permits for this operator. Decision E already settles the metric: GiB/s, not
   GFLOPS.

Sprint 6 status
~~~~~~~~~~~~~~~~

* **Correctness:** the ``d8-d15`` AAPCS64 fix completed across all 15 remaining
  kernels and the roofline probe; an all-kernel ABI test added; 65 unreadable
  benchmark rows recovered.
* **Ablation:** one consolidated, per-row-verified table covering the full
  ladder for both norms across three regimes, with the byte convention restated
  in the header and each row divided by the ceiling **measured at its own
  footprint** — the §1.2 correction, which also required making ``main_norm``
  sweep the ceiling as a curve.
* **SME2:** ``cpu_supports_sme2()`` added; V7 built for both norms, verified
  bit-identical to V6, measured +17.2 % (RMSNorm) and +2.7 % (LayerNorm) in the
  DRAM regime; the pre-registered "no help" expectation is recorded as wrong.
  A 2-vector control then falsified **both** candidate explanations, leaving a
  threshold effect on load-instruction count as the supportable claim.
* **SME2 on ZA:** the reasoned skip was retracted after measuring that ``mova``
  is issue-bound with a 4:1 fold (not the assumed 2:1); both ZA kernels rebuilt
  (+62 % RMSNorm, +114 % LayerNorm) and still losing to V7 — now by measurement
  rather than extrapolation, and for a different reason than first diagnosed.
* **Report corrections:** four claims already in the report were found wrong or
  overstated (the half-applied ABI fix, the single-constant roofline, the
  Welford accuracy claim, the ZA skip) and are corrected in place across
  Sprints 0/1 through 6; nine claims made *during* the sprint were refuted by
  the next measurement. The full log is ``sprint6_errors.md``.
* **JIT/TEIR:** feature-dependent emission, three new encoders pinned to golden
  words, encoding diffs green for both SME2 kernels, TEIR single-thread RMSNorm
  +17.3 %.
* **Suite:** 7/7 CTest suites green on the M4; ``test_norm`` at 460 k+
  assertions.

Deliberately left open
~~~~~~~~~~~~~~~~~~~~~~~

* **A V7 tail path.** The 1-3 block tail stays plain SVE; multi-vector partial
  predication would need ``WHILELO ...,VLx4`` and runs at most three times per
  call.  (The V7 kernels were later made tail-free by having ``WHILELO``
  govern full and partial groups alike, so this is closed in the current code.)
* **Threading past 8 cores.** Scaling is genuinely flat from ~8 threads
  (8 → 16 buys ~2 %) once chunks are group-aligned.  The earlier explanation
  offered here — a strided walk from row chunking — was not established; see
  the Sprint-5 correction box, where the mechanism behind the base-offset
  sensitivity is recorded as **unidentified**.
* **The threshold behind V7's win.** The 2-vector control shows the gain
  saturates between four and two load instructions per column, but not why the
  threshold sits there. Both pre-registered mechanisms were falsified; naming
  the real one needs microarchitectural counters this harness does not read.
