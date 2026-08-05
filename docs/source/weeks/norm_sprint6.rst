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

.. list-table:: Consolidated ablation, useful bytes (1R+1W), % of the 59.5 GiB/s single-core SSVE roofline
   :header-rows: 1

   * - stage
     - RMS 16 MiB
     - RMS 256 MiB (DRAM)
     - LN 16 MiB
     - LN 256 MiB (DRAM)
   * - scalar reference
     - 1.03
     - 0.68
     - 0.95
     - 0.54
   * - V0 SSVE
     - 25.05
     - 10.39
     - 12.80
     - 7.44
   * - V6 (Sprint-2 incumbent)
     - 38.46 (65 %)
     - 21.05 (35 %)
     - 16.47 (28 %)
     - 13.26 (22 %)
   * - **V7 (SME2)**
     - 38.46
     - **24.92 (42 %)**
     - 16.49
     - 13.60
   * - ZA residency
     - 25.11
     - 10.13
     - 14.41
     - 7.55
   * - Welford
     - —
     - —
     - 12.01
     - 6.90

Every row verified; maximum deviation from the reference across the whole table
is 1.9e-5 absolute, which is the accuracy FRSQRTE+NR actually delivers rather
than a tolerance chosen to pass.

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
explanation is falsified; the memory-level-parallelism one survives. The
threaded path points the same way: the win is +17 % single-threaded through TEIR
but only +2.6 % at 16 threads, where the memory system is already saturated by
other cores and there is no per-core request capacity left to relieve.

This also refines Sprint 2b rather than contradicting it: V6 made the
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

Sprint 6 status
~~~~~~~~~~~~~~~~

* **Correctness:** the ``d8-d15`` AAPCS64 fix completed across all 15 remaining
  kernels and the roofline probe; an all-kernel ABI test added; 65 unreadable
  benchmark rows recovered.
* **Ablation:** one consolidated, per-row-verified table covering the full
  ladder for both norms across three regimes, against both ceilings, with the
  byte convention restated in the header.
* **SME2:** ``cpu_supports_sme2()`` added; V7 built for both norms, verified
  bit-identical to V6, measured +17.2 % (RMSNorm) and +2.7 % (LayerNorm) in the
  DRAM regime; the pre-registered "no help" expectation is recorded as wrong,
  with the discriminating experiment that explains why.
* **JIT/TEIR:** feature-dependent emission, three new encoders pinned to golden
  words, encoding diffs green for both SME2 kernels, TEIR single-thread RMSNorm
  +17.3 %.
* **Suite:** 7/7 CTest suites green on the M4; ``test_norm`` at 460 k+
  assertions.

Deliberately left open
~~~~~~~~~~~~~~~~~~~~~~~

* **SME2 multi-vector on the ZA path.** The ROADMAP flagged it as the other
  candidate. It is not attempted here: Sprint 3 measured ZA as ``mova``-bound at
  ~10 GiB/s at *every* footprint, and the pre-registered arithmetic — even a 2×
  ``mova`` speed-up only reaches ~20 GiB/s, a tie at best with V6, and only in
  the N ≤ 64 window that is cache-resident anyway — is unchanged by anything
  measured this sprint. Recorded as a reasoned skip, not an oversight.
* **A V7 tail path.** The 1-3 block tail stays plain SVE; multi-vector partial
  predication would need ``WHILELO ...,VLx4`` and runs at most three times per
  call.
* **Threading past 8 cores.** Scaling still saturates around 8 threads; the
  row-chunked schedule gives each thread a strided walk over a 256 MiB
  footprint. Unchanged by this sprint.
