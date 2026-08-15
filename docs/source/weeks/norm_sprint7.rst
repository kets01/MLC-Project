MLC-Norm Sprint 7 — Numerical Stability & Streaming Overhead
=============================================================

Sprint 7 — the two tradeoffs, measured
----------------------------------------

``context.md`` §8 names two "hard problems" the project promises to surface
rather than hide: single-pass variance is numerically dangerous, and streaming
mode is not free. Both have been *design premises* since Sprint 0 — the
LayerNorm kernels are two-pass because of the first, and every kernel keeps one
streaming region per call because of the second — but neither had ever been
measured on this machine.

This sprint measures them. Both results survive, and both turn out to be true
for a different reason, or by a different margin, than assumed.

7a — Where each variance formulation breaks
---------------------------------------------

The project never implemented the dangerous formulation, so the claim about it
had no counterexample. ``src/norm/stability.cpp`` supplies one: the naive
single-pass ``E[x²] − mean²``, the centred two-pass, and Welford, all
accumulating in **FP32** (the kernels' precision — a float64 demonstrator would
prove nothing about them), against a float64 oracle.

The stress axis is a **shift**. Adding a constant *s* to every element leaves
the variance unchanged but scales the problem's condition number
:math:`\kappa = 1 + \mathrm{mean}^2/\mathrm{var}` by roughly :math:`s^2`, which
isolates conditioning from every other property of the data.

.. note::

   The summation order is trustworthy because the build sets no
   ``-ffast-math``. Under strict IEEE the compiler may not reassociate a
   floating-point reduction, so the sequential order written in the source is
   the order that executes. Without that guarantee the three estimators would
   not be comparable to each other.

.. list-table:: Relative error vs the float64 oracle, N=512, identical input
   :header-rows: 1

   * - shift
     - κ
     - naive 1-pass
     - two-pass
     - Welford
   * - 0
     - 1.0
     - 5.09e-07
     - 3.89e-07
     - 2.07e-07
   * - 1e1
     - 2.0e2
     - 1.39e-04
     - 3.98e-07
     - 1.98e-07
   * - 1e2
     - 2.0e4
     - 8.52e-03
     - 1.18e-07
     - 4.16e-07
   * - 1e3
     - 2.0e6
     - 1.25e+00
     - 3.75e-07
     - 4.73e-06
   * - 1e4
     - 2.0e8
     - 1.59e+02
     - 2.81e-05
     - 5.41e-05
   * - 1e5
     - 2.0e10
     - **1.02e+04** (negative → NaN)
     - 1.51e-07
     - 8.94e-05
   * - 1e6
     - 2.0e12
     - **6.80e+06** (negative → NaN)
     - 7.43e-07
     - 6.71e-03

**Naive's error tracks κ.** Each row multiplies the shift by 10 and hence κ by
~100, and the naive column rises by about the same factor per step. This is the
classical conditioning result reproduced on our hardware, not an analogy to it.

**The failure is qualitative, not gradual.** From shift 1e5 the naive estimator
returns a **negative variance**. A negative variance has no square root, so a
LayerNorm built on this formulation does not produce an inaccurate number — it
produces ``NaN``. That is the difference between an accuracy tradeoff and an
unusable kernel, and it is why decision B treats stability as part of
correctness rather than as a separate concern.

**Two-pass is flat across 12 orders of magnitude of κ**, which is the
justification for LayerNorm's extra pass. The cost of that immunity is exactly
the 3R+1W traffic structure the Sprint-2 ablation spends its effort on, and the
reason LayerNorm runs ~2× slower than RMSNorm. Accuracy bought with bandwidth,
priced.

**Welford sits between**, vastly better than naive and slightly worse than
two-pass, degrading slowly. This matches the Sprint-6 §1.3 correction: its
weakness is not the variance recurrence but the running mean, which updates at
full input magnitude every element.

.. warning::

   Stated rather than hidden: past shift ~1e5 the FP32 **input** is itself
   quantized — the ULP near 1e6 is 0.0625, larger than the unit-scale signal —
   so beyond that point every estimator measures the variance of the quantized
   data. They remain comparable (same input, same oracle computed from that
   same input), but the "true" variance drifts slightly, which is why the
   two-pass column is not perfectly monotonic.

The shipped kernels on the same axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scalar estimators prove nothing about our SSVE kernels, so the shipped kernels
were run on the identical stress data (M=64, N=512), max\|error\| against the
float64-accumulating reference:

.. list-table::
   :header-rows: 1

   * - shift
     - LN V0 (exact FSQRT)
     - LN V6 (FRSQRTE+NR)
     - LN Welford
     - RMS V6
   * - 0
     - 7.15e-07
     - 1.72e-05
     - 1.72e-05
     - 1.73e-05
   * - 1e2
     - 1.30e-04
     - 1.30e-04
     - 1.02e-04
     - 6.79e-06
   * - 1e4
     - 5.37e-03
     - 5.38e-03
     - 2.16e-02
     - 3.10e-06
   * - 1e5
     - 1.23e-02
     - 1.23e-02
     - 2.14e-01
     - 5.25e-06
   * - 1e6
     - 2.07e-03
     - 2.08e-03
     - 3.02e-01
     - 5.19e-06

Three readings, in increasing order of how much they change the project's
story:

**1. The FRSQRTE accuracy cost is real but narrow.** At shift 0, V0's exact
``FSQRT``+``FDIV`` reaches 7.2e-07 against V6's 1.7e-05 — a ~24× penalty for
the ablation's winning variant, which Sprint 6 §1.3 found had never been
documented. But from shift 1e2 upward the two are **indistinguishable**. The
substitution costs accuracy only where the problem is already well conditioned.

**2. What limits LayerNorm on hard data is not the variance at all.** Two-pass
already fixed the variance — that column is flat. The residual error lives in
the *output* expression :math:`(x - \mu)`: when :math:`x \sim 10^5` and
:math:`(x-\mu) \sim 1`, FP32 carries the difference in only a handful of bits.
At shift 1e5 the ULP near 1e5 is ~1.6e-2 relative to a unit signal and the
measured error is 1.2e-2 — the two agree, so this is an **input-representation
limit that no variance algorithm can remove**. That V0 and V6 coincide here is
the evidence: the sqrt arithmetic is not the binding constraint. (The
non-monotonic dip at 1e6 is the same effect from the other side — once the
input is coarsely quantized, :math:`(x-\mu)` becomes exactly representable more
often, so the kernel's own rounding falls.)

**3. RMSNorm is flat at ~5e-06 across the entire sweep** — about 2300× better
than LayerNorm at shift 1e5. This confirms ``context.md`` §8's claim on the
shipped kernel: RMSNorm never forms a mean, so it never performs the cancelling
subtraction, in the reduction *or* in the output. Its stability is structural,
and it is the same property that makes it faster (2R+1W vs 3R+1W). **Accuracy
and throughput point the same way**, which is unusual enough to be worth
stating plainly.

RMSNorm's own limit
~~~~~~~~~~~~~~~~~~~~~

The honest counterpart, so the comparison is not one-sided: RMSNorm is immune
to shift but must accumulate :math:`\sum x^2` in FP32, which **overflows**. The
boundary is sharp and predictable — the reduction goes to ``+inf`` once
:math:`N x^2 > \mathrm{FLT\_MAX} \approx 3.4\times10^{38}`, i.e. at
:math:`|x| \approx \sqrt{3.4\times10^{38}/N}`, about 8.1e17 for N=512. A test
straddles that boundary at ±10 % to show the failure is where the arithmetic
says it is, not merely somewhere between two decades.

7b — What the streaming transition actually costs
----------------------------------------------------

Every per-call cost quoted by this project so far was **inferred**, from the
intercept of a linear fit. That instrument has already failed once: Sprint 2a
read ``t0 ≈ 1 µs/call`` from it, and Sprint 4 found that ~70 % of that was an
uncached ``cpu_supports_sme()`` sysctl in the wrapper. The fit had silently
absorbed a bug that had nothing to do with streaming mode.

``src/norm/smstart_probe.S`` measures the transition directly, with an
identical transition-free loop as the control so the loop's own branch and
counter cost is subtracted rather than attributed.

.. list-table:: Transition cost (2e6 iterations, best-of-20, Apple M4)
   :header-rows: 1

   * - variant
     - ns/iteration
     - minus control
   * - empty loop (control)
     - 0.226
     - —
   * - ``smstart`` + ``smstop`` (SM+ZA)
     - 9.292
     - **9.07**
   * - ``smstart sm`` + ``smstop sm`` (SM only)
     - 9.294
     - **9.07**

**One round trip through streaming mode costs ~9.1 ns**, and SM+ZA is identical
to SM-only within noise — so managing ``PSTATE.ZA`` is free. The ZA kernels'
problem (Sprints 3 and 6) was never the transition.

The per-call floor
~~~~~~~~~~~~~~~~~~~~

Measured, not fitted: the smallest call the kernel can be asked to perform
(M=1, N=1), timed in batches of 20 000 because a single call is at the
resolution limit of two ``Clock::now()`` reads.

.. list-table::
   :header-rows: 1

   * - component
     - ns/call
   * - measured floor
     - 46–53
   * - of which the streaming transition
     - 9.07 (~20 %)
   * - everything else
     - 37–44

The transition is about a **fifth** of the floor — real, but not what makes a
small call expensive. The remainder is prologue/epilogue (this project saves
``d8–d15`` on every call, the AAPCS64 fix from Sprints 5/6), pointer setup and
the ``inv_rms`` serialization.

.. note::

   A linear fit is deliberately **not** used here. Fitting ``t(N) = t0 + b·N``
   over N=16..512 at M=128 sweeps the footprint across cache levels, so
   per-element throughput is not constant and the intercept comes out
   *negative* — the same invalid-fit condition ``small_n_sweep()`` reports. A
   negative "fixed cost" is not a small error; it is the model announcing that
   it does not apply. Measuring the floor directly avoids the question.

What the design decision was worth, quantified: the alternative to one
streaming region per call is one transition per row, which at M=128 costs
128 × 9.07 ns ≈ **1.16 µs**, roughly 22× the entire per-call floor. The
decision in ``context.md`` §8 is right, and now right by a measured margin
rather than by assumption — but the constant itself is 9 ns. *"Streaming mode
is expensive" is true per row and false per call*, and only the second is what
these kernels do.

When does the SME kernel actually win?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RMSNorm at N=512 (a realistic transformer feature dimension), sweeping rows
against the scalar reference:

.. list-table::
   :header-rows: 1

   * - M rows
     - scalar µs
     - V7 µs
     - speedup
     - groups touched
   * - 1
     - 0.250
     - 7.708
     - 0.03×
     - 1
   * - 4
     - 2.166
     - 6.250
     - 0.35×
     - 1
   * - 8
     - 4.375
     - 6.208
     - 0.70×
     - 1
   * - **16**
     - 8.833
     - 6.167
     - **1.43×**
     - 1
   * - 64
     - 66.666
     - 6.208
     - 10.74×
     - 1
   * - 256
     - 378.375
     - 25.291
     - 14.96×
     - 4

**The crossover is at M ≈ 16 rows, and the cause is not streaming overhead.**
Look at V7's column: it is flat from M=1 to M=64, because that is **one group**.
V6/V7 process four VL-row blocks at a time — VL=16 FP32 lanes at SVL=512, so a
group is 64 rows — and a partially-filled group costs the same as a full one,
because the unused lanes are predicated off rather than skipped. At M=1 the
kernel performs 64 rows' worth of work to produce one row of result. At M=256
the time is 4× that of M=64, exactly the four groups.

So the small-tensor penalty is **group granularity**, not ``SMSTART``. The 9 ns
transition cannot explain a 6 µs plateau; the group can, exactly.

Two things follow. First, the fix is a narrower group for small M — which is
precisely the shape-specialized emission decision the JIT was built to make
(Sprint 4) and that Sprint 2b deferred as "parameterized group depth". Second,
this is the *same* granularity that made Sprint 6's threaded chunking sensitive
to alignment: a chunk that is not a multiple of 64 rows pays this partial-group
cost on every thread, which is why group-aligned chunking bought +42 %.

Sprint 7 status
~~~~~~~~~~~~~~~~

* **Instruments:** ``stability.cpp`` (three FP32 variance estimators + a
  conditioning measure + the FP32 sum-of-squares), ``smstart_probe.S``
  (transition cost with a matched control, plus an ``RDSVL`` runtime SVL query
  so the harness derives the 64-row group rather than writing the literal —
  decision D).
* **Tests:** 9 new cases. The 7a stability cases are **host-portable** and run
  on CI; the 7b probe cases are SME-guarded and skip. The probes are also
  pinned against the ``d9–d15`` clobber, since a new SME entry point is a
  latent instance of the bug this project has hit five times.
* **Suite:** 133 cases, 749 715 assertions, green on the M4.
* **Both §8 claims survive**, with corrected mechanisms: single-pass variance
  fails catastrophically (negative → NaN) exactly as claimed; streaming mode is
  not free, but at 9 ns it is a fifth of the per-call floor and the real
  small-tensor cost is group granularity.

What this sprint did not settle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **The remainder of the per-call floor is not itemised.** 37–44 ns is
  attributed to prologue/epilogue, pointer setup and serialization as a group,
  not measured component by component.
* **A narrow-group kernel for small M was not built.** The granularity cost is
  measured and its fix identified, but implementing it is a JIT emission
  change, and this sprint's scope was characterization.
