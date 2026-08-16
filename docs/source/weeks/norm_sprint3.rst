MLC-Norm Sprint 3: Hand-written SME/ZA Kernels
==============================================

Sprint 3: Hand-written SME/ZA RMSNorm kernel (``rms_norm_za``)
--------------------------------------------------------------

The second kernel *architecture* for RMSNorm: the same mathematics as the SSVE
winner V6, but staged through the SME **ZA** array instead of streamed twice.
The goal is not a new speed record, since Sprint 2a already put V6 near the
single-core streaming ceiling, but to attribute exactly what the ZA tile does,
and does not, buy a bandwidth-bound horizontal reduction (context.md §5;
ROADMAP Sprint 3, "honest expectation-setting").

The hypothesis, written before measuring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZA does not raise DRAM bandwidth. RMSNorm must see a row's **entire**
sum-of-squares before it can normalize, so V6 re-reads ``x`` in pass 2
(2R+1W). That second read is only avoidable if the whole row-block stays
resident on core between the two passes. ZA gives ~4 KB of extra on-core
storage (four 32-bit tiles ZA0..ZA3, each ``SVL_S × SVL_S`` = 16×16 FP32
on the M4) beyond the 32 Z-registers, which is exactly enough to hold one
``SVL``-row × (≤ 4·SVL)-col block. The *only* lever available is therefore to
stage ``x`` in ZA during the reduction, read it back from ZA in pass 2,
and turn 2R+1W into **1R+1W**, but only where the row fits
(``N ≤ 4·SVL`` = 64 on the M4). That is precisely the small-N,
streaming-overhead-dominated regime, so the pre-registered expectation was:
**a possible small-N win if the saved read outweighs the cost of shuffling
``x`` through ZA, and no win at all for the large-N bandwidth-bound shapes
(row does not fit → fall back to streaming → tie or lose to V6).**

Kernel design (``rms_norm_za.S``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reduction stays SSVE.** The per-row sum-of-squares is a horizontal
  reduction (``fmla`` accumulating ``x*x`` in a Z-register), never routed
  through ZA, because a matrix accumulator maps poorly onto a horizontal
  collapse (context.md §5). ZA is used as **pure residency staging**, not
  compute.
- **Pass 1 (1 memory read):** for each column, ``ld1w`` the ``x`` vector
  (SVL consecutive rows, column-major), ``fmla`` it into the accumulator,
  and ``mova`` it into a ZA tile slice.
- **inv_rms:** ``1/√(sumsq/N + ε)`` per lane, via ``FRSQRTE`` + one
  Newton-Raphson step (same sequence as V1/V6).
- **Pass 2 (1 memory write):** for each column, ``mova`` ``x`` back **from
  ZA**, apply ``γ[col]`` and ``inv_rms``, ``st1w``.
- **VLA (decision D):** tile geometry is derived from ``cntw`` (SVL), not a
  literal 16; the path split is ``N`` against ``4·SVL`` computed at runtime.
  The four ZA tiles are emitted as four ``.macro`` expansions, because the
  tile number is an instruction immediate rather than a register and so cannot
  be looped.
- **PSTATE handling:** bare ``smstart``/``smstop`` enable and disable
  **both** PSTATE.SM and PSTATE.ZA for the whole norm (one streaming
  region); every ZA slice is written before it is read, so no ``zero {za}``
  is needed, and ZA is disabled before return (no lazy-save/AAPCS64 hazard).
- **Fallback (``N > 4·SVL``):** a correct streaming two-pass (2R+1W, no ZA)
  so the kernel is right for every shape. This is not the performance
  story; V6 remains the large-N incumbent.
- A single accumulator sums in strict column order (reference order), so no
  reassociation tolerance is needed: verified at ``kTol = 1e-5`` (86 test
  cases in ``test_norm``, with tile-boundary, row-tail, mismatched-ld,
  fallback and large-magnitude-stress cases all passing on M4).

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
moves strictly *less* memory (1R+1W against 2R+1W, a real 33 % traffic
reduction in this regime). The clinching data point: the ZA kernel's own
streaming **fallback** at N=2048 (≈25 GiB/s) is *faster* than its
**ZA-resident** path at N=64 (8–11 GiB/s). Staging through ZA is the
bottleneck, not memory.

The mechanism: each element now costs two extra ``mova`` operations, one
``Z→ZA`` in pass 1 and one ``ZA→Z`` in pass 2, and ``mova`` to and from the ZA
array is throughput-limited on the M4. At the modest bandwidth these
shapes sustain (10–35 GiB/s), the memory system was never the binding
constraint that ZA relieves, so trading a DRAM read for two ``mova`` operations
is a net loss. This is the same lesson as context.md §5's reduction warning,
now confirmed for **residency staging** as well: routing a bandwidth-bound
horizontal-reduction norm through the matrix array costs more than it saves.

Reconciling with the Sprint-2c note: the "+33 % traffic reduction
available" framing referred to the *fusion arithmetic*.  The measurement
shows that the reduction is only reachable in the ``N ≤ 4·SVL`` regime (the
row must fit in ZA), and even there the ``mova`` cost dominates, so the traffic
saving never converts into a time saving. For the large-N shapes that are
bandwidth-bound, ZA cannot hold the row at all, putting it structurally out
of reach. "ZA adds nothing here, and here is why" is the pre-registered,
fully valid ablation outcome (ROADMAP Sprint 3).

DRAM-regime addendum: no footprint rescues ZA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table above tops out at a 1 MB footprint (M=4096, N=64), which is
cache-resident, so a skeptic could argue that ZA never got the chance to save a
*real* DRAM read. This addendum closes that gap by re-measuring at true-DRAM
footprints (32–64 MB, well past the 16 MB L2), in both the ZA fast path (N=64,
huge M, so the row still fits in ZA) and the streaming fallback (N=2048). This
is precisely the regime where ZA's 1R+1W traffic saving *should* pay if it ever
does.

.. list-table:: Sprint 3 DRAM addendum — ``rms_norm_za`` vs ``rms_norm_ssve_v6`` (M4, useful bytes = 1R+1W)
   :header-rows: 1

   * - shape
     - footprint
     - path
     - V6 GiB/s
     - ZA GiB/s
     - ZA vs V6
   * - M=131072, N=64
     - 32 MB
     - ZA (N≤64)
     - 22.4
     - 10.1
     - **−55 %**
   * - M=262144, N=64
     - 64 MB
     - ZA (N≤64)
     - 22.6
     - 10.1
     - **−55 %**
   * - M=4096, N=2048
     - 32 MB
     - fallback
     - 25.4
     - 11.1
     - **−57 %**
   * - M=8192, N=2048
     - 64 MB
     - fallback
     - 23.5
     - 11.1
     - **−53 %**

Verdict: **no footprint rescues ZA.** The ZA path stays pinned at ~10 GiB/s
regardless of total size, at the same ``mova``-throughput ceiling as the 1 MB
cache case, because ``mova`` rather than memory is what limits it. Meanwhile V6
*does* slow in DRAM (37 → 22 GiB/s), confirming that V6 is the kernel actually
touching the memory system, yet it still delivers ~2.2× the ZA path. The
mechanism: V6's 4-row-block reuse distance keeps each row's pass-2 re-read
L1/L2-resident even at a 64 MB footprint (22 GiB/s is far above what a genuine
2R-from-DRAM would sustain), so the DRAM read ZA "saves" was never a DRAM read.
Combined with the row-fits constraint (residency is only reachable at N≤64,
which is cache-resident anyway), the regime where ZA saves traffic and the
regime where traffic binds do not overlap, at any footprint.

.. note::

   **Known AAPCS64 hazard, re-confirmed and escalated (fix before TEIR, ROADMAP
   Sprint 5).** This is the same SMSTART/D-register clobber first hit in
   Sprint 2 and logged in ``docs/dev-notes/sprint-errors/sprint4_errors.md``
   (#6, #8), not a new discovery. Getting stable timings again required forcing
   all loop and timestamp state into memory: a fresh Release build zeroed every
   benchmark after the first streaming call. Root cause:
   ``smstart``/``smstop`` zero all of ``d8–d15`` (they alias the low bits of
   ``z8–z15``), and ``layer_norm_ssve_v6.S`` also uses ``z8–z15`` as
   accumulators, but both V6 kernels save and restore **only ``d8``**, so a
   caller's ``d9–d15`` are clobbered. This was invisible to the Catch2 tests and
   worked around in the bench with ``volatile``. The escalation: TEIR's runtime
   (Sprint 5) calls the kernel without that workaround, so the latent bug
   becomes live. The numbers above were taken with a robust standalone driver;
   the fix (save and restore ``d8–d15`` in both the ``.S`` and the JIT
   generator, then re-run the encoding-diff) is a Sprint-5 prerequisite.

- **RMSNorm architecture verdict:** SSVE **V6 stays the frozen incumbent**
  for Sprint 4's JIT. ``rms_norm_za`` is kept in the tree and the ablation
  table as a *measured, explained* negative, which is the honest-engineering
  deliverable rather than a failure.
- **LayerNorm ZA (Mariza):** gate resolved by building the prototype rather
  than taking the reasoned skip, see below. Verdict: also negative, and by
  a *larger* margin than RMSNorm's.

Sprint 3: Hand-written SME/ZA LayerNorm kernel (``layer_norm_za``)
------------------------------------------------------------------

The gated second half of Sprint 3. RMSNorm's ZA verdict was negative, but
LayerNorm's 3R+1W structure has more theoretical residency headroom than
RMSNorm's 2R+1W (its *two* extra reads, not one, are what ZA staging could
eliminate), so the gate was resolved by measuring rather than by skipping.

The hypothesis, written before measuring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RMSNorm's ``mova``-throughput finding (above) is norm-agnostic in principle,
but LayerNorm's larger headroom makes it worth a direct test rather than an
inference. Rather than the partial "3R→2R" fusion sketched in Sprint 2c
(fusing only the mean and variance sub-passes, still re-reading ``x`` once for
normalize), this prototype goes further: ``x`` is staged in ZA **once**
during the mean pass and reused from ZA for **both** the variance pass and
the normalize pass, a true 3R+1W → 1R+1W fusion and a 50 % traffic cut against
RMSNorm's 33 %. This is the most decisive version of the test available,
because it costs *three* ``mova`` operations per element (one store into ZA,
two loads back out) against RMSNorm's two (one store, one load).
Pre-registered expectation: if a 33 % traffic cut already lost 39–65 % to
``mova`` throughput, a 50 % cut needs a proportionally *larger* saving to break
even, and instead pays a proportionally larger ``mova`` tax. Expected outcome:
also a loss, and likely by a *larger* margin than RMSNorm's, which would
confirm ``mova`` throughput rather than DRAM bandwidth as the real,
norm-agnostic ceiling. A win would be the surprising result.

Kernel design (``layer_norm_za.S``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Reduction stays SSVE** (context.md §5): both the mean sum and the
  sum-of-squared-deviations are horizontal reductions (``fadd`` / ``fmla``
  accumulating in Z-registers), never routed through the ZA matrix
  accumulator. ZA is pure residency staging for ``x``.
- **Pass 1 (1 memory read):** ``ld1w`` each column's ``x``, accumulate into
  the mean sum, ``mova`` it into a ZA tile slice.
- **Pass 2 (0 memory reads):** ``mova`` ``x`` back **from ZA**, centre on the
  mean, accumulate the squared deviation.
- **inv_std:** ``FRSQRTE`` + one Newton-Raphson step (same sequence as V1/V6).
- **Pass 3 (1 memory write):** ``mova`` ``x`` back **from ZA a second time**,
  apply ``(x-mean)*inv_std*gamma[col]+beta[col]``, ``st1w``.
- **VLA (decision D):** tile geometry derived from ``cntw`` (SVL); the path
  split is ``N`` against ``4·SVL`` computed at runtime, with the same
  four-tile ZA layout as ``rms_norm_za``.
- **PSTATE handling:** bare ``smstart``/``smstop`` for the whole norm (one
  streaming region), every ZA slice written before read, ZA disabled before
  return.
- **Fallback (``N > 4·SVL``):** a correct streaming three-pass (3R+1W, no
  ZA, single-block predicated).  It is deliberately not the
  4-block-contiguity-grouped structure V6 uses, since this fallback is a
  correctness net rather than the performance story.
- Single accumulators (mean, variance) run in strict column order across all
  four tile sections, matching the reference's summation order exactly, so
  they are verified at ``kTol``/``kAbsMarginNR`` with no reassociation
  widening needed.  Verified against ``layer_norm_ref`` (tile boundaries, row
  tails, mismatched leading dims, the ``N > 4·SVL`` fallback, and the
  large-magnitude stress case; 5 new test cases in ``test_norm``, all passing
  on M4).

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

Verdict: ZA loses decisively, and by more than RMSNorm's loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZA-resident fast path is **51–68 % slower than V6**, a larger loss
than RMSNorm's 39–65 %, exactly as the pre-registered hypothesis predicted.
The mechanism is the same one RMSNorm exposed: ``mova`` throughput, not DRAM
bandwidth, is the binding constraint at these shapes, and LayerNorm's
prototype pays it three times per element instead of twice. The 50 % traffic
cut (against RMSNorm's 33 %) is real, and it converts into *more* saved DRAM
time than RMSNorm's cut did, but that extra saving still is not enough to
cover the extra ``mova`` operation's cost, so the margin gets *worse* rather
than better. This is the cleanest confirmation available that the RMSNorm
finding generalizes: trading DRAM reads for ZA residency is a structural loss
on this hardware for a bandwidth-bound horizontal-reduction norm, regardless of
how many reads are being traded away.

The fallback path (no ZA, N > 4·SVL) is 11–12 % slower than V6, which is
expected, since this fallback is a plain single-block three-pass without V6's
4-row-block contiguity grouping or multi-accumulator ILP (the same pattern
``rms_norm_za``'s own fallback shows against V6, there by 28–35 %); it exists
for correctness on every shape, not to compete with V6.

- **LayerNorm architecture verdict:** SSVE **V6 stays the frozen incumbent**
  for Sprint 4's JIT, for both norms. ``layer_norm_za`` is kept in the tree
  and the ablation table as a *measured, explained* negative, and its
  larger-than-RMSNorm loss margin is itself evidence: it closes the question
  Sprint 2c left open (whether LayerNorm's larger headroom might tip ZA
  staging into a win) with a direct, decisive answer rather than an
  inference from RMSNorm alone.

Correctness note: an eps register-stash bug, found and fixed during this work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While building ``layer_norm_za``, an empirical check (an all-zero RMSNorm row,
sumsq = 0 exactly) surfaced a real correctness bug affecting **every**
existing SSVE/ZA kernel except the two original LayerNorm baselines
(``layer_norm_ssve``/``_v1``). Each stashed ``eps`` in the callee-saved S8/D8
register before ``smstart``, intending to read it back afterwards, either via
a stack slot that still held the *caller's* old ``d8`` (saved by
``str d8`` *before* the eps ``fmov`` overwrote the register), or by reading
S8/D8 directly post-``smstart``, which also fails because streaming-mode
entry clobbers D8 too, not only D9–D15 as had been assumed. Every existing
tolerance-based test passed regardless, since eps's effect on non-degenerate
data is far below the FP32 comparison tolerance; the bug is only observable
when sumsq or variance is exactly zero, where eps is the only thing standing
between a valid reciprocal and ``1/0 = inf`` (after which ``inf * 0 = NaN``
propagates through the normalize step). Fixed in all 13 affected kernels by
stashing eps to a **dedicated stack slot before smstart** and reloading from
that same address afterwards, since memory, unlike any register, is unaffected
by a PSTATE.SM transition. Two new regression tests lock this in (all-zero
RMSNorm row, all-constant LayerNorm row, both required to be finite). Full
test suite re-verified green after the fix (352 473 assertions, 93 cases).

Sprint 3 status
~~~~~~~~~~~~~~~

- **RMSNorm kernel:** ``rms_norm_za.S``, verified on M4 (86 test cases in
  ``test_norm``), guarded by ``cpu_supports_sme()``, VLA, one streaming
  region with correct PSTATE.SM/ZA handling. Measured verdict: ZA residency
  is 39–65 % slower than V6 in its fast path.
- **LayerNorm kernel:** ``layer_norm_za.S``, verified on M4 (5 test cases,
  full 3-pass ZA residency), same guards and VLA discipline. Measured
  verdict: ZA residency is 51–68 % slower than V6 in its fast path, a
  larger loss than RMSNorm's, confirming that the mechanism generalizes.
- **Lever attributed for both:** the loss is the ``mova`` staging cost,
  quantified against V6 and against each kernel's own non-ZA fallback.
- **V6 (both norms) frozen as the incumbent for Sprint 4's JIT.**
- **Correctness fix:** the eps register-stash bug above, fixed across all
  13 affected kernels with two new regression tests.
