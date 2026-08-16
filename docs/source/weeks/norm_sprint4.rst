MLC-Norm Sprint 4: JIT Generation (mini_jit::Norm)
==================================================

Sprint 4: JIT generation (mini_jit::Norm)
-----------------------------------------

Goal: generate the norm kernels **at runtime** instead of hand-writing them,
via a ``mini_jit::Norm`` generator following the week-6 ``Unary`` pattern.
``generate(ntype)`` emits instruction words through ``InstGen`` into a week-5
``JitEngine`` executable buffer; ``get_rms_kernel()`` / ``get_layer_kernel()``
return a reusable function pointer.  Emission is a one-time cost.

What gets emitted, and what deliberately does not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generator emits the **measured Sprint-2 winners**, ``rms_norm_ssve_v6``
and ``layer_norm_ssve_v6``, transcribed 1:1 (same registers, same
instruction order).  The ZA path is **not** emitted: the ROADMAP gated it on
"if it earned its row", and Sprint 3's verdict was a decisive loss for both
norms (39–68 % slower than V6, ``mova``-throughput-bound).  Emitting a
kernel architecture that measurement rejected would add encoder surface and
maintenance for a path no shape selects, so the reasoned skip *is* the
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
   from its function address at runtime, so the reference is what the
   toolchain assembled and can never go stale.  Green for both
   norms: **157/157 words (RMSNorm) and 208/208 words (LayerNorm)**
   identical.  (These counts are as of Sprint 5: the kernels were 143 and
   194 words when this section was first written, and grew by the 14
   save/restore instructions the Sprint-5 ``d8–d15`` AAPCS64 fix added to
   each.  The diff was re-run after that change and stayed green, which is
   the point of anchoring it to the linked kernel rather than to a copy.)
   A green diff means the JIT kernel *inherits* the trust of a kernel that
   already passed the full Sprint-2/3 suite (including the eps-stash fix),
   instead of re-earning it.
3. **Execution anyway.**  The emitted kernels are run against the C++
   reference on the same shape set as the V6 tests (groups, tails,
   mismatched leading dimensions, N = 1, stress inputs, the zero-variance
   eps regression), which also exercises the ``JitEngine`` buffer path
   (``MAP_JIT``, W^X toggling, icache invalidation).  All green on M4.

Layers 1–2 are host-portable and run on CI; layer 3 skips without SME.

What the exactness found: two latent InstGen bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparing the golden words surfaced two week-6 encoder bugs that every
behavioral test had passed over (full detail in
``docs/dev-notes/sprint-errors/sprint4_errors.md``):

- ``sve_ptrue_all(fp32)`` emitted ``PTRUE P.B`` (size bits 00), not the
  ``PTRUE P.S`` its own comment promised.  The defect was functionally masked
  because an ALL-true ``.B`` predicate governs ``.S`` elements identically.
  Fixed and pinned by test; week-5/6 suites re-verified green.
- ``sme_smstart_sm()`` in fact encodes ``SMSTART`` (SM **and** ZA,
  ``MSR SVCRSMZA``), not ``SMSTART SM``.  This is correct for Gemm (which
  needs ZA) but wrong by name, and wrong for the norm kernels, which run
  SM-only to avoid the ZA lazy-save hazards.  It was left in place for its
  dependents; correctly named ``sme_smstart_sm_only()`` /
  ``sme_smstop_sm_only()`` were added alongside.

"It assembles and runs" is not "it is the instruction you meant"
(CLAUDE.md §10).  An encoding diff checks what the code *is*, and caught in
minutes what weeks of behavioral tests structurally could not.

A harness bug found by the parity run, and a Sprint-2a correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first parity benchmark showed the word-identical JIT kernel **+76 %
faster** than the hand-written one at M=128/N=64, which is physically
impossible for identical code, so the gap had to be the call path.  Cause: the
hand-written kernels are benchmarked through their ``mini_jit::norm`` guard
wrappers, and ``cpu_supports_sme()`` performed an uncached ``sysctlbyname``
**syscall on every call** (~1 µs), while the JIT pointer is called directly.
Fix: cache the result in a static, since CPU features cannot change at runtime.

Consequence for earlier numbers: the Sprint-2a small-N sweep ran through the
same wrapper, so its headline "fixed cost t0 ≈ 1.4 µs/call, overhead =
streaming work at N ≈ 41" was **~70 % syscall**.  With the guard fixed, the
N=16 call drops from 1.375 µs to ~0.42 µs, throughput is flat (~29 GiB/s)
from N=32 upward, and the true fixed per-call cost is too small for the
linear fit to resolve (intercept within noise of zero, ≲0.4 µs).  The
qualitative Sprint-2a conclusions stand (a fixed per-call cost exists; the
N=64 dip is real), but the *magnitude* was harness, not hardware, and is
recorded here as a correction in the same spirit as Sprint 2b's correction of
the 2a residency interpretation.

Fixing the fix: the sweep's GiB/s column then printed 0.00, because the now
trivially-inlinable guard shifted register allocation so that the ``1/2^30``
constant inside ``to_gibs`` landed in a callee-saved FP register that
``SMSTART`` zeroes.  ``volatile`` inputs cannot protect a hoisted
*constant*; the robust fix separates the bench loop from the print loop so
that no SME call sits between computation and output.  (This is the third
appearance of the D9–D15 hazard in this project, each time in a new disguise.)

Parity and emission cost (M4, corrected harness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Emission (one-time): **~4.7 µs** for RMSNorm and **~4.8 µs** for LayerNorm
(143 and 194 words at the time of this measurement; 157 and 208 after the
Sprint-5 ABI fix).  It is recouped after a handful of small-shape calls, and
the pointer is reused thereafter.

.. list-table:: JIT-emitted V6 vs hand-written V6 (GiB/s, useful bytes, % of the ceiling at each shape's footprint)
   :header-rows: 1

   * - Shape (M×N)
     - ceiling
     - RMS hand
     - RMS JIT
     - Δ
     - LN hand
     - LN JIT
     - Δ
   * - 128×64
     - 104.7
     - 38.56 (36.8 %)
     - 38.56 (36.8 %)
     - +0.0 %
     - 16.46 (15.7 %)
     - 16.65 (15.9 %)
     - +1.1 %
   * - 128×2048
     - 116.0
     - 38.52 (33.2 %)
     - 38.52 (33.2 %)
     - +0.0 %
     - 16.51 (14.2 %)
     - 16.51 (14.2 %)
     - −0.0 %
   * - 1024×2048
     - 115.6
     - 38.44 (33.3 %)
     - 38.45 (33.3 %)
     - +0.0 %
     - 16.49 (14.3 %)
     - 16.48 (14.3 %)
     - −0.0 %
   * - 4096×2048 (DRAM)
     - 63.3
     - 25.84 (40.8 %)
     - 26.21 (41.4 %)
     - +1.5 %
     - 14.42 (22.8 %)
     - 14.44 (22.8 %)
     - +0.1 %

Percentages restated in Sprint 6 (§1.2): the table originally divided all four
shapes by 59.5 GiB/s, the ceiling at a 256 MiB footprint, and only the last row
is near that regime.  The parity conclusion is untouched, since it rests on the
hand-vs-JIT Δ columns, which are ratios.

Parity holds within noise at every shape, which is exactly what a
word-identical kernel must show once the call paths are equalized.  The mmap'd
JIT page and the executable's text section make no measurable difference.

.. note::

   **What this section measures today.**  The table above is the Sprint-4
   measurement, when the generator emitted V6.  ``main_norm`` now asks for
   ``isa_t::automatic``, which is what a real caller gets, so on an SME2 host it
   emits **V7** and benchmarks it against the hand-written **V7**.  Until that
   was fixed the hand-written side stayed V6 regardless, which showed up as a
   ~10 % "JIT speed-up" in the DRAM regime that was really V7 against V6.  The
   parity conclusion is unchanged: word-identical code, parity within noise.

Sprint 4 status
~~~~~~~~~~~~~~~

- ``mini_jit::Norm`` emits both SSVE V6 winners; ZA path skipped with its
  measured rationale (Sprint 3).
- 30 new ``InstGen`` encoders, each pinned to a toolchain golden word;
  two latent week-6 encoder bugs found and resolved.
- Encoding diff green for both norms (143 + 194 words at the time; 157 + 208
  after the Sprint-5 ABI fix, re-verified); JIT kernels verified against the
  reference on M4; full suite green.
- Benchmark parity within noise at all shapes; emission ~5 µs one-time.
- Harness corrections: ``cpu_supports_sme()`` syscall caching (with the
  Sprint-2a t0 restatement) and the bench/print loop separation.
- ``ctest`` discovery fixed repo-wide (``enable_testing()`` ordering).

Sprint 4 addendum: a benchmark shape that is unambiguously true-DRAM
--------------------------------------------------------------------

**Goal:** every "true-DRAM" kernel number so far (the 2b/2c/4 tables' 64 MB
``M=4096, N=2048`` row) rests on one argument, namely that the footprint
exceeds the M4's 16 MB L2. That argument does not rule out a **larger shared
SLC** sitting between L2 and DRAM on M-series and absorbing part of the 50
repetitions ``bench()`` runs back-to-back on the same buffer. The roofline
probes (Sprint 2a) already use a footprint proven to sit beyond *any* plausible
M-series cache, 128 MiB per array, because that is what makes ``peak_ssve``
trustworthy as a ceiling. No kernel had ever been measured at that same
scale, so every "% of roofline" number to date compares a possibly
cache-assisted numerator against a guaranteed-DRAM denominator.

What was added
~~~~~~~~~~~~~~~

A new benchmark section in ``apps/main_norm.cpp`` (``main_norm``, Section 7)
uses shape ``M=4096, N=8192``, chosen so that ``M×N`` equals ``PROBE_N``
exactly, giving the identical 128 MiB-per-array footprint as
``measure_peak_ssve_1core``. Kernel and probe now sit in the same,
unambiguously-DRAM regime, so the ratio between them is finally a defensible
one. It benchmarks the scalar reference, hand-written V6, ZA residency, and the
JIT-emitted kernel, for both norms, at 10 repetitions instead of 50 (each call
already moves 256 MiB, so more repetitions buy no extra precision at real time
cost).

.. admonition:: This addendum found half of Sprint 6 §1.2, two sprints early
   :class: important

   The paragraph above states the problem almost exactly: *"every '% of
   roofline' number to date compares a possibly cache-assisted numerator
   against a guaranteed-DRAM denominator."*  That is §1.2.

   Sprint 4 fixed it by moving the **numerator** to DRAM, benchmarking a
   shape whose footprint matches the probe's.  That makes the *one* ratio it
   measures valid, and it is why the 256 MiB column below needed no
   correction.  What it did not do is generalize: every other shape in the
   report stayed cache-resident and kept the DRAM denominator.

   Sprint 6 fixed the other half by moving the **denominator** to match each
   shape, measuring the ceiling as a curve.  Either fix makes one comparison
   valid; only the second makes all of them valid.  The 64 MiB column below is
   restated accordingly (ceiling 63.3 GiB/s at that footprint, not 59.5).

Findings (Apple M4; each column against the ceiling at its own footprint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: True-DRAM shape (M=4096, N=8192, 256 MiB total) vs the 64 MiB shape
   :header-rows: 1
   :widths: 22 16 16 16

   * - variant
     - 64 MiB (M=4096,N=2048), ceiling 63.3
     - 256 MiB (M=4096,N=8192), ceiling 59.5
     - drop
   * - RMS V6 hand-written
     - 25.84 (40.8 %)
     - 20.22 (34.0 %)
     - **−22 %**
   * - RMS V6 JIT-emitted
     - 26.21 (41.4 %)
     - 20.26 (34.0 %)
     - **−23 %**
   * - LN V6 hand-written
     - 14.42 (22.8 %)
     - 13.12 (22.0 %)
     - **−9 %**
   * - LN V6 JIT-emitted
     - 14.44 (22.8 %)
     - 13.13 (22.1 %)
     - **−9 %**

The correction also decomposes the drop, which the single-denominator version
could not.  The ceiling itself falls 63.3 → 59.5 GiB/s (−6 %) between the two
footprints, so that much of each drop is the machine rather than the kernel:

* **RMSNorm −22 %** = −6 % ceiling + **−17 % efficiency** (40.8 % → 34.0 %).
  Most of it is genuinely the kernel losing cache assistance, as the original
  text concluded.
* **LayerNorm −9 %** = −6 % ceiling + **−3.5 % efficiency** (22.8 % → 22.0 %).
  Here the drop is *mostly the ceiling*: LayerNorm barely changes its
  efficiency at all, which strengthens the original reading that its extra
  arithmetic per byte keeps it closer to compute-bound.

The **64 MB shape was still partially cache-assisted**, which is a real,
measurable gap rather than noise. RMSNorm drops further than LayerNorm (−22 %
against −9 %) because RMSNorm is the more bandwidth-sensitive kernel (2R+1W,
less FP work per byte) and so is more exposed when cache assistance is removed;
LayerNorm's extra arithmetic per byte already keeps it closer to compute-bound,
so losing the cache assist costs it less.

ZA residency and JIT parity both replicate at the new scale:

.. list-table:: ZA vs V6, true-DRAM shape (M=4096, N=8192)
   :header-rows: 1

   * - norm
     - V6 GiB/s
     - ZA GiB/s
     - ZA vs V6
   * - RMSNorm
     - 20.22
     - 10.06
     - **−50 %**
   * - LayerNorm
     - 13.12
     - 7.41
     - **−44 %**

Both are ``N=8192 > 4·SVL=64``, so ZA runs its streaming fallback with no
ZA-resident fast path.  This extends the Sprint-3 fallback-vs-V6 comparison
(previously only measured up to N=2048, where the loss was 28–35 % for
RMSNorm and 11–12 % for LayerNorm) to a much larger N deeper in the DRAM
regime, where the gap widens further for both norms. The JIT ties
hand-written within noise (RMS +0.2 %, LN +0.1 %), so parity holds at this
scale too, as expected from the encoding-diff guarantee (Sprint 4).

Sprint 4 addendum status
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Benchmark:** ``main_norm`` Section 7 added; shape matches ``PROBE_N``
  exactly (128 MiB/array); scalar reference, V6, ZA, and JIT all measured
  for both norms.
- **Finding:** the 64 MB shape used throughout Sprints 2b/2c/4 understated
  the true-DRAM cost by 9–23 % (LayerNorm and RMSNorm respectively), recorded
  here as a correction to the existing tables' "true-DRAM" framing, in the
  same spirit as the Sprint-2b residency correction and the Sprint-4 syscall
  correction above.
- **No kernel or build changes.**  This is a measurement-only addition; all
  existing ablation tables and their conclusions stand as the correct
  comparison *among variants at that shape*, and only the "is 64 MB really
  DRAM" premise is revised.
