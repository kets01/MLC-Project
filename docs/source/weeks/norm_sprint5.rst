MLC-Norm Sprint 5: TEIR Integration (norm as a compiler primitive)
==================================================================

Sprint 5: TEIR integration
--------------------------

Goal: let the week-7 TEIR runtime place the norm in a loop nest and invoke the
JIT-generated kernel, i.e. the ``context.md`` flow end to end, then parallelize
the outer row axis with OpenMP and measure how far the threaded aggregate gets
toward the chip-wide roofline.

The sprint delivered that, but most of its engineering went into three
pre-existing defects that the integration exposed. Each is recorded below with
how it was found, because in every case the *test that should have caught it*
was structurally incapable of doing so.

.. _d8-d15-canonical:

Prerequisite: the d8–d15 AAPCS64 violation
------------------------------------------

.. admonition:: The canonical account of this defect
   :class: note

   This bug recurs across Sprints 3–8 and **this section is the single full
   explanation of it**; later sprints reference it rather than restate it.
   In brief, for readers arriving from elsewhere: AArch64's procedure-call
   standard makes the low 64 bits of ``v8–v15`` (i.e. ``d8–d15``)
   callee-saved, so a kernel that clobbers them without saving and restoring
   them is **ABI-nonconforming even if a standalone benchmark appears to
   work**. It was found and fixed here; the concerning part is how long it
   survived, which is why the history is kept rather than tidied away.

Both frozen V6 kernels saved and restored only ``d8``, while
``smstart``/``smstop`` clobber the whole callee-saved ``d8–d15`` range (they
alias the low 128 bits of ``z8–z15``). A caller holding live FP values in
``d9–d15`` across the call had them silently zeroed.

This was invisible to the entire Catch2 suite, since no test happens to hold a
live value in those registers across a single call, and it had been *worked
around* caller-side in the benchmark harness with ``volatile``. TEIR's runtime
calls the kernel from a generated loop nest that has no such workaround, so the
latent bug would have become a live one.

Fixed in ``rms_norm_ssve_v6.S`` (frame 80 → 128 B) and
``layer_norm_ssve_v6.S`` (80 → 144 B), mirrored instruction-for-instruction in
``mini_jit::Norm``, and the Sprint-4 encoding-diff re-run to confirm that the
generator and the hand-written kernels remain **bit-identical** (157 and 208
words). Two regression tests pin ``d9–d15`` with local register variables and
an ``asm`` barrier around the call, which is the only construct that observes
the physical registers rather than a compiler-tracked copy.

A real ``.teir`` loader
------------------------

``load_teir()`` was a stub: it pattern-matched the *filename* and returned a
hardcoded C++ tree, never opening the file. ``TeirParser`` now parses the
course format properly, covering tensors, axes with per-tensor strides,
primitives with ``M``/``N``/``K`` axis groups and metadata, and the schedule
including ``guard first(@axis)`` and multi-child iteration steps.

Two IR additions were needed: a ``Sequence`` node (``Iteration::body`` holds a
single child, but ``contraction.teir`` schedules ``children [@inv_zero,
@inv_gemm]``) and a per-axis iteration index in ``RuntimeContext`` so that a
guard can ask whether it is on the first step of its axis.

Layouts are **derived from the file**, never assumed: for each operand the
parser finds which direction has element stride 1 and takes the other
direction's stride as the leading dimension, rejecting anything contiguous in
neither direction (streaming mode has no gather or scatter on this target, so
no kernel could serve it). All five files, the three course files plus the new
``rmsnorm.teir`` / ``layernorm.teir``, now execute from disk.

Defect 1: the week-6 GEMM was wrong for non-uniform data
---------------------------------------------------------

Driving ``matmul.teir`` through the runtime crashed. The cause was not the new
code:

* For ``trans_b=0`` the k-loop advanced B by **one element** per step and
  loaded 16 contiguous floats, giving a sliding window over B's raw memory
  rather than a row of B. FMOPA needs B's N-direction as a vector; for
  column-major B that direction is not contiguous, so it requires a transpose
  the generator never performed. The result is not a GEMM for any non-uniform B.
* It zeroed ZA per call (``C = A*B``) although its own documentation said
  ``C += A*B``; the TEIR schedules (guarded ``Zero`` then accumulation over
  outer K chunks) and the week-3 hand-written kernels both require
  accumulation.
* It used ``W12`` as the MOVA slice index while ``X12`` held the A byte-stride.
  ``MOVZ W12`` zeroes the upper half of ``X12``, so every macro-tile after the
  first ran with a corrupted stride.

None of this was visible to the existing test, which multiplies **all-ones**
matrices and checks ``MC[0] == K``. Any 16 contiguous floats of an all-ones
matrix *are* the correct operand vector, so the test cannot distinguish a
correct kernel from an addressing bug.

The generator was redesigned: K is processed in 16-wide chunks; C is loaded
into the ZA accumulators first (accumulate semantics); operands whose required
direction is not contiguous are staged through a ZA tile (contiguous loads into
horizontal slices, transposed vectors read back from vertical slices, which is
the ZA use ``context.md`` §5 sanctions, 2-D movement rather than reduction);
strides live in ``x6``/``x7``/``x16`` so that ``x12–x15`` stay free for slice
indexing. All eight ``(trans_a, trans_b, trans_c)`` combinations are supported,
and invalid shapes now fail loudly instead of emitting wrong code.

**Arbitrary K** (the course spec) is supported: full 16-wide chunks plus one
predicated remainder chunk. The remainder only affects the ZA-staged paths,
whose stage loads fetch 16 k-contiguous elements and would read past the valid
K region, so those loads use a ``WHILELO`` predicate; the zeroed inactive
lanes land in ZA columns the remainder's k-loop never extracts.

Verification is now against a scalar reference on data where every element is
distinct, with a non-zero initial C so that the accumulate path is exercised
too: all eight layouts at 48×48×32 and 48×48×40, both ``.teir`` shapes
(32×64×512 and 96×64×256), and remainder-only/K=1 cases.

Defect 2: ``Unary``'s ``trans_b`` was never implemented
--------------------------------------------------------

``transposition.teir`` describes a genuine transpose (its output tile is
column-contiguous while its input tile is row-contiguous), which the old
runtime served with a contiguous ``memcpy``-equivalent, passing a test that
filled the tensor with a single value and checked two elements.

The course's ``Unary.h`` has always specified ``trans_b`` as "0 if B is stored
in column-major order, 1 if row-major", i.e. ``B := op(A)`` with **opposite**
storage, the memory transpose. The repository's implementation ignored the
flag entirely (``[[maybe_unused]]``). It is now implemented as a ZA-staged
transpose for ``identity`` and ``relu``, which is where this functionality
belongs; the parser emits it by setting ``trans_b`` from the file's strides.

Defect 3: the benchmark was measuring an unoptimized build
------------------------------------------------------------

The first threaded numbers put RMSNorm *above* the chip-wide ceiling (117 %),
which is a statement that something is wrong rather than a result. Three
separate causes, peeled off in order:

1. **The chip probe gave each thread one equal slice.** On the M4 the cores are
   heterogeneous (4 P + 6 E), so wall time was set by the slowest E-core while
   the P-cores idled: it measured slowest-core × T, not the memory system.
   Replaced with dynamically distributed chunks.
2. **A race in that fix.** Handing out ``n_chunks × R`` work items and mapping
   them with ``idx = c % n_chunks`` let two threads write the same chunk
   concurrently (different repetitions), causing cache-line ping-pong and a
   data race. The repetitions now run as barrier-separated passes, so each
   chunk has exactly one owner per pass and a full 256 MiB sweep happens before
   any chunk is revisited (which also keeps the probe in the DRAM regime).
3. **The build had no optimization flags at all.** ``CMAKE_BUILD_TYPE`` was
   empty, so the C++ probes and references ran at ``-O0`` while the
   hand-written ``.S`` kernels were unaffected. The tell was single-core NEON
   reading 10.8 GiB/s against SSVE's 59.4: NEON is *faster* than streaming
   mode on this chip (79.5 vs 59.5), so the C++-side numbers had to be wrong.

Rebuilt per ``context.md`` §13 (``-DCMAKE_BUILD_TYPE=Release``), the three
ceilings reproduce the Sprint-2a figures and are finally consistent with each
other (chip > 1-core NEON > 1-core SSVE):

.. list-table::
   :header-rows: 1

   * - Ceiling
     - GiB/s
   * - single-core NEON (compiler-vectorized scale-add)
     - 79.75
   * - single-core SSVE (streaming ``LD1W``/``ST1W`` probe), the kernel roofline
     - 59.49
   * - chip-wide (10 threads, NEON), the Sprint-5 threading target
     - 85.79

A separate lesson: an earlier run of the threading study used M=N=2048
(16 MiB per array, partly L2-resident and reused across timing repetitions) and
compared it against a probe using 128 MiB arrays. That is the Sprint-2a
"peak of *what*" trap in a new guise, a cache-assisted kernel measured against
a DRAM-only ceiling. The study now runs at M=4096, N=8192 (256 MiB working
set), matching the probe's regime.

.. note::

   Because of that fix, **this section needed no re-baselining in Sprint 6.**
   Numerator and denominator are both at 256 MiB, so the percentages here were
   already footprint-matched, the one place in the report where they were.
   The Sprint-6 §1.2 correction generalizes this same reasoning to every other
   shape, by measuring the ceiling as a curve instead of moving every workload
   to DRAM.

Threaded scaling (M4, Release, 256 MiB working set)
-----------------------------------------------------

Row axis marked ``is_parallel``; the kernel stays single-tile and TEIR
parallelizes across row chunks. Every configuration is verified against the C++
reference before its GiB/s is reported (decision B), and the kernels are called
from OpenMP worker threads with **no** ``volatile`` workaround, which is only
safe because of this sprint's ``d8–d15`` fix.

.. list-table:: RMSNorm
   :header-rows: 1

   * - threads
     - chunk rows
     - GiB/s
     - speed-up
     - % of chip ceiling
   * - 1
     - 4096
     - 20.28
     - 1.00×
     - 23.6 %
   * - 2
     - 2048
     - 26.18
     - 1.29×
     - 30.5 %
   * - 4
     - 1024
     - 34.35
     - 1.69×
     - 40.0 %
   * - 8
     - 512
     - 38.79
     - 1.91×
     - 45.2 %
   * - 16
     - 256
     - 42.80
     - 2.11×
     - 49.8 %

.. list-table:: LayerNorm
   :header-rows: 1

   * - threads
     - chunk rows
     - GiB/s
     - speed-up
     - % of chip ceiling
   * - 1
     - 4096
     - 13.26
     - 1.00×
     - 15.4 %
   * - 2
     - 2048
     - 17.92
     - 1.35×
     - 20.8 %
   * - 4
     - 1024
     - 22.40
     - 1.68×
     - 26.1 %
   * - 8
     - 512
     - 24.28
     - 1.83×
     - 28.2 %
   * - 16
     - 256
     - 25.80
     - 1.94×
     - 30.0 %

Reading the numbers: RMSNorm scales 2.11× to about half the chip ceiling and
LayerNorm 1.94× to 30 %. Scaling saturates well before the ceiling in both
cases.

.. admonition:: Corrected in Sprint 6 (§2.2–2.3) — part of this curve was the harness
   :class: warning

   The original explanation for the flattening, *"past 8 threads the M4's
   E-cores contribute little bandwidth, and the row-chunked schedule gives each
   thread a strided walk"*, attributed to P/E heterogeneity what was largely a
   **sawtooth produced by this harness's own chunk misalignment**.

   At thread counts where ``M / threads`` is not a multiple of the kernel's
   4·VL group, the measured throughput collapsed.  With group-aligned
   (256-row) chunking the effect disappears exactly where predicted and
   nowhere else:

   .. list-table:: RMSNorm V7, naive vs group-aligned chunking (256 MiB)
      :header-rows: 1

      * - threads
        - naive chunking
        - group-aligned
        - Δ
      * - 1 / 2 / 4
        - 24.77 / 29.29 / 35.81
        - 24.80 / 29.35 / 35.85
        - ~0 % (control)
      * - 6
        - 29.69
        - **42.24**
        - **+42.3 %**
      * - 8
        - 40.98
        - 40.71
        - −0.7 % (control)
      * - 10
        - 29.20
        - **41.86**
        - **+43.4 %**
      * - 16
        - 45.77
        - 44.75
        - −2.2 % (control)

   The zero-effect rows are the control: they are the thread counts that were
   *already* group-aligned, and they show no change, which rules out a generic
   "aligned is faster" explanation.

   With aligned chunking the curve is monotonic and the real shape is a knee at
   ~4–6 threads with genuine flatness from 8 (8 → 16 buys ~2 %).  **The
   mechanism behind the base-offset sensitivity was not identified**: a 64 B-
   aligned offset is still 11 % slower than the allocation start, so it is not
   simply cache-line alignment.  The fix is justified by its control rather
   than by an explanation, and is recorded that way.

This is more headroom than Sprint 2a predicted. That prediction ("one core
already sustains ~66 % of the chip aggregate, so threading buys ≤1.5×") was
formed at cache-assisted shapes; in the true-DRAM regime a single core reaches
only 20.3 of 85.8 GiB/s (24 %), so there was proportionally more to gain. The
prediction's *premise* was regime-specific, not its arithmetic.

LayerNorm stays roughly 1.7× behind RMSNorm throughout, consistent with the
structural 3R+1W vs 2R+1W traffic difference established in Sprint 2c.

Defect 4: ``test_week3`` hung on SME hardware
-----------------------------------------------

While measuring, ``test_week3`` was observed pinning a core at 100 % for over
40 minutes. Bisecting by section: every GEMM section and the ReLU/Zero unary
sections complete in seconds, while the **"Identity Transpose Test"** never
completes.

That section calls ``identity_16_16_asm(..., trans_b=1)``, whose implementation
uses a scatter store with vector offsets
(``st1w {z0.s}, p0, [x1, z31.s, uxtw #2]``) inside a ``smstart sm`` region.
Scatter/gather with vector offsets is not available in Streaming SVE without
``FEAT_SME_FA64``, which Apple has not implemented.  This is the exact
constraint documented in our own ``rms_norm_ssve.S`` and the reason the norm
kernels use ``LD1W``/``LD1RW``/``ST1W`` with scalar bases throughout.

CI never catches this because the CI runners have no SME, so the whole test
case is skipped: it reports green without executing a single instruction of
that kernel. It also blocked a full ``ctest`` run from completing on the M4.

Fixed with the same ZA-staged transpose used in ``Unary``'s ``trans_b=1``
path: load A's 16 columns (each contiguous) into ZA *horizontal* slices, so
``za0[j][i] = a[i + j*ld_a]``; then vertical slice ``i`` is
``{ za0[j][i] for j }`` = row ``i`` of A, which is exactly the 16 contiguous
floats row ``i`` of B needs. ``smstart``/``smstop`` are now the bare forms so
that PSTATE.ZA is enabled alongside PSTATE.SM and disabled again before return.
Verified against ``identity_16_16_cpp``; the section completes in well under a
second instead of spinning indefinitely.

This stays within the assignment: the SME chapter's Task 1 is "SSVE unary
kernels for **permutation**, zero and ReLU", the permutation kernel is the
``trans_b=1`` case, and that chapter teaches the ZA array, tiles and tile
slices. No implementation technique is prescribed, and with scatter
unavailable there is no pure-vector alternative short of element-at-a-time
stores.

Also registered ``Week7Tests`` with CTest: every other week calls
``add_test(...)`` but week7 never did, so the TEIR tests had been invisible to
``ctest`` and to CI.

Status
------

Done: the ``d8–d15`` prerequisite; a real ``.teir`` parser driving all five
files; the norm registered as a TEIR primitive and verified through the full
path against the C++ reference; OpenMP row-parallel execution with measured
scaling against a re-validated chip ceiling; and the week-6 GEMM, week-6
``Unary`` and week-3 transpose defects fixed with tests that can detect them.

**A full ``ctest`` now completes on the M4 for the first time, 7/7 suites
green.** It previously never finished, which is why the roofline probes were
being distorted by a spinning core.

Open: ``bench_week7``'s matmul now runs a genuine 8192³ contraction, so its
timing is not comparable to the numbers it printed while the generator was
producing wrong results.
