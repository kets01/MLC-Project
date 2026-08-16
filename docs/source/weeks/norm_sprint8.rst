MLC-Norm Sprint 8: Correctness, Dispatch & Provenance Hardening
===============================================================

Sprint 8: making the results defensible
---------------------------------------

This sprint contains no new kernel and no new optimization. It fixes a genuine
API defect, and it closes the gap between "these numbers are true" and "these
numbers are *demonstrably* true", which for a performance project is the
difference between a result and a claim.

The API defect: a silent no-op
--------------------------------

Until this sprint, the only entry points into the norm kernels were the
ISA-specific ones, and every one of them was written as:

.. code-block:: cpp

   void layer_norm_ssve(...) {
       if (!cpu_supports_sme()) return;      // <-- silently does nothing
       ::layer_norm_ssve(...);
   }

So on any CPU without SME, such as an M1/M2 CI runner or any non-Arm host,
ordinary calling code

.. code-block:: cpp

   layer_norm_ssve(a, b, gamma, beta, M, N, ld, ld, eps);
   use(b);                                   // reads whatever was in b

produced **no computation, no status and no diagnostic**. The caller could not
distinguish success from a no-op, and would go on to consume stale or
uninitialized memory.

This is worth stating plainly because the project's own rules already forbade
it. ``CLAUDE.md`` §4 says to *"fail fast and clearly… error loudly at the
boundary — not with silent wrong numbers"*, and decision B makes correctness the
precondition for every performance claim. The guard was added so that tests
could skip gracefully on CI, which is a real need, but it solved a test-harness
problem by degrading the library's contract, and a silent no-op is the worst of
the available options because it is indistinguishable from success.

The fix is a two-layer API:

.. list-table::
   :header-rows: 1

   * - layer
     - behaviour
   * - ``layer_norm()`` / ``rms_norm()`` (**new, public**)
     - Dispatches to the best implementation available: ``FEAT_SME2`` → V7,
       ``FEAT_SME`` → V6, otherwise the scalar reference. **Always computes a
       correct result**, on any CPU.
   * - ``layer_norm_ssve_v7()`` and the other 24 named kernels
     - ISA-specific, with a documented hard precondition. Calling one without
       the required feature prints a diagnostic naming the function and the
       missing feature, then aborts. It never returns an uncomputed buffer.

``norm_dispatch_target()`` reports which of the three the current host selects,
so a benchmark row can always say which code ran.

The tests for this are deliberately **not** SME-guarded. The whole point is that
the dispatchers work everywhere, so they must execute on the CI runner, which
is the only machine that exercises the scalar fallback at all. One of them is a
direct regression test for the original defect: fill the output with a sentinel,
call the dispatcher, and require that no element still holds the sentinel.

Gate before timing, everywhere
--------------------------------

The consolidated ablation already verified each row and printed ``ok = yes/no``.
That is weaker than it looks: a row could fail verification and still print a
GiB/s figure, and a reader scanning the throughput column would never notice.

The rule is now **gate first, and refuse to time on failure**:

1. run the exact function that is about to be timed, on the exact shape;
2. compare its complete output against the float64 reference;
3. if it disagrees, print the failure and **do not produce a timing at all**.

The run then states the total, which is the sentence this earns:

.. code-block:: text

   Correctness gate: 66 / 66 configurations verified against the float64
   reference BEFORE timing.
   No configuration produced a timing without first matching the reference.

The external baselines got the same treatment. Previously the Python drivers
dumped one shape (128×64) for cross-verification while the results table
reported three, and verifying one while publishing three is an inference rather
than a check, since a kernel can be right at one shape and wrong at a tail or
boundary. Both drivers now dump **every** benchmarked shape and write a manifest
listing them; the C++ checker walks the manifest:

.. code-block:: text

   # framework executorch 1.4.1 (torch 2.13.0)
   # python 3.12.14 | threads 1
   128x64        LayerNorm 3.099e-06 OK   RMSNorm 3.576e-07 OK
   1024x2048     LayerNorm 7.153e-07 OK   RMSNorm 1.073e-06 OK
   4096x8192     LayerNorm 7.153e-07 OK   RMSNorm 2.265e-06 OK
   6 / 6 checks passed.

Best-case is not the only number
----------------------------------

Every table in this harness reported best-of-N, i.e. the minimum sample. That is
a defensible instrument for estimating a *ceiling*, since taking the minimum
suppresses preemption, migration and unrelated system activity, so it answers
"what can this kernel do when nothing interferes".

It is a poor number to present **alone**, because it cannot distinguish

* "18.4 µs is what this kernel does", from
* "one lucky run hit 18.4 µs and the other 49 were 24–31 µs".

Those have very different engineering meanings and identical minima. The
minimum is therefore kept and explicitly labelled best-case, with a median and
a p10–p90 spread from the same samples printed beside it:

.. code-block:: text

   stage              lever                        GiB/s   median    p10-p90
   V6 (incumbent)     4-row-block contiguity       38.46    38.21   38.0-38.4
   V7 (SME2)          4-vector LD1W/ST1W           38.46    38.38   38.0-38.5
   ZA residency       ZA staging, 1R+1W            24.91    24.75   24.5-24.9

The result is reassuring rather than dramatic, and that is itself the finding:
min, median and the p10–p90 band agree to within about 1 %, so the best-case
figures this project has reported all along **were** typical. That was
previously an assumption; it is now shown. A wide min-to-median gap would have
been a signal that a headline number was fragile.

Provenance
------------

A performance table is only reproducible if the reader knows which build, which
machine and which CPU features produced it. ``main_norm`` now opens with:

.. code-block:: text

   RUN PROVENANCE — the conditions every number below was produced under
     git commit        : 252d4cf-dirty
     build type        : Release
     compiler          : AppleClang 16.0.0.16000026
     OS                : macOS 15.2 (arm64)
     CPU               : Apple M4
     FEAT_SME (sysctl) : 1
     FEAT_SME2 (sysctl): 1
     cpu_supports_sme(): true,  cpu_supports_sme2(): true
     streaming VL      : 16 FP32 lanes (RDSVL)
     norm dispatch     : SME2 (V7 multi-vector)
     OpenMP threads    : 10
     scheduler QoS     : QOS_CLASS_USER_INTERACTIVE (P-core hint)

Two details are deliberate. The commit is marked ``-dirty`` when the working
tree has uncommitted changes, because a run from a modified tree is not
reproducible from its SHA and saying so is the point. And the SME lines are the
**detected** values from ``sysctl``, not a datasheet claim: this project
carried a stale "the M4 is SME1" assumption in its own documents for several
sprints while the hardware was reporting ``FEAT_SME2 = 1``.

Sprint 8 status
~~~~~~~~~~~~~~~~

* **API:** public ``layer_norm()``/``rms_norm()`` dispatchers with a scalar
  fallback; 25 ISA-specific entry points converted from silent no-op to
  documented precondition plus loud abort; ``norm_dispatch_target()`` added.
* **Tests:** 4 new host-portable dispatch cases (they run on CI, where the
  scalar fallback is the path taken), including a sentinel-based regression
  test for the original defect. Suite green on the M4: **137 cases, 780 264
  assertions**.
* **Correctness gating:** 66/66 internal configurations and 6/6 per-baseline
  external checks, all gated *before* timing.
* **Statistics:** median and p10–p90 alongside the best-case minimum.
* **Provenance:** build, machine and detected-feature block printed with every
  run; environment manifests written by both baseline drivers.

What this sprint deliberately did not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **No kernel changed**, so no performance number moved. The tables here should
  reproduce the Sprint 6/7 values, and they do.
* **CMake presets and the terminology/claim corrections** are the next steps,
  not this one, since this sprint was scoped to the correctness and provenance
  defects.
