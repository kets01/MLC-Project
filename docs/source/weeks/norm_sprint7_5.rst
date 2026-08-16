MLC-Norm Sprint 7.5 — External Baselines
=========================================

External framework comparison
-------------------------------

Every performance claim in this report so far has been *internal*: kernel
against kernel, against our own measured roofline. That answers "did the
optimizations work" but not "is the result any good". This step places both
norms against two widely-used implementations on the same machine.

.. warning::

   **This project already got a vendor comparison wrong once**
   (``docs/dev-notes/sprint-errors/sprint6_errors.md`` §2.7–2.8), and the harness rules below each exist
   because of one of those failures:

   * an "Accelerate RMSNorm" that was really our own ``vDSP_svesq`` +
     ``vDSP_vsmul`` composition — Accelerate has no RMSNorm;
   * ``vDSP_normalize`` called "LayerNorm" when it has no eps, no γ and no β;
   * a headline "16–37× faster" that **measured a layout mismatch, not a
     kernel** — vDSP forced to walk our column-major matrix at ``stride = ld``,
     one cache miss per element. Re-run in its own contiguous layout the same
     composition was recorded as *faster* than us (65.77 vs 38.47 GiB/s at
     16 MiB), which is what turned the headline from a win into an error.

   Neither of those vDSP figures is used anywhere in this report: both come
   from the error log rather than from the correctness-gated harness below, and
   neither was ever verified to compute the same function as our reference.
   They are cited here only as the reason the rules exist.

Method
~~~~~~~~

* **Native layout for everyone.** Our kernels are column-major over a strided
  feature axis; the libraries normalize the last, contiguous axis of a
  row-major tensor. Both are efficient *in their own* layout — V6's grouping
  makes each of our column touches 256 contiguous bytes — so each is measured
  in the layout it was designed for. The transpose lives only in the
  correctness checker, never in a timed path.

  .. important::

     **Which question this answers, and which it does not.** Measuring each
     implementation in its preferred representation answers:

       *What throughput can each implementation achieve on this operator, given
       the tensor layout it was designed for?*

     It does **not** answer:

       *What would it cost to replace a framework's operator with our kernel,
       for the same incoming tensor representation?*

     Those are different questions and the second is the one a compiler
     integrator actually faces — it would have to include any layout conversion
     at the boundary, which this comparison deliberately excludes from both
     sides. We answer only the first, and every margin below should be read
     with that scope attached. (The Sprint-6 failure was the mirror image:
     forcing one implementation into the other's layout and reporting the
     mismatch as a kernel result.)
* **Correctness gates the comparison** (decision B, applied to baselines).
  Every library output is dumped to raw FP32 and verified against our float64
  reference *before* its throughput is quoted. This is what catches the failure
  mode above — two sides computing different functions.
* **One thread**, same shapes, same byte convention (useful = 1R + 1W).
* **Versions used for the reported run**, on a standalone Python 3.12 installed
  specifically for this: **PyTorch 2.13.0** and **ExecuTorch 1.4.1**, with the
  **XNNPACK delegate** — the path that is actually deployed on device —
  exercised rather than skipped. Nothing is patched; both libraries are
  stock.

.. note::

   An earlier attempt ran on the project's Python 3.9 environment, where the
   newest installable ExecuTorch is 0.3.0 and the XNNPACK partitioner cannot
   even be imported (``TypeError: 'staticmethod' object is not callable`` —
   staticmethod objects only became callable in Python 3.10). Reporting only
   the portable reference kernels would have been the mirror image of the
   Sprint-6 error: handicapping the baseline and calling the gap a win. A
   newer interpreter was installed instead, so the strongest available path is
   the one measured.

Correctness first
~~~~~~~~~~~~~~~~~~~

.. list-table:: max\|diff\| vs our float64 reference, worst over the three shapes
   :header-rows: 1

   * - implementation
     - LayerNorm
     - RMSNorm
     - verdict
   * - ``torch_eager``
     - 1.19e-06
     - 2.38e-07
     - same function
   * - ``torch_compile``
     - 1.31e-06
     - 9.54e-07
     - same function
   * - ``et_portable``
     - 3.10e-06
     - 2.27e-06
     - same function
   * - ``et_xnnpack``
     - 3.10e-06
     - 2.27e-06
     - same function

All four agree with our reference to FP32 rounding, so the throughput numbers
below compare implementations of the *same* function. Every row of both results
tables has its own entry here: **12 checks per framework, 24 in total, all
passing**. The harness writes one manifest row per (implementation, norm,
shape) and the C++ checker walks that manifest, so the set of configurations
*timed* and the set *verified* cannot drift apart. Earlier versions dumped one
output per norm — the eager PyTorch result and whichever ExecuTorch mode was
available — which left ``torch.compile`` and the second ExecuTorch mode timed
and published but never checked.

Results
~~~~~~~~~

Single-threaded, useful bytes (1R+1W), each in its native layout. Best baseline
per row is the one our margin is quoted against.

.. list-table:: LayerNorm — GiB/s, higher is better
   :header-rows: 1

   * - shape
     - **ours (V7)**
     - torch eager
     - torch.compile
     - ET portable
     - ET XNNPACK
     - our margin
   * - 128×64
     - **16.84**
     - 13.08
     - 4.09
     - 4.05
     - 5.07
     - 1.3×
   * - 1024×2048 (16 MiB)
     - **16.47**
     - 7.79
     - 4.55
     - 6.16
     - 6.25
     - 2.1×
   * - 4096×8192 (256 MiB)
     - **13.52**
     - 7.52
     - 4.43
     - 5.87
     - 5.98
     - 1.8×

.. list-table:: RMSNorm — GiB/s, higher is better
   :header-rows: 1

   * - shape
     - **ours (V7)**
     - torch eager
     - torch.compile
     - ET portable
     - ET XNNPACK
     - our margin
   * - 128×64
     - **39.58**
     - 9.16
     - 4.76
     - 2.54
     - 3.06
     - 4.3×
   * - 1024×2048 (16 MiB)
     - **38.45**
     - 3.19
     - 7.19
     - 4.23
     - 4.40
     - 5.3×
   * - 4096×8192 (256 MiB)
     - **24.63**
     - 2.82
     - 6.12
     - 3.27
     - 4.22
     - 4.0×

We are ahead of both baselines at every shape: **LayerNorm by 1.3–2.1×** and
**RMSNorm by 4.0–5.3×**. That asymmetry is the interesting part, and it is not
about our kernels.

Fusion, not kernel quality, explains the asymmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profiling what each PyTorch module actually dispatches to on CPU:

.. list-table::
   :header-rows: 1

   * - module
     - ATen ops, by self time
   * - ``torch.nn.LayerNorm``
     - ``aten::native_layer_norm`` — a single fused kernel
   * - ``torch.nn.RMSNorm``
     - ``aten::mul``, ``aten::pow``, ``aten::sum``, ``aten::_fused_rms_norm``,
       ``aten::div_``

``_fused_rms_norm`` exists at the dispatcher level, but the CPU self time is
dominated by the elementwise ops around it: **on CPU, eager RMSNorm decomposes**,
each pass reading and writing the whole tensor. That is why eager RMSNorm sits
at 2.82 GiB/s while ``torch.compile`` — which fuses the decomposition —
roughly doubles it to 6.12.

The consequence is a clean, attributable inversion:

.. list-table:: Which norm is faster, within each implementation
   :header-rows: 1

   * - implementation
     - faster norm
     - ratio
   * - **ours**
     - RMSNorm
     - 1.8–2.3× faster than our LayerNorm
   * - PyTorch (eager)
     - LayerNorm
     - 2.7× faster than its RMSNorm

The mathematics says RMSNorm should be the *cheaper* operation — one pass, no
mean, 2R+1W against LayerNorm's 3R+1W, which is exactly decision C's premise and
what our own ablation measures. PyTorch eager reverses that ordering, and the
reason is an implementation artifact: the cheaper operation is the one that
happens not to have a fused CPU kernel. **A fused implementation of the
expensive norm beats a decomposed implementation of the cheap one** — which is
the clearest argument in this report for why writing the kernel was worth doing.

Against the pre-registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expectations were written into the ROADMAP and committed *before* any
measurement, which is the practice ``docs/dev-notes/sprint-errors/sprint6_errors.md`` credits for every claim
that survived contact.

.. list-table::
   :header-rows: 1

   * - prediction
     - outcome
   * - **P1** — PyTorch eager slow, we win 5–10×, from dispatch overhead and
       lack of fusion
     - **Partly wrong.** RMSNorm 4.0–5.3× (just under the range), LayerNorm
       only 1.3–2.1× (well under). The stated *reason* is also wrong for
       LayerNorm, which is fully fused and well optimized. The attribution
       splits by norm, not by framework.
   * - **P2** — ExecuTorch portable slower than PyTorch eager
     - **Confirmed but narrowly**, and much closer than on the old version:
       LayerNorm 5.87 vs 7.52, RMSNorm 3.27 vs 2.82 (portable actually *wins*
       here, because eager RMSNorm decomposes).
   * - **P3** — RMSNorm decomposes rather than running a fused kernel
     - **Confirmed, and in PyTorch as well as ExecuTorch** — which was not
       predicted. The profiler trace above is the evidence, and the
       eager-vs-compile gap (2.82 → 6.12) is its cost.
   * - **P4** — we may lose
     - **Not realized against these two.** See the limits below — it is not
       settled by this experiment.

Baselines move between versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Worth recording because it bears on how much any single comparison is worth. The
same harness on the older stack (PyTorch 2.4.0 / ExecuTorch 0.3.0) gave
materially different baselines:

.. list-table:: 4096×8192, GiB/s
   :header-rows: 1

   * - path
     - old stack
     - current stack
   * - torch.compile RMSNorm
     - 3.02
     - **6.12**
   * - ExecuTorch portable RMSNorm
     - 0.62
     - **3.27**
   * - ExecuTorch portable LayerNorm
     - 2.77
     - **5.87**

Our RMSNorm margin would have read 8–13× against the old stack and is 4.0–5.3×
against the current one. **The kernels did not change; the baseline did.** Any
"N× faster than library X" claim is a statement about a specific version on a
specific day, and is quoted here with versions attached for that reason.

The honest limit of this result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Beating PyTorch and ExecuTorch is not the same as being state of the art**,
and the report does not claim it is:

1. **No specialist vendor kernel was measured.** This is the largest gap, and
   it is a gap rather than a result. Apple's BNNS was investigated and produced
   no number: on macOS 15.2 the reachable LayerNorm entry point is a
   *deprecated* generic filter that we could create but not execute
   (``BNNSFilterApply`` → −1 in every configuration tried, including a minimal
   single-sample case), and **the Accelerate/BNNS API available on the macOS 15.2
   reference system does not expose a direct RMSNorm operation** — every
   "RMS" symbol in those headers is ``RMSProp``. Apple's current BNNSGraph
   does provide ``layerNorm(axes:epsilon:)`` and ``rmsNorm(scale:epsilon:)``,
   but both are absent from this SDK and post-date this OS. Full evidence in
   ``docs/dev-notes/tooling/bnns_investigation.md``.
2. **These are general-purpose frameworks, not specialist kernels.** A framework
   whose CPU RMSNorm decomposes in eager mode is a baseline for "what you get
   without a kernel", not a state-of-the-art bar for RMSNorm.
3. **The comparison includes runtime dispatch.** ExecuTorch times are
   per-invocation through its runtime, which is honest for an on-device
   comparison but is not a pure kernel number.
4. **Single-threaded only.** The frameworks parallelize by default and our TEIR
   path scales 2.1×; a threaded comparison is a separate question against a
   separate ceiling.

.. note::

   **A claim withdrawn.** An earlier version of this section stated that a
   specialist vendor library "still beats us", citing Apple's vDSP at
   65.77 GiB/s against our 38.46 at 16 MiB. That figure came from the Sprint-6
   error log, not from this correctness-gated harness — it was never verified
   to compute the same function, and the vDSP composition it referred to was
   the very thing §2.7 identified as mislabelled. It is therefore withdrawn
   rather than restated: we do not have a verified vendor number in either
   direction. The §2.7–2.8 entries stay in ``docs/dev-notes/sprint-errors/sprint6_errors.md``, because that
   is the record of our own mistake.

The defensible claim is therefore narrow and worth stating precisely: *against
two general-purpose frameworks at current versions, in their own layouts,
single-threaded, and verified to compute the same function on every benchmarked
shape, our kernels are 1.3–2.1× (LayerNorm) and 4.0–5.3× (RMSNorm) faster; the
RMSNorm margin is largely the absence of a fused CPU kernel on their side; and
no specialist vendor kernel was measured, so this is not a state-of-the-art
claim.*

Reproducing
~~~~~~~~~~~~~

.. code-block:: bash

   # A standalone Python 3.12 (ExecuTorch's XNNPACK partitioner needs >= 3.10).
   # Do NOT install into mlc_env, which holds the Sphinx toolchain the docs
   # workflow depends on.
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12 && uv venv --python 3.12 /tmp/et312

   # PINNED, not floated: the margins above are a statement about specific
   # versions.  The same harness on torch 2.4.0 / executorch 0.3.0 gave
   # torch.compile RMSNorm 3.02 GiB/s against 6.12 here, which would have made
   # our RMSNorm margin read 8-13x instead of 4.0-5.3x.  The kernels did not
   # change; the baseline did.
   uv pip install --python /tmp/et312/bin/python \
       -r bench/baselines/requirements-baselines.txt

   /tmp/et312/bin/python bench/baselines/torch_norm_bench.py       --outdir /tmp/t312
   /tmp/et312/bin/python bench/baselines/executorch_norm_bench.py  --outdir /tmp/et312d

   # Correctness gate — run before trusting any number above
   c++ -std=c++17 -O2 -I include bench/baselines/verify_baseline.cpp \
       build/src/norm/libnorm_lib.a build/src/week3/libweek3_lib.a \
       build/src/week6/libweek6_lib.a -o /tmp/verify_baseline
   /tmp/verify_baseline /tmp/t312 && /tmp/verify_baseline /tmp/et312d

The drivers write a ``manifest.txt`` recording the framework and Python
versions, the thread count and every shape measured; the checker reads that
same list, so the set of shapes *timed* and the set *correctness-gated* cannot
drift apart. The manifests from the runs reported above are checked in under
``bench/baselines/manifests/``. A rerun that produces different versions is a
different measurement, and its numbers should not be compared with these
without saying so.

What is left open
~~~~~~~~~~~~~~~~~~~

* **A specialist vendor kernel.** This is the most valuable missing baseline
  and it remains missing by decision, not oversight. Apple's BNNS was
  investigated and could not be driven on macOS 15.2, and it has no RMSNorm
  there at all (``docs/dev-notes/tooling/bnns_investigation.md``); the remaining route would be
  BNNSGraph via CoreML, which was scoped out. Until such a number exists,
  **this report makes no claim about how our kernels compare to a specialist
  vendor implementation, in either direction.**
* **Threaded comparison.** Everything here is single-threaded; the frameworks
  parallelize by default and our TEIR path scales to 2.1×, so a multi-thread
  comparison is a separate question with a separate ceiling.
* **The integration question.** As stated in the method, this measures each
  implementation in its preferred layout. What it would cost to substitute our
  kernel for a framework operator on the framework's own tensor representation
  — including any boundary conversion — is not measured.
