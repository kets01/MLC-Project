MLC-Norm Sprint 7.5 — External Baselines
=========================================

How do we compare to state of the art?
----------------------------------------

Every performance claim in this report so far has been *internal*: kernel
against kernel, against our own measured roofline. That answers "did the
optimizations work" but not "is the result any good". This step places both
norms against two widely-used implementations on the same machine.

.. warning::

   **This project already got a vendor comparison wrong once**
   (``sprint6_errors.md`` §2.7–2.8), and the harness rules below each exist
   because of one of those failures:

   * an "Accelerate RMSNorm" that was really our own ``vDSP_svesq`` +
     ``vDSP_vsmul`` composition — Accelerate has no RMSNorm;
   * ``vDSP_normalize`` called "LayerNorm" when it has no eps, no γ and no β;
   * a headline "16–37× faster" that **measured a layout mismatch, not a
     kernel** — vDSP forced to walk our column-major matrix at ``stride = ld``,
     one cache miss per element. In its own layout that library was **faster
     than us**: 65.77 vs 38.47 GiB/s at 16 MiB.

Method
~~~~~~~~

* **Native layout for everyone.** Our kernels are column-major over a strided
  feature axis; the libraries normalize the last, contiguous axis of a
  row-major tensor. Both are efficient *in their own* layout — V6's grouping
  makes each of our column touches 256 contiguous bytes — so each is measured
  in the layout it was designed for. The transpose lives only in the
  correctness checker, never in a timed path.
* **Correctness gates the comparison** (decision B, applied to baselines).
  Every library output is dumped to raw FP32 and verified against our float64
  reference *before* its throughput is quoted. This is what catches the failure
  mode above — two sides computing different functions.
* **One thread**, same shapes, same byte convention (useful = 1R + 1W).
* **Current upstream versions**, on a standalone Python 3.12 installed
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

.. list-table:: max\|diff\| vs our float64 reference, M=128 N=64
   :header-rows: 1

   * - implementation
     - LayerNorm
     - RMSNorm
     - verdict
   * - PyTorch 2.13.0
     - 1.19e-06
     - 2.38e-07
     - same function
   * - ExecuTorch 1.4.1
     - 3.10e-06
     - 3.58e-07
     - same function

Both agree with our reference to FP32 rounding, so the throughput numbers below
compare implementations of the *same* function.

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
     - 11.18
     - 3.87
     - 1.41
     - 5.01
     - 1.5×
   * - 1024×2048 (16 MiB)
     - **16.49**
     - 7.89
     - 4.63
     - 6.17
     - 6.06
     - 2.1×
   * - 4096×8192 (256 MiB)
     - **13.54**
     - 7.35
     - 4.44
     - 5.92
     - 5.93
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
     - **38.56**
     - 7.52
     - 4.52
     - 2.51
     - 3.23
     - 5.1×
   * - 1024×2048 (16 MiB)
     - **38.46**
     - 3.44
     - 6.65
     - 4.27
     - 6.64
     - 5.8×
   * - 1024×2048 → 4096×8192 (256 MiB)
     - **24.65**
     - 2.77
     - 5.90
     - 3.26
     - 4.19
     - 4.2×

We are ahead of both baselines at every shape: **LayerNorm by 1.5–2.1×** and
**RMSNorm by 4.2–5.8×**. That asymmetry is the interesting part, and it is not
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
at 2.77 GiB/s while ``torch.compile`` — which fuses the decomposition —
roughly doubles it to 5.90.

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
measurement, which is the practice ``sprint6_errors.md`` credits for every claim
that survived contact.

.. list-table::
   :header-rows: 1

   * - prediction
     - outcome
   * - **P1** — PyTorch eager slow, we win 5–10×, from dispatch overhead and
       lack of fusion
     - **Partly wrong.** RMSNorm 4.2–5.8× (just under the range), LayerNorm
       only 1.5–2.1× (well under). The stated *reason* is also wrong for
       LayerNorm, which is fully fused and well optimized. The attribution
       splits by norm, not by framework.
   * - **P2** — ExecuTorch portable slower than PyTorch eager
     - **Confirmed but narrowly**, and much closer than on the old version:
       LayerNorm 5.92 vs 7.35, RMSNorm 3.26 vs 2.77 (portable actually *wins*
       here, because eager RMSNorm decomposes).
   * - **P3** — RMSNorm decomposes rather than running a fused kernel
     - **Confirmed, and in PyTorch as well as ExecuTorch** — which was not
       predicted. The profiler trace above is the evidence, and the
       eager-vs-compile gap (2.77 → 5.90) is its cost.
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
     - **5.90**
   * - ExecuTorch portable RMSNorm
     - 0.62
     - **3.26**
   * - ExecuTorch portable LayerNorm
     - 2.77
     - **5.92**

Our RMSNorm margin would have read 8–13× against the old stack and is 4.2–5.8×
against the current one. **The kernels did not change; the baseline did.** Any
"N× faster than library X" claim is a statement about a specific version on a
specific day, and is quoted here with versions attached for that reason.

The honest limit of this result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Beating PyTorch and ExecuTorch is not the same as being state of the art**,
and the report does not claim it is:

1. **The fastest implementation ever measured on this machine is neither of
   these.** Sprint 6's corrected figure has Apple's vDSP, in its own contiguous
   layout, at **65.77 GiB/s** against our 38.46 at 16 MiB — about **1.7× faster
   than us**. Against the footprint-matched ceiling (115.6 GiB/s at 16 MiB)
   that is ~57 % for vDSP versus our ~33 %.
2. **These are general-purpose frameworks, not specialist kernels.** A framework
   whose CPU RMSNorm decomposes in eager mode is a baseline for "what you get
   without a kernel", not a state-of-the-art bar for RMSNorm.
3. **The comparison includes runtime dispatch.** ExecuTorch times are
   per-invocation through its runtime, which is honest for an on-device
   comparison but is not a pure kernel number.

The defensible claim is therefore narrow and worth stating precisely: *against
two general-purpose frameworks at current versions, in their own layouts,
single-threaded, and verified to compute the same function, our kernels are
1.5–2.1× (LayerNorm) and 4.2–5.8× (RMSNorm) faster; the RMSNorm margin is
largely the absence of a fused CPU kernel on their side, and a specialist vendor
library still beats us.*

Reproducing
~~~~~~~~~~~~~

.. code-block:: bash

   # A standalone Python 3.12 (ExecuTorch's XNNPACK partitioner needs >= 3.10).
   # Do NOT install into mlc_env, which holds the Sphinx toolchain the docs
   # workflow depends on.
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12 && uv venv --python 3.12 /tmp/et312
   uv pip install --python /tmp/et312/bin/python executorch

   /tmp/et312/bin/python bench/baselines/torch_norm_bench.py       --outdir /tmp/t312
   /tmp/et312/bin/python bench/baselines/executorch_norm_bench.py  --outdir /tmp/et312d

   # Correctness gate — run before trusting any number above
   c++ -std=c++17 -O2 -I include bench/baselines/verify_baseline.cpp \
       build/src/norm/libnorm_lib.a build/src/week3/libweek3_lib.a \
       build/src/week6/libweek6_lib.a -o /tmp/verify_baseline
   /tmp/verify_baseline /tmp/t312 && /tmp/verify_baseline /tmp/et312d

What is left open
~~~~~~~~~~~~~~~~~~~

* **A vDSP re-measurement through this harness.** The 65.77 GiB/s figure comes
  from the Sprint-6 error log rather than from this correctness-gated harness.
  Since it is the only implementation known to beat us, it is by far the most
  valuable baseline still missing.
* **Threaded comparison.** Everything here is single-threaded; the frameworks
  parallelize by default and our TEIR path scales to 2.1×, so a multi-thread
  comparison is a separate question with a separate ceiling.
