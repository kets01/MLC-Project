MLC-Norm Project
=================

**MLC-Norm** is the capstone project for the Machine Learning Compilers Lab:
optimizing the two dominant Transformer normalization primitives —
**LayerNorm** and **RMSNorm** — for AArch64 with the Scalable Matrix
Extension (SME) and its Streaming SVE (SSVE) mode, on top of the weeks 1–7
lab foundation documented elsewhere in this report.

Both norms are **memory-bandwidth bound**: they touch every element of the
activation tensor while doing very little arithmetic per element. The
project's throughline is therefore a data-movement story, not a
flop-counting one — hand-written SSVE kernels, an SME/ZA-tile variant, a
runtime JIT generator, and TEIR loop-nest integration, each verified against
a C++ reference before any GiB/s number is trusted.

Background
----------

The full target stack, built up one layer at a time:

- **C++ reference** → hand-written SSVE kernel → JIT-generated via
  ``mini_jit::Norm`` → composed into a TEIR loop nest via the week7 runtime.

The two norms differ in reduction structure, which is itself a deliberate
ablation axis (not a flag):

- **LayerNorm** — two-pass (mean, then variance + normalize-scale-shift);
  higher arithmetic intensity, numerically stable.
- **RMSNorm** — single-pass (sum of squares, no mean, no β); ~10–40 % faster
  at equal accuracy.

Both are bandwidth-bound; the evaluation metric throughout is **effective
GiB/s** against a measured hardware roofline, never GFLOPS.

Sprint status
-------------

.. list-table::
   :header-rows: 1
   :widths: 12 68 20

   * - Sprint
     - What it delivered
     - Status
   * - 0
     - Module scaffold wired into the existing build/test/CI/Sphinx pipeline
     - Done
   * - 1
     - C++ reference, correctness harness, bandwidth baseline
     - Done
   * - 2
     - Hand-written SSVE kernels for both norms; V0–V6 ablation against a
       validated single-core roofline
     - Done
   * - 3
     - Hand-written SME/ZA-tile kernels for both norms — a measured,
       explained *negative* result (39–68 % slower than the SSVE winner;
       rebuilt on SME2 multi-vector ``MOVA`` in Sprint 6 for +62 %/+114 %,
       and still losing)
     - Done
   * - 4
     - ``mini_jit::Norm`` runtime code generator, verified byte-identical to
       the hand-written SSVE winners via an encoding diff
     - Done
   * - 5
     - TEIR integration — the norms registered as primitives the runtime
       places in a loop nest, driven from real ``.teir`` files, with
       OpenMP row-parallel scaling measured against the chip ceiling
     - Done
   * - 6
     - Consolidated ablation; the ``d8-d15`` ABI fix completed across every
       kernel; SME2 multi-vector V7 (+17 % RMSNorm in DRAM) promoted into the
       JIT and TEIR; **four earlier report claims found wrong and corrected**
     - Done
   * - 7
     - The two ``context.md`` §8 tradeoffs measured: naive single-pass variance
       driven to negative-and-NaN, two-pass shown flat across 12 orders of κ,
       and the streaming transition timed directly at 9 ns — with the
       small-tensor penalty attributed to group granularity, not ``SMSTART``
     - Done
   * - 7.5
     - External baselines (PyTorch 2.13, ExecuTorch 1.4.1 incl. XNNPACK):
       **1.5–2.1× faster on LayerNorm, 4.2–5.8× on RMSNorm**, both verified to
       compute the same function — but the RMSNorm margin is mostly PyTorch
       decomposing rather than fusing, and a vendor DSP library still beats us,
       so this is not "state of the art"
     - Done

The best kernel per architecture (SSVE **V6**, 4-row-block contiguity
grouping, for both norms; **V7** where ``FEAT_SME2`` is present) is the
incumbent that Sprint 4's JIT emits and Sprint 5's TEIR runtime calls.

.. important::

   **A correction that touches every performance number in this report.**
   Until Sprint 6, every "% of roofline" divided by a single figure —
   59.5 GiB/s, the single-core streaming ceiling measured at a 256 MiB working
   set.  That is the **DRAM** ceiling, and it is the correct denominator only
   for DRAM-resident shapes.

   The ceiling is a curve: the same probe reaches **115.6 GiB/s at a 16 MiB
   footprint**.  Dividing a cache-resident kernel by the DRAM ceiling credits
   it for a constraint it never met, and inflated those percentages by ~2×.
   The most-quoted number in this report — RMSNorm V6 at "~95 % of the
   moved-bytes roofline" — is really **50 %**.

   No kernel got slower; the measurements are unchanged, and so is every
   kernel-to-kernel ratio the ablation is built on.  What changed is the
   denominator, and with it the claim that the SSVE kernels were near
   saturation.  ``main_norm`` now measures the ceiling across footprints and
   each row divides by the one matching its own working set.  Sections
   restated on that basis carry a note saying so.

Reading this section
---------------------

The detailed sprint write-ups are split into their own pages so each stays a
readable size; the split is by *sprint*, with Sprint 2 further split by
*norm* since RMSNorm and LayerNorm each get a full hand-written-kernel +
ablation treatment there.

.. toctree::
   :maxdepth: 2

   norm_sprint0_1
   norm_sprint2_rmsnorm
   norm_sprint2_layernorm
   norm_sprint3
   norm_sprint4
   norm_sprint5
   norm_sprint6
   norm_sprint7
   norm_sprint7_5
   sprint2_debug_log
