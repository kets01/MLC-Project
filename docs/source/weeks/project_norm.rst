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
       explained *negative* result (39–68 % slower than the SSVE winner)
     - Done
   * - 4
     - ``mini_jit::Norm`` runtime code generator, verified byte-identical to
       the hand-written SSVE winners via an encoding diff
     - Done
   * - 5
     - TEIR integration — registering the norms as primitives the runtime
       can place in a loop nest
     - In progress

The best kernel per architecture (SSVE **V6**, 4-row-block contiguity
grouping, for both norms) is frozen as the incumbent that Sprint 4's JIT
emits and Sprint 5's TEIR runtime calls.

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
   sprint2_debug_log
