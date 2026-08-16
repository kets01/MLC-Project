MLC-Norm Sprint 0 & 1 — Scaffold and Reference Harness
======================================================

Sprint 0 — Scaffold
-------------------

**Goal:** prove the norm module is wired into the existing build/test/CI/Sphinx
pipeline with placeholder logic — no real kernel yet.

Module layout
~~~~~~~~~~~~~

.. code-block:: text

    include/norm/norm.hpp       — canonical kernel signatures (pinned in Sprint 1)
    src/norm/reference.cpp      — scalar C++ reference for LayerNorm and RMSNorm
    src/norm/CMakeLists.txt     — build wiring
    tests/test_norm.cpp         — Catch2 verification harness (normal + stress cases)
    apps/main_norm.cpp          — GiB/s benchmark + STREAM-style peak-bandwidth probe

Sprint 0 status
~~~~~~~~~~~~~~~

- **Build:** ``cmake --build`` produces ``test_norm`` and ``main_norm``
- **Test:** placeholder identity copy verified; test runs in CI on M1/M2 (no SME required)
- **CI:** ``test_norm`` in the host-portable group alongside week1/week2
- **Docs:** this section

Background
~~~~~~~~~~

MLC-Norm adds LayerNorm and RMSNorm kernels for AArch64 SME/SSVE on top of
the existing lab foundation (weeks 1–7). The full stack target is:

- **C++ reference** → hand-written SSVE kernel → JIT-generated via ``mini_jit::Norm``
  → composed into a TEIR loop nest via the week7 runtime

The two norms differ in reduction structure:

- **LayerNorm** — **two reduction stages** (mean, then variance) plus output
  generation, hence **three input traversals**, 3R+1W; more arithmetic
  intensity, numerically stable
- **RMSNorm** — **one reduction stage** (sum of squares, no mean, no β) plus
  output generation, hence **two traversals**, 2R+1W

That 1.33× traffic ratio is the structural source of RMSNorm's advantage. Both
are bandwidth-dominated at the shapes measured here; the evaluation metric is
**effective GiB/s** against a measured, footprint-matched peak.

Sprint 1 — C++ reference, correctness harness, and bandwidth baseline
----------------------------------------------------------------------

**Goal:** an obviously-correct scalar reference for both norms, a Catch2
verification harness (including numerical-stability stress cases), and a
reproducible GiB/s measurement that sets the roofline target for all future
kernels.

Canonical kernel signature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both norms share a single interface defined in ``include/norm/norm.hpp`` under
``namespace mini_jit::norm``.  The same signatures are used by the C++
reference, the future ``mini_jit::Norm`` JIT generator, and the TEIR
registration — no layer invents its own.

**Layout convention:** column-major, explicit leading dimension
(``data[row + col * ld]``), matching the rest of the lab code.

.. code-block:: cpp

    // LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    void layer_norm_ref(const float* a, float* b,
                        const float* gamma, const float* beta,
                        int64_t m, int64_t n,
                        int64_t ld_a, int64_t ld_b,
                        float epsilon);

    // RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
    void rms_norm_ref(const float* a, float* b,
                      const float* gamma,
                      int64_t m, int64_t n,
                      int64_t ld_a, int64_t ld_b,
                      float epsilon);

Reference implementations 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``src/norm/reference.cpp`` implements both norms as deliberately simple scalar
C++ using ``double`` accumulation throughout — this makes it the trusted oracle
against which every future kernel is verified.

**LayerNorm — two-pass per row:**

1. Pass 1: accumulate mean (``sum / N``), then variance ``E[(x − mean)²]``.
   The two-pass structure avoids the catastrophic cancellation that a naive
   single-pass ``E[x²] − mean²`` formula produces on large-magnitude inputs.
2. Pass 2: normalize, scale by γ, shift by β.

**RMSNorm — single-pass per row:**

Sum of squares ``Σx²``, divide by N, add ε, take the inverse square root, then
scale by γ.  No mean subtraction and no β — one reduction stage instead of two,
so two traversals of the input instead of three.



Correctness harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_norm.cpp`` covers four cases for each norm:

- **Identity / unit-vector input** — analytically checkable output (e.g. all-equal
  input → LayerNorm output equals β; unit-vector input → RMSNorm output scales
  by ``sqrt(N)``).
- **Normal random-ish input** — verified against an independent scalar ``double``
  computation in the same test file.
- **Non-square matrix with ``ld_a ≠ ld_b``** — exercises leading-dimension
  handling.
- **Large-magnitude stress input** (``SHIFT = 1e4f`` plus small variation) —
  documents that the two-pass LayerNorm and the mean-free RMSNorm both remain
  accurate on inputs where a naive single-pass variance would lose bits.

All tests run on the CI runner (M1/M2, no SME required) with tolerance
``epsilon = 1e-5``.

Bandwidth harness and roofline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``apps/main_norm.cpp`` measures:

1. **Peak bandwidth** — a STREAM-style ``dst[i] = src[i] + 1.0f`` loop,
   best-of-10 runs, to establish the roofline target.  Originally run at a
   single size (128 MiB per array, beyond M-series L3); since Sprint 6 it is
   swept across footprints from 64 KiB to 256 MiB, because the ceiling turned
   out to depend strongly on working-set size (§1.2 of the Sprint-6 log).
2. **Norm effective bandwidth** — ``bytes = 2 × M × N × 4`` (one FP32 read +
   one FP32 write; γ/β assumed L1-resident), best-of-50 runs, for six shapes:

.. admonition:: Corrected in Sprint 6 — this table was measured twice wrong
   :class: warning

   The figures originally published here were affected by **two** defects that
   later sprints found, and both are corrected below.

   1. **The build was unoptimized.** ``CMAKE_BUILD_TYPE`` was empty, so the C++
      reference and the C++ probe compiled at ``-O0`` (Sprint 5 defect log).
      The "roofline target" of **10.62 GiB/s** was a debug-build scalar probe,
      not a hardware ceiling — the same probe in Release reads ~80 GiB/s in
      NEON.  Every original percentage divided one ``-O0`` number by another.
   2. **The ceiling is a curve, not a constant** (Sprint 6 §1.2).  Percentages
      are now taken against the streaming-mode ceiling measured *at each
      shape's own footprint*, which ranges from ~104 GiB/s at 64 KiB to
      59.5 GiB/s at 256 MiB.

   The re-measured numbers are Release-build, and the percentages fall sharply
   — not because the reference got slower, but because it is finally being
   compared against a real ceiling.  This is what the scalar oracle actually
   costs, and it is the honest starting point for Sprint 2's speed-ups.

**Roofline target:** the streaming-mode ceiling at the shape's own footprint.  For these six shapes that is 104.7–117.2 GiB/s — all of them
are cache-resident, so none of them faces the 59.5 GiB/s DRAM figure.

.. list-table:: Sprint 1 — reference GiB/s (scalar C++, Apple M4, Release, re-measured in Sprint 6)
   :header-rows: 1
   :widths: 8 8 10 17 17 20

   * - M (rows)
     - N (features)
     - ceiling
     - ``layer_norm_ref`` GiB/s
     - ``rms_norm_ref`` GiB/s
     - Notes
   * - 128
     - 64
     - 104.7
     - 4.09 (3.9 %)
     - 6.60 (6.3 %)
     - Small working set, fits in L1/L2
   * - 128
     - 512
     - 117.2
     - 2.42 (2.1 %)
     - 3.41 (2.9 %)
     -
   * - 128
     - 2048
     - 116.0
     - 2.24 (1.9 %)
     - 3.26 (2.8 %)
     - Row resident across passes
   * - 1024
     - 64
     - 117.2
     - 1.23 (1.1 %)
     - 1.37 (1.2 %)
     - More rows → more scalar loop overhead
   * - 1024
     - 512
     - 115.7
     - 1.33 (1.2 %)
     - 1.59 (1.4 %)
     -
   * - 1024
     - 2048
     - 115.6
     - 0.95 (0.8 %)
     - 1.03 (0.9 %)
     - Large tensor, pressure on caches

Two patterns visible in the data:

- **RMSNorm is faster than LayerNorm at every shape, but by a margin that
  shrinks with M** — 1.6× at M=128, N=64 down to 1.08× at M=1024, N=2048
  (the original text claimed a uniform 1.3–1.6×, which the Release numbers do
  not support).  The single-pass structure removes the second pass over the
  feature axis and the mean subtraction; as M grows, per-row scalar overhead
  comes to dominate both norms equally and the structural advantage is diluted.
- **Performance degrades as M grows** — the scalar loop overhead (branch,
  pointer arithmetic, FP divide) accumulates per row.  The SSVE kernel will
  amortise this by processing multiple elements per cycle.

The reference sits at **0.8–6.3 %** of the ceiling at its own footprint.  That
gap is the target for Sprint 2's SSVE kernel.

Sprint 1 status
~~~~~~~~~~~~~~~

- **Reference:** both ``layer_norm_ref`` and ``rms_norm_ref`` implemented and
  merged.
- **Signature:** canonical interface pinned in ``include/norm/norm.hpp``.
- **Tests:** 8 Catch2 cases (4 per norm), all green on CI, including
  large-magnitude stability-stress inputs.
- **Benchmark:** GiB/s harness + STREAM peak-bandwidth probe in place;
  numbers recorded from M4 (see table above).

