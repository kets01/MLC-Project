MLC-Norm Project
================

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

- **LayerNorm** — two-pass (mean, then variance + normalize-scale-shift); more
  arithmetic intensity, numerically stable
- **RMSNorm** — single-pass (sum of squares, no mean, no β); ~10–40% faster

Both are bandwidth-bound; the evaluation metric is **effective GiB/s** vs the
measured peak.

Sprint 1 — C++ reference, correctness harness, and bandwidth baseline
----------------------------------------------------------------------

**Goal:** an obviously-correct scalar reference for both norms, a Catch2
verification harness (including numerical-stability stress cases), and a
reproducible GiB/s measurement that sets the roofline target for all future
kernels.

Canonical kernel signature (decision A)
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

Reference implementations (decision B/C)
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
scale by γ.  No mean subtraction and no β — the simpler reduction is what makes
RMSNorm ~10–40% faster at equal accuracy relative to LayerNorm.

Correctness harness (decision B)
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

Bandwidth harness and roofline (decision E)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``apps/main_norm.cpp`` measures:

1. **Peak bandwidth** — a STREAM-style ``dst[i] = src[i] + 1.0f`` loop over
   128 MiB per array (beyond M-series L3), best-of-10 runs, to establish the
   roofline target.
2. **Norm effective bandwidth** — ``bytes = 2 × M × N × 4`` (one FP32 read +
   one FP32 write; γ/β assumed L1-resident), best-of-50 runs, for six shapes:

**Roofline target (STREAM-style probe, single-threaded scalar):** 10.62 GiB/s.
This is the single-threaded scalar ceiling; the SSVE and JIT kernels will
target the vectorised peak, which is significantly higher.

.. list-table:: Sprint 1 — reference GiB/s (scalar C++, Apple M4)
   :header-rows: 1
   :widths: 10 10 20 20 20

   * - M (rows)
     - N (features)
     - ``layer_norm_ref`` GiB/s
     - ``rms_norm_ref`` GiB/s
     - Notes
   * - 128
     - 64
     - 1.91 (18% of peak)
     - 3.10 (29% of peak)
     - Small working set, fits in L1/L2
   * - 128
     - 512
     - 1.66 (16% of peak)
     - 2.47 (23% of peak)
     -
   * - 128
     - 2048
     - 1.60 (15% of peak)
     - 2.38 (22% of peak)
     - Row resident across passes
   * - 1024
     - 64
     - 0.80 (8% of peak)
     - 1.29 (12% of peak)
     - More rows → more scalar loop overhead
   * - 1024
     - 512
     - 0.78 (7% of peak)
     - 1.14 (11% of peak)
     -
   * - 1024
     - 2048
     - 0.68 (6% of peak)
     - 0.89 (8% of peak)
     - Large tensor, pressure on caches

Two patterns visible in the data:

- **RMSNorm is consistently 1.3–1.6× faster than LayerNorm** at the same
  shape — the single-pass structure eliminates the second loop over the feature
  axis and the mean subtraction, halving the arithmetic work.
- **Performance degrades as M grows** — the scalar loop overhead (branch,
  pointer arithmetic, FP divide) accumulates per row.  The SSVE kernel will
  amortise this by processing multiple elements per cycle.

The numbers sit at 6–29 % of the single-threaded STREAM ceiling.  The gap is
the target for Sprint 2's SSVE kernel.

Sprint 1 status
~~~~~~~~~~~~~~~

- **Reference:** both ``layer_norm_ref`` and ``rms_norm_ref`` implemented and
  merged (PR #24).
- **Signature:** canonical interface pinned in ``include/norm/norm.hpp``.
- **Tests:** 8 Catch2 cases (4 per norm), all green on CI, including
  large-magnitude stability-stress inputs.
- **Benchmark:** GiB/s harness + STREAM peak-bandwidth probe in place;
  numbers to be recorded from M4.

Next
~~~~

Sprint 2: hand-written SSVE kernel for both norms — VLA, predicated tails,
verified against this reference, and measured vs the roofline.
