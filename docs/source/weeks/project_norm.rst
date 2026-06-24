MLC-Norm Project
================

Sprint 0 — Scaffold
-------------------

**Goal:** prove the norm module is wired into the existing build/test/CI/Sphinx
pipeline with placeholder logic — no real kernel yet.

Module layout
~~~~~~~~~~~~~

.. code-block:: text

    include/norm/norm.hpp       — placeholder signature (Sprint 0)
    src/norm/reference.cpp      — identity copy (replaced by real reference in Sprint 1)
    src/norm/CMakeLists.txt     — build wiring
    tests/test_norm.cpp         — host-portable Catch2 test
    apps/main_norm.cpp          — benchmark stub (GiB/s measurement path established)

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

Sprint 1 — C++ Reference, Verification Harness & Bandwidth Baseline
--------------------------------------------------------------------

**Goal:** an obviously-correct reference for both norms, the verification
harness, and the first GiB/s measurements — everything later is judged
against these.

Canonical kernel signature
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both norms share a single interface defined in ``include/norm/norm.hpp``
(decision A — one source of truth, reused by the JIT generator and TEIR
registration in later sprints):

.. code-block:: cpp

    // LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    void layer_norm_ref(const float* a, float* b,
                        const float* gamma, const float* beta,
                        int64_t m, int64_t n,
                        int64_t ld_a, int64_t ld_b, float epsilon);

    // RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
    void rms_norm_ref(const float* a, float* b,
                      const float* gamma,
                      int64_t m, int64_t n,
                      int64_t ld_a, int64_t ld_b, float epsilon);

Layout convention: **column-major**, explicit leading dimension
(``data[row + col * ld]``). Normalisation axis is **N** — each of the M
rows is normalised independently.

Reference implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both references accumulate in ``double`` to be numerically honest:

- **LayerNorm** (``layer_norm_ref``): two-pass per row. Pass 1 computes
  the mean and then the variance from ``(x − mean)²`` — the two-pass
  structure itself avoids catastrophic cancellation because variance never
  involves subtracting large numbers of similar magnitude. Pass 2
  normalises and applies scale/shift.

- **RMSNorm** (``rms_norm_ref``): single-pass per row. Accumulates
  ``sum(x²)``, divides by N, adds ε, takes the reciprocal square root,
  then scales. No mean subtraction, no β parameter.

Verification harness
~~~~~~~~~~~~~~~~~~~~

``tests/test_norm.cpp`` runs 8 Catch2 test cases covering:

- **Normal inputs**: small random-ish values, non-square shapes, separate
  ``ld_a``/``ld_b``.
- **Analytic ground-truth**: constant-row input → LayerNorm output must
  equal β exactly; unit-vector input → RMSNorm output is analytically
  known.
- **Stability-stress inputs**: values clustered around 1 × 10⁴ (large DC
  offset). These expose catastrophic cancellation in a naïve single-pass
  variance implementation. The two-pass LayerNorm reference passes
  comfortably; the RMSNorm reference is inherently stable (no subtraction).

All 2 268 assertions pass on both M1/M2 (CI) and M4 (local).

Bandwidth baseline (roofline anchor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured on **Apple M4** (16 GB unified memory) using ``apps/main_norm``.

**Peak bandwidth probe**: a ``noinline`` scale-add kernel
(``d[i] = s[i] + 1.0f``) over 128 MiB arrays. A plain ``memcpy`` is
short-circuited by macOS's VM copy-on-write mechanism; the ``+1.0f``
ensures ``d ≠ s`` after each pass, forcing real DRAM traffic.

.. code-block:: text

    Peak bandwidth (STREAM scale-add, M4): ~79.7 GiB/s
    Theoretical peak (Apple spec sheet):   ~111.8 GiB/s

The scalar references reach 1–8% of the measured peak — expected, since
they iterate with double-precision scalar arithmetic and make two passes
over the data for LayerNorm.

.. list-table:: Sprint 1 — reference GiB/s on M4 (best of 50 reps)
   :header-rows: 1
   :widths: 20 10 10 15 15

   * - Kernel
     - M (rows)
     - N (features)
     - GiB/s
     - % of peak
   * - layer_norm_ref
     - 128
     - 64
     - 4.09
     - 5.1%
   * - rms_norm_ref
     - 128
     - 64
     - 6.63
     - 8.3%
   * - layer_norm_ref
     - 128
     - 512
     - 2.42
     - 3.0%
   * - rms_norm_ref
     - 128
     - 512
     - 3.36
     - 4.2%
   * - layer_norm_ref
     - 128
     - 2048
     - 2.25
     - 2.8%
   * - rms_norm_ref
     - 128
     - 2048
     - 3.26
     - 4.1%
   * - layer_norm_ref
     - 1024
     - 64
     - 1.20
     - 1.5%
   * - rms_norm_ref
     - 1024
     - 64
     - 1.34
     - 1.7%
   * - layer_norm_ref
     - 1024
     - 512
     - 1.22
     - 1.5%
   * - rms_norm_ref
     - 1024
     - 512
     - 1.34
     - 1.7%
   * - layer_norm_ref
     - 1024
     - 2048
     - 0.95
     - 1.2%
   * - rms_norm_ref
     - 1024
     - 2048
     - 1.02
     - 1.3%

RMSNorm is consistently faster than LayerNorm at equal shapes because it
makes one pass instead of two (decision C).

Sprint 1 status
~~~~~~~~~~~~~~~

- **Reference:** both norms implemented and passing all correctness tests
- **Verification:** 8 test cases, 2 268 assertions, green on CI (M1/M2)
  and local (M4)
- **Bandwidth:** roofline anchor at ~79.7 GiB/s (measured); ~111.8 GiB/s
  (theoretical); reference kernels at 1–8% — the baseline Sprint 2 must beat

Next
~~~~

Sprint 2: hand-written SSVE kernel (VLA, predicated tail) for both norms —
the MVP that should close significantly on the roofline.
