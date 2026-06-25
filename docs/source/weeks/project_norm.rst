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

Next
~~~~

Sprint 1: C++ reference implementations, the numerical verification harness,
and the first GiB/s measurements (the oracle and the ruler).

----

Sprint 1 — C++ Reference and Bandwidth Baseline
------------------------------------------------

**Goal:** canonical C++ reference implementations, a thorough numerical
verification harness, and a first GiB/s measurement against the measured
STREAM peak — establishing correctness gates and a roofline baseline before
any assembly is written.

Canonical kernel interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

All kernels — the C++ reference, the hand-written SSVE kernel, and future
JIT/TEIR layers — share a single signature (Decision A in ``context.md``).
Both norms follow the **column-major** convention with an explicit leading
dimension, matching the layout used throughout weeks 1–7:

.. code-block:: cpp

    // element [row, col] lives at:  ptr[row + col * ld]
    void layer_norm_ref(const float* a, float* b,
                        const float* gamma, const float* beta,
                        int64_t m, int64_t n,
                        int64_t ld_a, int64_t ld_b, float epsilon);

    void rms_norm_ref  (const float* a, float* b,
                        const float* gamma,
                        int64_t m, int64_t n,
                        int64_t ld_a, int64_t ld_b, float epsilon);

Reference implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

Both kernels are in ``src/norm/reference.cpp``. Internal accumulation uses
``double`` to avoid single-precision cancellation in the test oracle.

**LayerNorm** — two-pass per row:

1. Compute mean: ``mean = Σ a[row, col] / N``
2. Compute variance: ``var = Σ (a[row, col] − mean)² / N``
3. Normalise and apply affine: ``b = gamma * (a − mean) / sqrt(var + eps) + beta``

**RMSNorm** — single-pass per row (no mean, no beta):

1. Compute RMS: ``rms = sqrt(Σ a[row, col]² / N + eps)``
2. Scale: ``b = gamma * a / rms``

Verification harness
~~~~~~~~~~~~~~~~~~~~

``tests/test_norm.cpp`` contains 8 Catch2 test cases and 2 268 assertions.
The test cases cover:

- Identity input (all elements equal): output must equal ``beta``
- Normal random inputs (M×N square, default leading dimension)
- Non-square matrices with separate ``ld_a`` / ``ld_b`` (padded layout)
- Large-magnitude stress inputs (DC offset ``SHIFT = 1e4``) to expose
  single-pass cancellation bugs in any future kernel

Each case computes the same quantity with an independent scalar ``for``-loop
reference, then compares element-by-element within tolerance ``kTol = 1e-5``
(relative). All 2 268 assertions pass.

Bandwidth baseline
~~~~~~~~~~~~~~~~~~

``apps/main_norm.cpp`` measures effective memory bandwidth for both kernels
over six shapes (M ∈ {128, 1024}, N ∈ {64, 512, 2048}).

The STREAM peak is probed with a scale-add kernel guarded by
``__attribute__((noinline))`` to prevent dead-store elimination:

.. code-block:: cpp

    __attribute__((noinline))
    static void bw_scale_add(float* __restrict__ d,
                              const float* __restrict__ s, size_t n) {
        for (size_t i = 0; i < n; ++i) d[i] = s[i] + 1.0f;
    }

Results on M4 (Release build, Apple Silicon, measured peak **76.8 GiB/s**):

+------------------+-------+--------+----------+----------+
| Kernel           |   M   |    N   | GiB/s    | % peak   |
+==================+=======+========+==========+==========+
| layer_norm_ref   |  128  |    64  |  3.16    |   4.1 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     |  128  |    64  |  5.39    |   7.0 %  |
+------------------+-------+--------+----------+----------+
| layer_norm_ref   |  128  |   512  |  1.79    |   2.3 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     |  128  |   512  |  1.93    |   2.5 %  |
+------------------+-------+--------+----------+----------+
| layer_norm_ref   |  128  |  2048  |  1.68    |   2.2 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     |  128  |  2048  |  2.45    |   3.2 %  |
+------------------+-------+--------+----------+----------+
| layer_norm_ref   | 1024  |    64  |  0.66    |   0.9 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     | 1024  |    64  |  0.76    |   1.0 %  |
+------------------+-------+--------+----------+----------+
| layer_norm_ref   | 1024  |   512  |  0.65    |   0.8 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     | 1024  |   512  |  0.73    |   0.9 %  |
+------------------+-------+--------+----------+----------+
| layer_norm_ref   | 1024  |  2048  |  0.57    |   0.7 %  |
+------------------+-------+--------+----------+----------+
| rms_norm_ref     | 1024  |  2048  |  0.63    |   0.8 %  |
+------------------+-------+--------+----------+----------+

The reference kernels reach only 1–7 % of peak. The three-pass structure
(LayerNorm) and the scalar ``double`` loop prevent memory-level parallelism.
This establishes the lower bound the SSVE kernel must beat.

Sprint 1 status
~~~~~~~~~~~~~~~

- **Build:** clean under ``-Wall -Wextra -Werror -pedantic``
- **Test:** 8 test cases, 2 268 assertions — all pass in CI and locally
- **CI:** ``test_norm`` runs on M1/M2 CI runner (no SME required)
- **Bandwidth:** measured on M4; peak probe and all 12 shapes complete

----

Sprint 2 — Hand-Written SSVE LayerNorm Kernel
----------------------------------------------

**Goal:** implement LayerNorm as a hand-written AArch64 SSVE kernel in
``src/norm/layer_norm_ssve.S``, verify it numerically against the Sprint 1
reference across 6 test cases, and commit all files one by one on branch
``feat/sprint2-layernorm-ssve``.

This sprint is done by **Mariza**; RMSNorm SSVE is handled separately by
**Ketsia**.

Key design decisions
~~~~~~~~~~~~~~~~~~~~

**SME1 mandatory streaming SVE subset.**
Not all SVE instructions are available in streaming mode (``PSTATE.SM = 1``).
The most important restriction: *scatter-gather loads*
(``LD1W {z.s}, p, [Xn, Zm.S, UXTW #2]``) and integer vector multiply
(``MUL Zdn.S, Pg/M, ...``) are absent from the SME1 mandatory subset and
raise SIGILL at runtime on the M4.

**Vectorise across rows, loop over columns.**
In column-major layout a *column* is contiguous in memory.  Rather than
loading a row (which needs a gather), the kernel loads one column slice of
SVL rows at a time using a plain contiguous ``LD1W``:

.. code-block:: text

    outer loop: row chunks of SVL rows
      pass 1: for each column → LD1W SVL rows → accumulate per-row sum z8
              → FDIV z8/n → per-row mean z8
      pass 2: for each column → LD1W, FSUB mean, FMUL square, accumulate z9
              → FDIV z9/n, FADD eps, FSQRT, FDIV 1/sqrt → per-row inv_std z5
      pass 3: for each column → LD1W, FSUB, FMUL inv_std
              → LD1RW gamma[col], FMUL; LD1RW beta[col], FADD; ST1W

**Vector length agnostic (VLA).**
``CNTW x8`` queries SVL at runtime.  ``WHILELO p1.s, x_rbase, x_m``
generates the predicate for the tail row chunk (active lane j iff
r\_base + j < m), so the kernel is correct for any SVL — not just the
16-lane 512-bit SVL of the M4.

**Streaming mode boundary.**
Epsilon arrives in ``s0`` (the first FP argument register) before
``SMSTART SM`` could alter FP state; it is spilled to the stack frame and
reloaded after entering streaming mode.

``-march=armv9-a+sme`` scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SME march flag is applied **only to the assembly source**, not to the
C++ sources, using ``set_source_files_properties``:

.. code-block:: cmake

    set_source_files_properties(layer_norm_ssve.S
        PROPERTIES COMPILE_OPTIONS "-march=armv9-a+sme")

Applying the flag to ``reference.cpp`` caused the compiler to auto-vectorize
with regular (non-streaming) SVE, which SIGILLs on Apple Silicon because
non-streaming SVE is absent on M-series chips.

C++ namespace bridge
~~~~~~~~~~~~~~~~~~~~

The assembly exports a plain C symbol ``layer_norm_ssve``.
``src/norm/layer_norm_ssve.cpp`` declares it with ``extern "C"`` and wraps
it in the ``mini_jit::norm`` namespace so all callers use the canonical
signature from ``norm.hpp``.

Verification test cases
~~~~~~~~~~~~~~~~~~~~~~~

Six new Catch2 test cases in ``tests/test_norm.cpp``, all guarded with
``if (!cpu_supports_sme()) SKIP("SME required")``:

+------+-------------------------------------------+------------------------------+
| Case | Shape / condition                         | What it exercises            |
+======+===========================================+==============================+
|  1   | M=8,  N=4  (N < SVL)                     | tail-only path               |
+------+-------------------------------------------+------------------------------+
|  2   | M=8,  N=16 (N = SVL)                     | single full vector, no tail  |
+------+-------------------------------------------+------------------------------+
|  3   | M=6,  N=19 (N = SVL + 3)                 | full vector + small tail     |
+------+-------------------------------------------+------------------------------+
|  4   | M=8,  N=37 (N = 2·SVL + 5)               | multiple vectors + tail,     |
|      |                                           | outer row loop               |
+------+-------------------------------------------+------------------------------+
|  5   | M=5,  N=19, ld_a=ld_b=8 (ld > M)         | padded matrix, stride test   |
+------+-------------------------------------------+------------------------------+
|  6   | M=4,  N=37, SHIFT=1e4                    | large-magnitude stability    |
+------+-------------------------------------------+------------------------------+

Cases 1–5 use relative tolerance ``kTol = 1e-5``.
Case 6 uses an absolute margin of ``1e-3``: with a DC offset of 1×10⁴ and
differences of ±5, catastrophic cancellation limits FP32 accuracy to roughly
``ε_float × SHIFT × inv_std ≈ 5×10⁻⁴``; a relative tolerance would be
unfairly strict for elements whose normalised value is near zero.

Test results on M4 (all 14 test cases, 3 081 assertions):

.. code-block:: text

    All tests passed (3081 assertions in 14 test cases)

CI (M1/M2) skips the 6 SSVE cases with ``SKIP("SME required")`` and runs
the 8 Sprint 1 cases as before.

Sprint 2 status
~~~~~~~~~~~~~~~

- **Build:** clean; assembly assembles under ``-march=armv9-a+sme``
- **Test:** 14 test cases, 3 081 assertions — all pass on M4; CI skips gracefully
- **Kernel:** ``layer_norm_ssve.S`` — VLA, SME1 mandatory subset only,
  three passes (mean / variance / normalise), column-major contiguous loads
- **Next:** add ``layer_norm_ssve`` to the ``main_norm`` GiB/s table (Sprint 3)
