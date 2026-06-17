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
