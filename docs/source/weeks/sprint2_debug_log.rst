Sprint 2b — Debug Log
=====================

This log documents the five bugs encountered during the Sprint 2b ablation
study (V0–V3 RMSNorm SSVE variants) and the exact fix applied to each.  The
goal is an honest engineering record: fast wrong kernels are worthless (decision
B), and surfacing the real problems is a project deliverable (``context.md §8``).

---

Error 1 — Tool permission failure mid-file write
-------------------------------------------------

**Symptom**

The Write tool stream closed partway through writing ``rms_norm_ssve_v1.S``,
leaving the file incomplete.

**Root cause**

A tool-permission interruption during the streaming write.

**Fix**

Resumed the session and rewrote all three ``.S`` files from the point of
interruption.  No code change; purely a tooling/session issue.

---

Error 2 — Assembler error: wrong register type in ``LD1W``/``ST1W`` offset
---------------------------------------------------------------------------

**Symptom**

.. code-block:: text

    error: register must be x0..x30 with required shift 'lsl #2'

on lines in ``rms_norm_ssve_v3.S`` using ``[X8, X24]`` and ``[X9, X25]``.

**Root cause**

``LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2]`` (scalar-register offset) interprets
``Xm`` as an **element count** that is word-scaled (``Xm × 4`` bytes).  ``X24``
and ``X25`` were computed in the prologue as **byte** strides
(``LSL X24, X5, #2``), not element counts — the wrong type for this addressing
form.

**Fix**

Use ``X5`` (``ld_a`` in elements, the original argument, still live in the
register) and ``X6`` (``ld_b`` in elements) directly with the ``LSL #2`` shift:

.. code-block:: asm

    // Before (wrong — X24 is bytes):
    ld1w {z6.s}, p1/z, [x8, x24]

    // After (correct — X5 is elements; LSL #2 converts to bytes):
    ld1w {z6.s}, p1/z, [x8, x5, lsl #2]

The prologue writes the byte forms into ``X24``/``X25`` but leaves the element
forms untouched in ``X5``/``X6``, so they remain valid for this addressing mode.

---

Error 3 — V2 and V3 produce NaN (``SMSTART SM`` zeroes all Z registers)
------------------------------------------------------------------------

**Symptom**

All V2 and V3 Catch2 tests failed:

.. code-block:: text

    nanf == Approx(expected_value)

V0 and V1 were unaffected.

**Root cause**

``SMSTART SM`` on the Apple M4 **zeroes every Z register (Z0–Z31)** on entry to
streaming mode.  In V2 and V3, the ``1/N`` pre-computation was performed before
``SMSTART``:

.. code-block:: asm

    fmov  s0, #1.0
    scvtf s1, x23      // float(N)
    fdiv  s0, s0, s1   // 1.0/N  ← stored in S0 = Z0.S[0]

    smstart sm         // ← ZEROES Z0; S0 now reads 0.0

    dup   z5.s, z0.s[0]  // broadcasts 0.0, not 1/N

With ``Z5 = 0``, the mean-square computation became ``sumsq × 0 = 0``;
``FRSQRTE(0) = +∞``; NaN propagated to the output.

**Fix**

Move the scalar ``FDIV`` and ``DUP`` to **after** ``SMSTART SM``.  Scalar NEON
instructions (``FMOV``, ``SCVTF``, ``FDIV``) are legal in streaming SVE mode on
SME1:

.. code-block:: asm

    smstart sm
    ptrue  p0.s, all

    // eps reload (see Error 4)
    ldr    s0, [sp, #56]
    dup    z8.s, z0.s[0]

    // 1/N — now inside the streaming region:
    fmov  s0, #1.0
    scvtf s1, x23
    fdiv  s0, s0, s1
    dup   z5.s, z0.s[0]    // z5 = {1/N, 1/N, ...}  correct

---

Error 4 — V3 N=1 test produces NaN (``SMSTART SM`` also zeroes ``Z8`` / eps)
-----------------------------------------------------------------------------

**Symptom**

After fixing Error 3, one V3 test still failed:

.. code-block:: text

    nanf == Approx(0.0)   // V3, N=1 test, row 6

The reference output for row 6 was ``0.0`` (that row's input was exactly zero).

**Root cause**

Same underlying cause as Error 3, but for ``eps``.  The V1/V2/V3 prologue saves
``eps`` into S8 (a scalar alias for the low 32 bits of Z8) with:

.. code-block:: asm

    fmov  s8, s0     // save eps into callee-saved S8
    str   d8, [sp, #56]   // and spill D8 to the stack

``SMSTART SM`` zeroes Z8, so after ``SMSTART`` the register S8 reads 0.0.
The broadcast ``DUP Z8.S, Z8.S[0]`` then fills Z8 with 0.0 instead of ``eps``.

With ``eps = 0`` and an all-zero input row, the denominator of inv_rms is
``0 + 0 = 0``; ``FRSQRTE(0) = +∞``; multiplying by zero gives NaN (``∞ × 0``).

The N=1 test was the first case where an active row had an exactly-zero input.
V1 had the same latent bug but the existing V1 test shapes never triggered a
zero-sum row.

**Fix**

After every ``SMSTART SM`` in V1, V2, and V3, reload ``eps`` from the D8 stack
slot (written by the prologue's ``STR D8, [SP, #56]``) and re-broadcast:

.. code-block:: asm

    smstart sm
    ptrue  p0.s, all

    ldr    s0, [sp, #56]    // s0 = eps (low 32 bits of the saved D8)
    dup    z8.s, z0.s[0]    // z8 = {eps, eps, ...}

The stack is not affected by ``SMSTART``, so the spilled value is safe.
V0 was left unchanged (it uses ``Z8`` differently and already passes all tests).

---

Error 5 — V0/V1/V2 show ``inf GiB/s`` in the ablation benchmark
----------------------------------------------------------------

**Symptom**

The ablation section of ``apps/main_norm.cpp`` printed:

.. code-block:: text

    V0 (FSQRT+FDIV)   M=128  N=2048  inf GiB/s  ...
    V1 (FRSQRTE+NR)   M=128  N=2048  inf GiB/s  ...
    V2 (V1 + inv_N)   M=128  N=2048  inf GiB/s  ...
    V3 (V2 + unroll)  M=128  N=2048  24.96 GiB/s ...

V3 was always correct; V0/V1/V2 always showed ``inf``.

**Root cause**

The kernel's prologue saves and restores only **D8** (the single callee-saved FP
register it uses).  ``SMSTART SM`` zeroes **all Z registers (Z0–Z31)**, which
includes **D0–D15**.  D9–D15 are callee-saved by the AArch64 AAPCS, so a called
function must not clobber them — but the kernel's epilogue only restores D8,
leaving D9–D15 undefined after every call.

At ``-O2``, the compiler kept the timing results (``sec_v0``, ``sec_v1``,
``sec_v2``) in callee-saved registers **D9**, **D10**, **D11** across successive
``bench_ssve*`` calls.  Each call's ``SMSTART`` zeroed those registers without
the kernel restoring them.  The elapsed-seconds values became 0.0 → GiB/s was
computed as ``bytes / 0.0 = +∞``.

Only ``sec_v3``, the last call whose result arrived in D0 (a caller-saved
return-value register, reloaded immediately), was unaffected.

**Fix**

Declare all per-shape timing results and the peak bandwidth as ``volatile
double``.  ``volatile`` forces the compiler to spill each value to **the stack**
immediately rather than keeping it in a register across subsequent calls:

.. code-block:: cpp

    volatile double vsec_v0 = bench_ssve   (...);
    volatile double vsec_v1 = bench_ssve_v1(...);
    volatile double vsec_v2 = bench_ssve_v2(...);
    volatile double vsec_v3 = bench_ssve_v3(...);
    volatile double vpeak   = peak;

Stack slots are not touched by ``SMSTART``, so the values survive across every
kernel call.

**Why the kernel does not save D9–D15:** saving six additional double-precision
registers in the prologue/epilogue costs 12 extra instructions per call.  Since
the kernel itself never uses D9–D15 (it works entirely with Z-register aliases
and X-registers), the correct fix is on the **caller** side: use ``volatile`` or
``noinline`` wrappers to prevent the compiler from keeping live values in those
registers across calls that enter streaming mode.

---

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 5 30 30 35

   * - #
     - Symptom
     - Root cause
     - Fix
   * - 1
     - Incomplete ``.S`` file after write
     - Tool permission interruption
     - Resumed and rewrote from interruption point
   * - 2
     - Assembler error: wrong register type in ``LD1W`` offset
     - ``X24``/``X25`` hold byte strides; ``LD1W [Xn, Xm, LSL #2]`` needs element count
     - Use original ``X5``/``X6`` (element counts) with ``LSL #2``
   * - 3
     - V2/V3 output NaN (all test cases)
     - ``SMSTART SM`` zeroes Z0; pre-``SMSTART`` ``1/N`` broadcast gives 0
     - Move scalar FDIV + DUP to after ``SMSTART SM``
   * - 4
     - V3 N=1 output NaN for zero-input row
     - ``SMSTART SM`` zeroes Z8; eps reloaded as 0.0
     - Reload eps from D8 stack slot and re-broadcast immediately after ``SMSTART SM``
   * - 5
     - V0/V1/V2 show ``inf GiB/s`` in benchmark
     - Compiler kept timing results in D9–D11; ``SMSTART`` zeroed them; kernel only restores D8
     - Declare all timing variables ``volatile double`` to force stack spill
