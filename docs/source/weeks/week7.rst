Week 7: Tiled Execution Intermediate Representation (TEIR)
=========================================================

This week involved implementing a **TEIR Runtime** to orchestrate large-scale tensor operations. The system serves as a bridge between high-level tensor descriptions (TEIR trees) and the optimized SSVE/SME micro-kernels developed in previous weeks.

Objectives
----------

* Develop a recursive execution engine to traverse TEIR loop nests.
* Implement a ``RuntimeContext`` to manage memory pointers across high-dimensional tensors.
* Invoke specialized JIT kernels for Transposition (Identity) and Matmul (GEMM).
* Analyze performance on Apple M4 hardware for massive 8192^3 workloads.

Implementation and Constraints
------------------------------

**Recursive Traversal**
The runtime uses a recursive function to "unroll" the TEIR tree into nested C++ loops. For each ``Iteration`` node, the engine calculates memory offsets and descends to the next dimension. At the ``Invocation`` leaf, it triggers the Week 6 machine code.

**Parallelization (OpenMP)**
While the implementation supports OpenMP via ``#pragma omp parallel for``, the library was not detected in the current environment (Apple Clang constraint). Consequently, all benchmarks were executed in **single-threaded mode**. 

Performance Results
-------------------

The following results were captured on the Apple M4 system:

+--------------------+-----------------------+--------------------------+
| Task               | Dimensions            | Performance              |
+====================+=======================+==========================+
| Transposition      | 96 x 128 x 48 x 32    | 90.97 GiB/s              |
+--------------------+-----------------------+--------------------------+
| Matrix Mult        | 8192^3                | 125.22 GFLOPS            |
+--------------------+-----------------------+--------------------------+
| Tensor Contraction | 128 x 96 x ... x 256  | 9,385,627.21 GFLOPS *    |
+--------------------+-----------------------+--------------------------+

*\* Note: The anomalous result for Tensor Contraction is attributed to a stride-calculation bug where the loops may have skipped data processing, resulting in an artificially low execution time.*

Challenges and Observations
---------------------------

1. **Environment Setup:** Building OpenMP on macOS requires manual linking against Homebrew's ``libomp``. Without this link, the performance for the 8192^3 Matmul is limited by a single CPU core (125 GFLOPS).
2. **Verification Failures:** Functional verification for Transposition failed (0.0f vs 5.0f). This indicates that the pointer advancement logic (strides) across 4 dimensions did not correctly reach the final memory addresses, even though the core execution flow was successful.