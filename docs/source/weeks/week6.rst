Week 6: Code Generation
=======================================

This week introduced code generation, moving from fixed-size opcodes
to dynamic generators. The goal was to implement the Unary and Gemm interfaces
to produce optimized machine code for varying dimensions (M, N, K).

Generator Implementation

Instead of loading static arrays, we implemented logic to emit instructions
based on runtime parameters.

  - Unary Generator (SSVE): Dynamically calculates loop bounds (M \times N / 16)
    and emits SSVE instructions. Supported primitives include identity, zero,
    and relu.
  - GEMM Generator (SME): Generates an SME-based kernel using fmopa. The
    generator handles an arbitrary K dimension by constructing a dedicated inner
    loop.

Performance Results (Unary)

We benchmarked 9 settings for the Unary Identity primitive. Performance is
reported in GiB/s (Read + Write).

+--------------+--------------+
|Dimensions     | Performance | 
+==============+==============+
| 64 x 64 | 57.22 GiB/s       |
+--------------+--------------+
| 128 x 128 | 106.71 GiB/s    |
+--------------+--------------+
| 512 x 128 | 433.16 GiB/s    |
+--------------+--------------+
| 512 x 512 | 272.28 GiB/s    |
+--------------+--------------+

The performance increased significantly compared to Week 5 (22 GiB/s).
We believe this is because larger dimensions amortize the fixed cost of the smstart and
smstop instructions. The peak of 433 GiB/s demonstrates the generator's ability
to saturate the CPU's cache bandwidth.

The generative JIT approach provides the flexibility to handle different matrix
shapes while maintaining peak hardware performance. 

Performance Results (GEMM)
We benchmarked the GEMM generator across 27 different dimension combinations. The results show significant scaling as matrix volume increases.
+--------------------+----------------------+
| Dimensions (M,N,K) | Performance (GFLOPS) |
+====================+======================+
| 64 x 64 x 64 | 119.79 GFLOPS |
+--------------------+----------------------+
| 64 x 512 x 512 | 805.47 GFLOPS |
+--------------------+----------------------+
| 128 x 128 x 512 | 1007.59 GFLOPS |
+--------------------+----------------------+
| 128 x 512 x 512 | 1026.12 GFLOPS |
+--------------------+----------------------+
| 512 x 512 x 512 | 1008.39 GFLOPS |
+--------------------+----------------------+

At K=64, performance is roughly 120 GFLOPS. Increasing K to 512 boosts performance to over 1 TFLOPS (1026.12 GFLOPS).
Reaching a peak of 1.026 TFLOPS on mobile/desktop hardware confirms that our JIT-generated machine code is achieving high utilization of the Apple Silicon SME matrix units but still less than the static kernel creation.