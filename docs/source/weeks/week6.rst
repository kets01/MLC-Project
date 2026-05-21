Here is a concise version of the report, matching the length and style of your
Week 5 document.

Week 6: Parametric JIT Code Generation

This week introduced parametric code generation, moving from fixed-size opcodes
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
This is because larger dimensions amortize the fixed cost of the smstart and
smstop instructions. The peak of 433 GiB/s demonstrates the generator's ability
to saturate the CPU's cache bandwidth.

The generative JIT approach provides the flexibility to handle different matrix
shapes while maintaining peak hardware performance. 
