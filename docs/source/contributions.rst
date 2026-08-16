Team Contributions
==================

We share responsibilities to ensure both members understand the full stack of the project.

Week 1
------
* **Mariza Yamdjeu**:
    * Implemented ``inner_product_asm``.
    * Set up the CMake build system and Catch2 integration.
    * Performed debugging verification via LLDB/GDB.

* **Ketsia Kemkuini**:
    * Implemented ``outer_product_asm`` with pointer-increment optimization.
    * Configured the GitHub Actions CI pipeline.
    * Sphinx documentation.


Week 2
------
* **Ketsia Kemkuini**:
    * **Permutation Optimization**: Developed the optimized AArch64 Neon permutation kernel using ``TRN1/TRN2`` in-register transpose logic.
    * **Throughput Benchmark**: Developed the timing logic in the benchmark application to calculate G-Instr/s and GFLOPS for FMADD.
    * **Infrastructure**: Restructured the project and CI/CD pipeline to support modular weekly subdirectories and native Apple Silicon runners.

* **Mariza Yamdjeu**:
    * **Throughput Kernel**: Implemented the ``fmadd_asm`` kernel in AArch64 using an unrolling factor of 8.
    * **Permutation Reference**: Implemented the C++ baseline for the `abc` to `cba` permutation for validation.
    * **Analysis**: Performed the scaling measurements for `|c|` and analyzed the "Memory Wall" effect on GiB/s.
    

Week 3
------

* **Mariza Yamdjeu**:
    * **Unary Kernel Suite**: Implemented the full set of Week 3 unary operations 
      (``identity``, ``zero``, ``relu``) in AArch64 SME/SVE, including both the 
      optimized assembly kernels and their C++ reference baselines.
    * **Validation & Benchmarking**: Developed the unified test harness and 
      high‑resolution microbenchmarks for 16×16 FP32 tiles, including correctness 
      verification, structured matrix printers, and GiB/s throughput analysis.
    * **Integration**: Extended and merged the Week 3 build system to support 
      combined unary + GEMM workflows, resolving cross‑module conflicts and 
      ensuring a clean, modular CMake structure.

* **Ketsia Kemkuini**:
    * **SME GEMM Kernel**: Designed and implemented the optimized 
      512×512×512 FP32 GEMM kernel using Arm SME streaming mode, including 
      correct handling of mixed row/column‑major layouts and SME tile configuration.
    * **Performance Engineering**: Built the GEMM benchmarking pipeline with 
      warm‑up phases, FLOP accounting, and GFLOPS reporting, enabling stable 
      performance measurements across 100+ iterations.
    * **Project Evolution**: Contributed the structural changes required to 
      integrate SME kernels into the Week 3 module, ensuring compatibility with 
      the existing unary‑ops framework and CI infrastructure.


Week 5
------
* **Ketsia Kemkuini**
    * Generated runtime opcode sequences for unary kernels
    * Adapted Week 3 SME kernels for JIT execution
    * Built correctness tests
    * Performed debugging and opcode validation experiments

* **Mariza Yamdjeu**
    * Designed and implemented the JIT runtime infrastructure
    * Integrated executable memory allocation using ``MAP_JIT``
    * Developed runtime benchmark pipeline

Week 6
------
* **Ketsia Kemkuini**
    * Implemented the JIT code generation logic for the gemm kernel.
    * Tested the generated code for correctness and performance for gemm.
    * Conducted benchmarking for the 27 settings and reported performance metrics.

* **Mariza Yamdjeu**
    * Implemented the JIT code generation logic for the unary kernel.
    * Tested the generated code for correctness and performance for each unary primitive (identity, zero, relu).
    * Conducted benchmarking for the 9 Unary kernel (primitive identity) settings and reported performance metrics in GiB/s.

Week 7
------

* **Ketsia Kemkuini**
    * Designed the ``TeirRuntime`` recursive execution engine.
    * Implemented the ``RuntimeContext`` for tracking data pointers across nested loops.
    * Developed the pointer arithmetic logic for multi-dimensional data traversal.
    * Managed the benchmarking process for Transposition and reported GiB/s performance.

* **Mariza Yamdjeu**
    * Defined the C++ structures for the TEIR Blueprint (Axis, Iteration, Invocation).
    * Implemented the builders for the Matmul and Contraction TEIR trees.
    * Integrated the Week 6 SME GEMM micro-kernels into the TEIR leaf nodes.
    * Performed the GFLOPS performance analysis for the heavy math tasks.
 

Final Project: LayerNorm and RMSNorm
------------------------------------

The final project was developed jointly and the workload was split 
between us. Responsibilities were divided across the two
normalization paths while common infrastructure, benchmarking, validation,
analysis, and final deliverables were shared.

* **Ketsia Kemkuini**:
    * Developed and optimized the **RMSNorm** path, including the scalar
      reference, SSVE variants, four-block grouping, SME2 V7, the V7x2 control,
      and ZA-based implementations.
    * Contributed to the **JIT code-generation path**, including optimized
      V6/V7 emission and validation against the corresponding hand-written
      kernels.
    * Contributed to the **performance evaluation**, including the kernel
      ablation, bandwidth measurements, streaming-transition measurements,
      small-shape crossover experiments, and interpretation of the
      working-set-dependent behavior.
    * Contributed to the **external baseline and reproducibility workflow**,
      including PyTorch/ExecuTorch benchmarking, output verification, result
      provenance, and submission-oriented build infrastructure.
    * Co-developed the **TEIR/JIT integration**, debugging and correctness
      checks, and co-authored the final report and presentation.

* **Mariza Yamdjeu**:
    * Developed and optimized the **LayerNorm** path, including the scalar
      reference, SSVE variants, independent accumulators, four-block grouping,
      SME2 V7, Welford, and ZA-based implementations.
    * Contributed to the **JIT code-generation path**, including integration of
      generated normalization kernels with the common runtime interface and
      validation of generated instruction streams.
    * Contributed to the **numerical and performance evaluation**, including
      the LayerNorm stability study, variance-estimator comparison, correctness
      analysis, and interpretation of the optimization results.
    * Contributed to the **TEIR and parallel-execution path**, including
      normalization integration, runtime shape/stride propagation, outer
      scheduling, and OpenMP scaling experiments.
    * Co-developed the **TEIR/JIT integration**, debugging and correctness
      checks, and co-authored the final report and presentation.

