Team Contributions
==================

We share responsibilities to ensure both members understand the full stack of the project.

Week 1
------
   * **Mariza Yamdjeu**: 
    * Implemented ``inner_product_asm`` .
    * Set up the CMake build system and Catch2 integration.
    * Performed debugging verification via LLDB/GDB.

* **Ketsia Kemkuini**: 
    * Implemented ``outer_product_asm`` with pointer-increment optimization.
    * Configured the GitHub Actions CI pipeline.
    * Sphinx documentation .


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
