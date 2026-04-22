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
    * Implemented the C++ reference for the permutation kernel.
    * Developed the unified benchmark application (``bench_week2``).
    * Conducted performance analysis and GiB/s scaling measurements.
* **Mariza Yamdjeu**:
    * Implemented the optimized AArch64 Neon permutation kernel.
    * Developed the ``TRN1/TRN2`` in-register transpose logic.
    * Handled the modularization of the CMake system into weekly subdirectories.

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
    