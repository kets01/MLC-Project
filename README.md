# Machine Learning Compiler Lab

This repository contains the work completed throughout the **Machine Learning Compiler Lab**, including the weekly exercises, supporting compiler/JIT infrastructure, tests, documentation, and the final optimization project.

The final project studies high-performance **LayerNorm** and **RMSNorm** on the Apple M4 using Arm **SME**, **Streaming SVE**, and **SME2**, together with runtime code generation and TEIR-based scheduling.

## Repository overview

```text
MLC-Project/
├── src/
│   ├── week1/ ... week7/   Semester lab implementations
│   └── norm/               Final LayerNorm/RMSNorm project
├── include/                Shared headers and interfaces
├── apps/                   Application and benchmark entry points
├── bench/                  PyTorch/ExecuTorch baselines and verification
├── tests/                  Catch2 tests
├── data/                   Input and TEIR data
├── docs/                   Documentation and checked-in results
├── paper/                  Final report
├── presentation/           Final presentation
└── .github/                CI workflows
```

## Final project: LayerNorm and RMSNorm

The final project develops LayerNorm and RMSNorm from scalar references to optimized hand-written and JIT-generated kernels.

The evaluated optimization path includes:

- baseline SSVE kernels,
- reciprocal-square-root refinement,
- independent reduction accumulators,
- four-block row grouping,
- SME2 multi-vector accesses,
- ZA-based activation reuse,
- runtime-generated V6/V7 kernels,
- and TEIR/OpenMP scheduling.

On the 256 MiB benchmark shape, four-block grouping raises RMSNorm from **10.04 to 21.05 GiB/s**, and SME2 multi-vector accesses further increase it to **24.63 GiB/s**. The JIT-generated kernels match the performance of their hand-written counterparts.

## Build and test

CMake 3.21+ is recommended for the provided presets.

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release
```

Run the normalization benchmark with:

```bash
./build/src/norm/main_norm
```

Available presets:

| Preset | Purpose |
|---|---|
| `release` | Optimized build for normal development and benchmarks |
| `debug` | Debug build with AddressSanitizer and UndefinedBehaviorSanitizer |
| `release-submission` | Configuration used for the reported project measurements |

For performance measurements, use an optimized preset.

## Correctness and reproducibility

Kernel implementations are checked against the scalar reference before timing. JIT-generated kernels are additionally compared against the corresponding hand-written instruction stream.

PyTorch and ExecuTorch baselines are under:

```text
bench/baselines/
```

Checked-in benchmark results are stored in:

```text
docs/source/_static/results/
```

Benchmark output includes run provenance such as the git revision, compiler, build type, operating system, detected SME/SME2 support, and streaming vector length.

## Hardware note

The SME/SME2 kernels are evaluated on an **Apple M4**. The project can still be built on AArch64 systems without SME execution support; hardware-specific tests are skipped when the required feature is unavailable.

## Team

- Ketsia Kemkuini
- Mariza Sitcheu
