# Machine Learning Compiler Lab - Project

This repository contains the implementation for the Machine Learning Compiler Lab. 

## Team Members
* Ketsia Kemkuini
* Mariza Sitcheu

## Project Structure
* `apps/`: Main application entry points.
* `docs/`: Project documentation and reports (Sphinx).
* `include/`: C++ header files and assembly function prototypes.
* `src/`: Implementation files (C++ and AArch64 Assembly).
* `tests/`: Unit tests using Catch2.
* `.github/workflows/`: CI/CD pipeline configuration.

## Requirements
* **Compiler:** GCC/Clang with AArch64 support.
* **Build System:** CMake 3.10+ (3.19+ to use the presets below).
* **Testing Framework:** Catch2 (integrated via FetchContent).
* **Documentation:** Sphinx with `furo`.
* **SME kernels:** run on SME hardware (Apple M4). They *build* anywhere; the
  SME-executing tests skip on runners without SME, and the norm entry points
  fall back to a scalar reference rather than silently doing nothing.

## Building the Project

Use a preset — each one names a configuration rather than leaving it implicit:

```bash
cmake --preset release            # optimized: ctest + benchmarks
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| preset | build dir | purpose |
|---|---|---|
| `release` | `build/` | everyday work: correctness + benchmarks |
| `debug` | `build-debug/` | Address + UB sanitizers (**not** for timing — instrumentation invalidates every GiB/s figure) |
| `release-submission` | `build-submission/` | the exact configuration behind the reported benchmark tables |

```bash
cmake --preset debug && cmake --build build-debug -j     # sanitizers
cmake --preset release-submission                        # submission build
```

> **Always set a build type.** A bare `cmake ..` leaves `CMAKE_BUILD_TYPE`
> empty, which compiles the C++ reference and the bandwidth probes at `-O0`
> while leaving the `.S` kernels unaffected — an unoptimized C++ probe measured
> against optimized assembly. That defect made single-core NEON read
> 10.8 GiB/s against SSVE's 59.4 and invalidated a round of benchmark tables
> before it was found (Sprint 5). The presets exist mainly so this cannot
> happen by accident; if you configure manually, pass
> `-DCMAKE_BUILD_TYPE=Release`.

Benchmarks print a provenance header — git SHA (marked `-dirty` on a modified
tree), build type, compiler, OS, and the CPU's *detected* SME/SME2 features —
so any reported number carries the conditions that produced it:

```bash
./build/src/norm/main_norm
```