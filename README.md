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
* **Build System:** CMake 3.10+
* **Testing Framework:** Catch2 (integrated via FetchContent).
* **Documentation:** Sphinx with `sphinx-rtd-theme`.

## Building the Project
To build the project and tests, run the following commands:

```bash
mkdir build && cd build
cmake ..
make