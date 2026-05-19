Week 5: JIT Compilation for SME Kernels
=======================================

This week extended the optimization work from Week 3 by introducing
**runtime code generation (JIT compilation)** for previously developed
unary kernels and GEMM kernels.

The objective was to dynamically generate executable AArch64 machine code
at runtime instead of relying solely on statically compiled kernels.

JIT Runtime Implementation
--------------------------

A reusable ``JitEngine`` was developed to dynamically generate executable
kernels during runtime.

The workflow included:

* Generating raw AArch64 machine opcodes
* Allocating executable memory using ``mmap()``
* Using ``MAP_JIT`` for Apple Silicon compatibility
* Writing instructions into memory
* Invalidating instruction caches using ``sys_icache_invalidate()``
* Executing generated kernels through function pointers


JIT Kernel Validation
---------------------

Correctness tests were developed for all dynamically generated kernels.

The validation process included:

* Comparing JIT outputs with reference implementations
* Verifying executable memory permissions
* Testing function pointer execution
* Debugging runtime-generated opcode behavior

To isolate failures, progressively simpler kernels were tested.

A minimal kernel containing only:

.. code-block:: cpp

   return {
       0xd65f03c0
   };

executed successfully, confirming that:

* JIT memory allocation worked correctly
* ``MAP_JIT`` configuration was valid
* runtime execution was functioning properly

**Resolution of Encoding Issues**

Initial attempts resulted in Illegal instruction: 4. This was resolved by reviewing our encoding process:

```bash
# 1. Compile with clang to handle #ifdefs and SME/SVE architecture flags
clang -arch arm64 -march=armv9-a+sme+sve -c gemm_512_512_512.S -o temp.o

# 2. Extract opcodes and format into a C++ uint32_t array
/opt/homebrew/opt/llvm/bin/llvm-objdump -d temp.o | ./formatter.o

Runtime Execution Results
-------------------------

The following table shows the performance for our differents kernels (Identity, ReLU, Zero, and GEMM).

+--------------+--------------+
| Kernel       | Performance  | 
+==============+==============+
| JIT Identity | 22.11 GiB/s  |
+--------------+--------------+
| JIT ReLU     |21.95 GiB/s   |
+--------------+--------------+
| JIT Zero     |16.46 GiB/s   |
+--------------+--------------+
| JIT GEMM	   |1516.56 GFLOPS|
+--------------+--------------+

The overall performance looks similar to week3 implementation(without JIT), proving that the dynamic generation process introduces zero runtime overhead.

