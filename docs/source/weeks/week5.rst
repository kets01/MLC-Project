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

The generated kernels included:

* ``identity``
* ``relu``
* ``zero``
* ``gemm``

These kernels reused the SME/SVE instructions implemented in Week 3 such as:

* ``smstart`` / ``smstop``
* ``ptrue``
* ``ld1w`` / ``st1w``
* ``fmopa``

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

Runtime Execution Results
-------------------------

When full SME kernels were executed dynamically, the program failed with:

.. code-block:: bash

   Illegal instruction: 4

The issue occurred specifically for SME/SVE instructions such as:

* ``smstart``
* ``ptrue``
* ``ld1w/st1w``
* ``fmopa``

Observation
-----------


* The same SME kernels worked correctly when statically compiled
* The same instructions failed when dynamically generated

This suggests that macOS imposes restrictions on executing certain advanced architectural
instructions from JIT-generated pages.

