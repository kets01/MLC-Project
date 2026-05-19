Week 3: Unary Operations & SME GEMM
===================================

This week extended the performance‑oriented workflow from Week 2 by introducing
two complementary exercises:

1. High‑throughput unary operations on 16×16 FP32 tiles using SME/SVE.
2. A full 512×512×512 FP32 GEMM kernel implemented using Arm SME streaming mode.

Both tasks required careful attention to memory layout, vectorization strategy,
and benchmarking methodology.

Unary Kernel Microbenchmarking (16×16 FP32)
-------------------------------------------

The unary kernels (``identity``, ``relu``, ``zero``) operate on a 16×16 tile
stored in column‑major format. Each tile contains 256 FP32 elements
(1 KiB per matrix), making it ideal for high‑frequency microbenchmarks.

**Methodology**

* 10 million iterations were executed per kernel to obtain stable bandwidth
  measurements.
* Each kernel was implemented in both C++ (reference) and SME/SVE assembly
  (optimized).
* Bandwidth was computed as:

  *Identity / ReLU:*  Read(A) + Write(B)  
  *Zero:*             Write(B) only

**Results**

+-----------+-------------+
| Kernel    | GiB/s       |
+===========+=============+
| Identity  | 22.41       |
+-----------+-------------+
| ReLU      | 21.03       |
+-----------+-------------+
| Zero      | 16.29       |
+-----------+-------------+

**Observation**

The identity and ReLU kernels achieve over **20 GiB/s**, demonstrating efficient
use of SME/SVE vector pipelines. The ``zero`` kernel is write‑bound and
therefore achieves lower bandwidth, consistent with expectations.

SME GEMM Benchmark (512×512×512 FP32)
--------------------------------------

The second task involved implementing a GEMM kernel using Arm SME in streaming
mode. The matrices use mixed layouts:

* ``A``: column‑major  
* ``B``: row‑major  
* ``C``: column‑major  

A warm‑up phase was used to activate SME streaming mode and stabilize CPU
frequency before timed execution.

**Results (100 iterations)**
 
*Performance:* **1529.38 GFLOPS**


