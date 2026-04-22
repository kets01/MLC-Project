
Week 2: Performance Microbenchmarking & Tensor Permutation
==========================================================

This week focused on performance analysis of floating-point units and the implementation of an optimized tensor permutation kernel using Neon SIMD instructions.

Task 1: Execution Throughput (FMADD)
------------------------------------
The objective was to measure the maximum execution throughput of the scalar ``FMADD`` instruction.

**Methodology**
To saturate the CPU pipelines and hide instruction latency, we implemented a loop unrolled. We utilized independent registers (``s0`` through ``s7``) to avoid Read-After-Write (RAW) data hazards, allowing the engine to execute multiple instructions in parallel.

**Results (Apple M4-1000000000 iterations)**

*Throughput:* 6.662 G-Instructions/sec

*Performance:* 13.325 GFLOPS

*Observation:* This corresponds to an Instructions Per Cycle (IPC) of ~1.51 on the Apple M4 (for 4,4 GHz), confirming the hardware's ability to execute multiple floating-point operations in parallel across its pipelines.



Task 2: Tensor Permutation (abc -> cba)
---------------------------------------
We developed a kernel to permute a 3D tensor from dimensions :math:`|a|=8, |b|=4, |c|=size\_c` to :math:`|c|, |b|, |a|`.

**Optimization Strategy**

*1.Vectorized Loads*: Loaded 128-bit chunks (4 floats) from the input.

*2.Shuffle Logic*: Used ``TRN1``, ``TRN2``, ``ZIP2``, and ``ZIP2`` instructions to rearrange data within registers.

*3.Burst Stores*: Stored results in contiguous 32-byte rows (8 floats) in the output tensor.

**Verification**
We verified the correctness using a small tensor where :math:`|c|=2`. We obtained the following:

.. code-block:: none

    Input Tensor (abc) [8 x 4 x 4]:
    i=0:
    j=0 | 0.000 1.000 2.000 3.000 
    j=1 | 10.000 11.000 12.000 13.000 
    j=2 | 20.000 21.000 22.000 23.000 
    j=3 | 30.000 31.000 32.000 33.000 
    i=1:
    j=0 | 100.000 101.000 102.000 103.000 
    j=1 | 110.000 111.000 112.000 113.000 
    j=2 | 120.000 121.000 122.000 123.000 
    j=3 | 130.000 131.000 132.000 133.000 
    i=2:
    j=0 | 200.000 201.000 202.000 203.000 
    j=1 | 210.000 211.000 212.000 213.000 
    j=2 | 220.000 221.000 222.000 223.000 
    j=3 | 230.000 231.000 232.000 233.000 
    i=3:
    j=0 | 300.000 301.000 302.000 303.000 
    j=1 | 310.000 311.000 312.000 313.000 
    j=2 | 320.000 321.000 322.000 323.000 
    j=3 | 330.000 331.000 332.000 333.000 
    i=4:
    j=0 | 400.000 401.000 402.000 403.000 
    j=1 | 410.000 411.000 412.000 413.000 
    j=2 | 420.000 421.000 422.000 423.000 
    j=3 | 430.000 431.000 432.000 433.000 
    i=5:
    j=0 | 500.000 501.000 502.000 503.000 
    j=1 | 510.000 511.000 512.000 513.000 
    j=2 | 520.000 521.000 522.000 523.000 
    j=3 | 530.000 531.000 532.000 533.000 
    i=6:
    j=0 | 600.000 601.000 602.000 603.000 
    j=1 | 610.000 611.000 612.000 613.000 
    j=2 | 620.000 621.000 622.000 623.000 
    j=3 | 630.000 631.000 632.000 633.000 
    i=7:
    j=0 | 700.000 701.000 702.000 703.000 
    j=1 | 710.000 711.000 712.000 713.000 
    j=2 | 720.000 721.000 722.000 723.000 
    j=3 | 730.000 731.000 732.000 733.000 

    Output Tensor (cba) [4 x 4 x 8]:
    k=0 (New contiguous dimension is 'a'):
    j=0 | 0.000 100.000 200.000 300.000 400.000 500.000 600.000 700.000 
    j=1 | 10.000 110.000 210.000 310.000 410.000 510.000 610.000 710.000 
    j=2 | 20.000 120.000 220.000 320.000 420.000 520.000 620.000 720.000 
    j=3 | 30.000 130.000 230.000 330.000 430.000 530.000 630.000 730.000 
    k=1 (New contiguous dimension is 'a'):
    j=0 | 1.000 101.000 201.000 301.000 401.000 501.000 601.000 701.000 
    j=1 | 11.000 111.000 211.000 311.000 411.000 511.000 611.000 711.000 
    j=2 | 21.000 121.000 221.000 321.000 421.000 521.000 621.000 721.000 
    j=3 | 31.000 131.000 231.000 331.000 431.000 531.000 631.000 731.000 
    k=2 (New contiguous dimension is 'a'):
    j=0 | 2.000 102.000 202.000 302.000 402.000 502.000 602.000 702.000 
    j=1 | 12.000 112.000 212.000 312.000 412.000 512.000 612.000 712.000 
    j=2 | 22.000 122.000 222.000 322.000 422.000 522.000 622.000 722.000 
    j=3 | 32.000 132.000 232.000 332.000 432.000 532.000 632.000 732.000 
    k=3 (New contiguous dimension is 'a'):
    j=0 | 3.000 103.000 203.000 303.000 403.000 503.000 603.000 703.000 
    j=1 | 13.000 113.000 213.000 313.000 413.000 513.000 613.000 713.000 
    j=2 | 23.000 123.000 223.000 323.000 423.000 523.000 623.000 723.000 
    j=3 | 33.000 133.000 233.000 333.000 433.000 533.000 633.000 733.000


**Performance Scaling**
The following table shows the measured GiB/s on the reference machine:

+----------+-----------+-----------+
| size_c   | GiB/s     | Time(s)   |
+==========+===========+===========+
| 2        | 17.33     | 0.275     |
+----------+-----------+-----------+
| 16       |116.00     |0.329      |
+----------+-----------+-----------+
| 128      |124.64     |2.448      |
+----------+-----------+-----------+
| 512      |126.33     |9.663      |
+----------+-----------+-----------+
| 1024     |70.12      |34.819     |
+----------+-----------+-----------+


*Observation:* The significant performance drop at `∣c∣=1024` highlights the "Memory Wall," where the large strides between tensor dimensions lead to cache misses and increased memory controller latency.