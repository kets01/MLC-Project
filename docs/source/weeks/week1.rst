Week 1: AArch64 Assembly
========================


Task Description
----------------
The goal was to implement the *Inner Product* and *Outer Product* for 32-bit unsigned integer vectors, ensuring 64-bit precision for the results.

Implementation Details
----------------------

Inner Product
^^^^^^^^^^^^^
We utilized the `umaddl` (Unsigned Multiply-Add Long) instruction. This allows us to multiply two 32-bit values and add them to a 64-bit accumulator in a single cycle.

Outer Product
^^^^^^^^^^^^^
The outer product was optimized using *post-indexing pointer arithmetic*. By using instructions like `str x8, [x3], #8`, we avoided expensive index calculations (i * size + j) inside the nested loops.

Debugging & Verification
------------------------
Below is a trace of the execution flow observed in the debugger:

.. image:: /_static/gdb.png
   :alt: GDB Assembly Layout Trace
   :width: 600px
   :align: center