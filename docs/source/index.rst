.. MLC Lab documentation master file, created by
   sphinx-quickstart on Thu Apr 16 10:20:07 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Machine Learning Compiler Lab Report
=====================

This documentation tracks our progress through the Machine Learning Compiler Lab at the University of Jena. 
**GitHub Repository:** `https://github.com/kets01/MLC-Project <https://github.com/kets01/MLC-Project>`_
   

Introduction
------------
This is the documentation for the Machine Learning Compiler Lab project.

Team Members
------------
* Ketsia Kemkuini
* Mariza Yamdjeu

Weekly Tasks & Progress
-----------------------

Week 1: Assembly Language (Inner and Outer product in AArch64)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this first week, we established our initialsoftware engineering workflow:
**Task:** Implemented the inner and outer product in AArch64 assembly.
**Build System:** CMake integration
**Unit Testing:** Integrated catch2 to verify our assembly implementation.
**CI/CD:** Set up GitHub Actions for continuous integration.
**Debugging & Verification**
Below is a trace of the execution flow observed in the debugger:

.. image:: /_static/gdb.png
   :alt: GDB Assembly Layout Trace
   :width: 600px
   :align: center

**GenAI Disclosure:** Used to assist with writing CMakeLists.txt file framework.
**Contributions:**
   * **Mariza Yamdjeu**: 
    * Implemented ``inner_product_asm`` .
    * Set up the CMake build system and Catch2 integration.
    * Performed debugging verification via LLDB/GDB.

* **Ketsia Kemkuini**: 
    * Implemented ``outer_product_asm`` with pointer-increment optimization.
    * Configured the GitHub Actions CI pipeline.
    * Sphinx documentation .