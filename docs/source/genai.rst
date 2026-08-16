GenAI Disclosure
================

Our usage is recorded below.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Tool
     - Used for
   * - Gemini 3
     - Debugging CMake ``FetchContent`` for Catch2; resolving a
       ``Segmentation Fault: 11``; adapting the GitHub Actions pipeline to
       native Apple Silicon runners; debugging a ``SIGILL``; assistance
       implementing recursive tree traversal for nested loops; troubleshooting
       OpenMP discovery on macOS/CMake; the initial reStructuredText **markup
       scaffolding** for this Sphinx site — ``.. toctree::`` directives,
       ``list-table`` skeletons and heading underlines, not prose.
   * - Claude(Fable 5/ Opus)
     - Reviewing our kernels, benchmarks and claims for errors; cross-checking
       measurements against the hardware; and organisational checks on the
       report's structure and consistency. It also assisted with implementation
       and with drafting report text, which we rewrite in our own words.
       Complete codes comments
   * - GPT-5.6 sol
     - Review our paper, generate vectorized Latex code version of the images
       in ``source/_static/figures`` for the paper