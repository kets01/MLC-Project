# Baseline run manifests

Written by the benchmark drivers, checked in as the provenance record for the
external-baseline numbers in `docs/source/weeks/norm_sprint7_5.rst`.

Each manifest lists the framework and Python versions of the run, the thread
count, and every `(M, N, eps)` shape that was measured. The C++ checker
(`bench/baselines/verify_baseline.cpp`) reads exactly this list, so the set of
shapes that were *timed* and the set that were *correctness-gated* cannot drift
apart — verifying one shape and publishing three is an inference, not a check.

Regenerate with the pinned environment in `../requirements-baselines.txt`; if a
rerun produces different versions, it is a different measurement and the report
figures should not be compared to it without saying so.
