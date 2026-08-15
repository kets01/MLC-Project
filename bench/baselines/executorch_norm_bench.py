#!/usr/bin/env python3
"""Sprint 7.5 - ExecuTorch baseline for LayerNorm / RMSNorm.

Same harness rules as torch_norm_bench.py (native row-major layout, one thread,
useful-bytes convention, outputs dumped for cross-verification against our
float64 reference before any number is quoted).

ExecuTorch-specific notes, because they change what the number MEANS:

  * Two execution paths are measured, because they are different products:
      - PORTABLE: reference kernels, no delegation.  This is what a bare
        ExecuTorch runtime does and it is not meant to be fast.
      - XNNPACK: the delegated path, which is what anyone actually deploys.
        Quoting only the portable number would be the mirror image of the
        Sprint-6 error - handicapping the baseline and calling the gap a win.
  * The measured time includes ExecuTorch's per-invocation runtime dispatch.
    That is honest for an on-device comparison (it is what the op costs in
    that runtime), but it is NOT a pure kernel number, and the report says so.
  * Versions are NOT pinned in this file; they are recorded at run time into
    the output manifest, because that is the only version information that is
    guaranteed to describe the run that produced the numbers.  The reported
    measurements were taken with ExecuTorch 1.4.1 / torch 2.13.0 on Python
    3.12.  (An earlier attempt on Python 3.9 could only install ExecuTorch
    0.3.0, whose XNNPACK partitioner does not even import there -- staticmethod
    objects became callable only in Python 3.10 -- so the delegated path, the
    one that is actually deployed, could not be measured at all.  Reporting
    only the portable kernels would have understated the baseline, which is the
    mirror image of the Sprint-6 error, so a newer interpreter was used
    instead.  Requires Python >= 3.10.)
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import torch
from torch.export import export


def ensure_flatc_on_path() -> None:
    """ExecuTorch serializes .pte by shelling out to `flatc` found on PATH.

    The wheel ships its own flatc under executorch/data/bin/ but does not add
    that directory to PATH, so `to_executorch()` dies with a FileNotFoundError
    wrapped in a misleading 'Failed to compile data.json to data.pte'.  Putting
    the bundled binary on PATH is the fix, and doing it here keeps the script
    self-contained rather than depending on how it was invoked.
    """
    import executorch

    # executorch is a namespace package, so __file__ is None; __path__ is the
    # reliable locator.
    roots = list(getattr(executorch, "__path__", []) or [])
    for root in roots:
        bin_dir = os.path.join(root, "data", "bin")
        if os.path.isdir(bin_dir):
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return
    print("warning: bundled flatc not found; .pte serialization will likely fail")


def make_input(m: int, n: int) -> torch.Tensor:
    idx = torch.arange(m * n, dtype=torch.float64)
    return (0.01 * (idx % 97.0)).to(torch.float32).reshape(m, n)


def make_gamma_beta(n: int):
    j = torch.arange(n, dtype=torch.float64)
    return ((1.0 + 0.001 * (j % 13.0)).to(torch.float32),
            (0.01 * (j % 7.0)).to(torch.float32))


def dump(path: str, t: torch.Tensor) -> None:
    flat = t.detach().contiguous().flatten().to(torch.float32)
    with open(path, "wb") as f:
        f.write(flat.numpy().tobytes())


def gibs(m: int, n: int, seconds: float) -> float:
    return (2.0 * m * n * 4.0 / (1024 ** 3)) / seconds


def build_pte(module: torch.nn.Module, example: torch.Tensor, delegate: bool):
    """Export `module` to an ExecuTorch program buffer."""
    from executorch.exir import to_edge

    exported = export(module, (example,))
    edge = to_edge(exported)
    if delegate:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        edge = edge.to_backend(XnnpackPartitioner())
    return edge.to_executorch().buffer


def load_runtime(buffer):
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    return _load_for_executorch_from_buffer(buffer)


def best_of(fn, reps: int) -> float:
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ensure_flatc_on_path()
    torch.set_num_threads(1)
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print("ExecuTorch runtime; row-major native layout; useful-bytes (1R+1W)\n")

    shapes = [(128, 64), (1024, 2048), (4096, 8192)]
    eps = 1e-5

    hdr = (f"{'shape':>14}{'MiB':>9}{'LN portable':>13}{'LN xnnpack':>13}"
           f"{'RMS portable':>14}{'RMS xnnpack':>13}")
    print(hdr)
    print("-" * len(hdr))

    shapes_written: list = []

    for (m, n) in shapes:
        x = make_input(m, n)
        gamma, beta = make_gamma_beta(n)
        mib = 2.0 * m * n * 4.0 / (1024 ** 2)

        ln = torch.nn.LayerNorm(n, eps=eps, dtype=torch.float32).eval()
        with torch.no_grad():
            ln.weight.copy_(gamma); ln.bias.copy_(beta)
        rms = torch.nn.RMSNorm(n, eps=eps, dtype=torch.float32).eval()
        with torch.no_grad():
            rms.weight.copy_(gamma)

        row = {}
        outputs = {}
        for tag, mod in (("LN", ln), ("RMS", rms)):
            for mode, delegate in (("portable", False), ("xnnpack", True)):
                key = f"{tag}_{mode}"
                try:
                    buf = build_pte(mod, x, delegate)
                    rt = load_runtime(buf)
                    out = rt.forward([x])[0]
                    row[key] = gibs(m, n, best_of(lambda: rt.forward([x]), args.reps))
                    outputs[key] = out
                except Exception as exc:  # noqa: BLE001
                    row[key] = float("nan")
                    print(f"  [{key} @ {m}x{n}] failed: {type(exc).__name__}: "
                          f"{str(exc)[:160]}")

        print(f"{m}x{n:<9}{mib:9.1f}{row['LN_portable']:13.2f}"
              f"{row['LN_xnnpack']:13.2f}{row['RMS_portable']:14.2f}"
              f"{row['RMS_xnnpack']:13.2f}")

        # Sprint 8: dump EVERY shape so each row of the results table carries
        # its own correctness gate, rather than inferring three shapes from one.
        # The XNNPACK outputs are the ones dumped where available, since that is
        # the path the reported numbers come from.
        shape_tag = f"{m}x{n}"
        dump(os.path.join(args.outdir, f"input_{shape_tag}.f32"), x)
        dump(os.path.join(args.outdir, f"gamma_{shape_tag}.f32"), gamma)
        dump(os.path.join(args.outdir, f"beta_{shape_tag}.f32"), beta)
        for key, fname in (("LN", f"ln_{shape_tag}.f32"), ("RMS", f"rms_{shape_tag}.f32")):
            out = outputs.get(f"{key}_xnnpack", outputs.get(f"{key}_portable"))
            if out is not None:
                dump(os.path.join(args.outdir, fname), out)
        shapes_written.append((m, n, eps))

    # Manifest: which shapes the checker must verify, plus the exact versions
    # this run used.  Versions are recorded rather than pinned in source,
    # because a rerun months later must be comparable to THIS run, not merely
    # resemble it.
    try:
        from importlib.metadata import version as _pkg_version
        et_version = _pkg_version("executorch")
    except Exception:  # noqa: BLE001
        et_version = "unknown"
    with open(os.path.join(args.outdir, "manifest.txt"), "w") as f:
        f.write(f"# framework executorch {et_version} (torch {torch.__version__})\n")
        f.write(f"# python {sys.version.split()[0]}\n")
        f.write(f"# threads {torch.get_num_threads()}\n")
        for (m, n, e) in shapes_written:
            f.write(f"{m} {n} {e}\n")

    print("\nPortable = reference kernels (not meant to be fast). XNNPACK = the "
          "delegated\npath that is actually deployed. Times include ExecuTorch's "
          "per-invocation\ndispatch, which is honest for an on-device comparison "
          "but is not a pure\nkernel number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
