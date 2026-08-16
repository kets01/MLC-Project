#!/usr/bin/env python3
"""Single-threaded ExecuTorch baseline for the MLC-Norm comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import statistics
import sys
import time

import torch

from pcore import Occupancy, request_p_core
from torch.export import export


SHAPES = [(128, 64), (1024, 2048), (4096, 8192)]
EPS = 1e-5


def ensure_flatc_on_path() -> None:
    import executorch

    roots = list(getattr(executorch, "__path__", []) or [])
    for root in roots:
        bin_dir = os.path.join(root, "data", "bin")
        if os.path.isdir(bin_dir):
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return
    print("warning: bundled flatc was not found", file=sys.stderr)


def make_input(m: int, n: int) -> torch.Tensor:
    idx = torch.arange(m * n, dtype=torch.float64)
    return (0.01 * (idx % 97.0)).to(torch.float32).reshape(m, n)


def make_gamma_beta(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.arange(n, dtype=torch.float64)
    gamma = (1.0 + 0.001 * (j % 13.0)).to(torch.float32)
    beta = (0.01 * (j % 7.0)).to(torch.float32)
    return gamma, beta


def dump(path: str, tensor: torch.Tensor) -> None:
    tensor.detach().contiguous().to(torch.float32).numpy().tofile(path)


def gibs(m: int, n: int, seconds: float) -> float:
    return (2.0 * m * n * 4.0 / (1024 ** 3)) / seconds


def samples(fn, reps: int) -> tuple[list[float], float]:
    fn()
    out = []
    with Occupancy() as occ:
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            out.append((time.perf_counter_ns() - t0) * 1e-9)
    return out, occ.ratio


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    x = p * (len(ordered) - 1)
    lo = int(x)
    hi = min(lo + 1, len(ordered) - 1)
    frac = x - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def format_stats(sampled: tuple[list[float], float], m: int, n: int) -> str:
    values, occupancy = sampled
    best = min(values)
    median = statistics.median(values)
    p10 = percentile(values, 0.10)
    p90 = percentile(values, 0.90)
    return (
        f"best {gibs(m, n, best):7.2f}  "
        f"median {gibs(m, n, median):7.2f}  "
        f"p10-p90 {gibs(m, n, p90):7.2f}-{gibs(m, n, p10):7.2f}  "
        f"cpu/wall {occupancy:5.2f}"
    )


def build_pte(module: torch.nn.Module, example: torch.Tensor, delegate: bool):
    from executorch.exir import to_edge

    edge = to_edge(export(module.eval(), (example,)))
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


def pin_threads(n: int) -> int:
    """Size the ExecuTorch threadpool to n workers and return what it became.

    torch.set_num_threads() does NOT reach here: ExecuTorch runs its own
    pthreadpool, which both the portable kernels and the XNNPACK delegate use,
    and it defaults to the machine's core count (10 on this M4).  Measured
    without this, the portable RMSNorm path ran at 2.5 CPU-seconds per wall
    second and the XNNPACK one at 1.7 -- so a comparison labelled
    single-threaded was not, and the portable RMSNorm figure was inflated
    roughly 2x (4.23 -> 1.93 GiB/s at 1024x2048).

    The count is READ BACK rather than assumed, and printed into the manifest
    header, so the thread count behind a published number is checkable from the
    artifact instead of taken on trust.
    """
    from executorch.extension.pybindings import portable_lib

    portable_lib._unsafe_reset_threadpool(n)
    return portable_lib._threadpool_get_thread_count()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    # ExecuTorch 1.4.1's XNNPACK lowering fails about half the time on some
    # shapes (see the retry loop below).  At a 50% rate three attempts would
    # still lose a cell once every eight, which is often enough to punch holes
    # in a comparison table; ten makes that negligible.  Retries are reported.
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Before any pool exists: workers inherit the creating thread's QoS.
    pinned = request_p_core()

    ensure_flatc_on_path()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    et_threads = pin_threads(args.threads)

    try:
        et_version = importlib.metadata.version("executorch")
    except importlib.metadata.PackageNotFoundError:
        et_version = "unknown"

    print(f"executorch {et_version}, torch {torch.__version__}, "
          f"torch threads={torch.get_num_threads()}, et threadpool={et_threads}, "
          f"p-core qos={'yes' if pinned else 'no'}")
    print("GiB/s: useful bytes (1R+1W); native row-major layout\n")

    manifest = [
        "# m n epsilon norm implementation filename",
        f"# framework executorch {et_version} (torch {torch.__version__})",
        f"# python {sys.version.split()[0]}",
        f"# threads {torch.get_num_threads()} (executorch threadpool {et_threads}, "
        f"requested {args.threads}, p-core qos {'yes' if pinned else 'no'})",
    ]

    for m, n in SHAPES:
        shape = f"{m}x{n}"
        x = make_input(m, n)
        gamma, beta = make_gamma_beta(n)

        ln = torch.nn.LayerNorm(n, eps=EPS, dtype=torch.float32).eval()
        rms = torch.nn.RMSNorm(n, eps=EPS, dtype=torch.float32).eval()
        with torch.no_grad():
            ln.weight.copy_(gamma)
            ln.bias.copy_(beta)
            rms.weight.copy_(gamma)

        dump(os.path.join(args.outdir, f"input_{shape}.f32"), x)
        dump(os.path.join(args.outdir, f"gamma_{shape}.f32"), gamma)
        dump(os.path.join(args.outdir, f"beta_{shape}.f32"), beta)

        print(shape)
        for norm_name, module in (("layer", ln), ("rms", rms)):
            label = "LayerNorm" if norm_name == "layer" else "RMSNorm  "
            prefix = "ln" if norm_name == "layer" else "rms"

            for mode, delegated in (("portable", False), ("xnnpack", True)):
                # ExecuTorch 1.4.1's XNNPACK lowering is non-deterministic on
                # some shapes: RMSNorm at 1024x2048 fails roughly one attempt in
                # three with "error: 0x23", preceded by "[method.cpp:987] No
                # chains", and succeeds on a retry with byte-identical inputs.
                # Retries are counted and reported rather than hidden, so a row
                # that needed several attempts is visible as such.
                attempts = 0
                last_exc: Exception | None = None
                while attempts < max(1, args.retries):
                    attempts += 1
                    try:
                        runtime = load_runtime(build_pte(module, x, delegated))
                        with torch.inference_mode():
                            output = runtime.forward([x])[0]

                        filename = f"{prefix}_et_{mode}_{shape}.f32"
                        dump(os.path.join(args.outdir, filename), output)
                        manifest.append(
                            f"{m} {n} {EPS} {norm_name} et_{mode} {filename}"
                        )

                        with torch.inference_mode():
                            timing = samples(
                                lambda: runtime.forward([x])[0], args.reps
                            )
                        note = f"   [{attempts} attempts]" if attempts > 1 else ""
                        print(f"  {label} {mode:8s} "
                              f"{format_stats(timing, m, n)}{note}")
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                if last_exc is not None:
                    print(
                        f"  {label} {mode:8s} unavailable after {attempts} "
                        f"attempts: {type(last_exc).__name__}: {last_exc}",
                        file=sys.stderr,
                    )
        print()

    with open(os.path.join(args.outdir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")

    print("Each timed implementation has a separate output entry in manifest.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
