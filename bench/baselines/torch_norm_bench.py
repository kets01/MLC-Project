#!/usr/bin/env python3
"""PyTorch baseline for the MLC-Norm comparison, at a chosen thread count."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

from pcore import Occupancy, request_p_core


SHAPES = [(128, 64), (1024, 2048), (4096, 8192)]
EPS = 1e-5


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


def samples(fn, reps: int) -> tuple[list[float], float]:
    fn()  # warm-up
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


def stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        min(values),
        statistics.median(values),
        percentile(values, 0.10),
        percentile(values, 0.90),
    )


def gibs(m: int, n: int, seconds: float) -> float:
    # Project-wide convention: useful bytes = one input read + one output write.
    return (2.0 * m * n * 4.0 / (1024 ** 3)) / seconds


def format_stats(sampled: tuple[list[float], float], m: int, n: int) -> str:
    values, occupancy = sampled
    best, median, p10, p90 = stats(values)
    # Time percentiles reverse when expressed as throughput.
    return (
        f"best {gibs(m, n, best):7.2f}  "
        f"median {gibs(m, n, median):7.2f}  "
        f"p10-p90 {gibs(m, n, p90):7.2f}-{gibs(m, n, p10):7.2f}  "
        f"cpu/wall {occupancy:5.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Before the first torch op: the intra-op pool inherits the QoS of the
    # thread that creates it, and is created lazily on first use.
    pinned = request_p_core()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, "
          f"p-core qos={'yes' if pinned else 'no'}")
    print("GiB/s: useful bytes (1R+1W); native row-major layout\n")

    manifest = [
        "# m n epsilon norm implementation filename",
        f"# framework torch {torch.__version__}",
        f"# python {sys.version.split()[0]}",
        f"# threads {torch.get_num_threads()} (requested {args.threads}, "
        f"p-core qos {'yes' if pinned else 'no'})",
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

        with torch.inference_mode():
            eager_ln_out = ln(x)
            eager_rms_out = rms(x)

        eager_ln_file = f"ln_torch_eager_{shape}.f32"
        eager_rms_file = f"rms_torch_eager_{shape}.f32"
        dump(os.path.join(args.outdir, eager_ln_file), eager_ln_out)
        dump(os.path.join(args.outdir, eager_rms_file), eager_rms_out)
        manifest.append(f"{m} {n} {EPS} layer torch_eager {eager_ln_file}")
        manifest.append(f"{m} {n} {EPS} rms torch_eager {eager_rms_file}")

        compiled_ln = None
        compiled_rms = None
        try:
            compiled_ln = torch.compile(ln)
            compiled_rms = torch.compile(rms)
            with torch.inference_mode():
                compiled_ln_out = compiled_ln(x)
                compiled_rms_out = compiled_rms(x)

            compiled_ln_file = f"ln_torch_compile_{shape}.f32"
            compiled_rms_file = f"rms_torch_compile_{shape}.f32"
            dump(os.path.join(args.outdir, compiled_ln_file), compiled_ln_out)
            dump(os.path.join(args.outdir, compiled_rms_file), compiled_rms_out)
            manifest.append(
                f"{m} {n} {EPS} layer torch_compile {compiled_ln_file}"
            )
            manifest.append(
                f"{m} {n} {EPS} rms torch_compile {compiled_rms_file}"
            )
        except Exception as exc:  # compilation availability is environment-dependent
            print(f"{shape}: torch.compile unavailable: {exc}", file=sys.stderr)
            compiled_ln = compiled_rms = None

        with torch.inference_mode():
            eager_ln_t = samples(lambda: ln(x), args.reps)
            eager_rms_t = samples(lambda: rms(x), args.reps)

        print(shape)
        print(f"  LayerNorm eager   {format_stats(eager_ln_t, m, n)}")
        print(f"  RMSNorm   eager   {format_stats(eager_rms_t, m, n)}")

        if compiled_ln is not None and compiled_rms is not None:
            with torch.inference_mode():
                compiled_ln_t = samples(lambda: compiled_ln(x), args.reps)
                compiled_rms_t = samples(lambda: compiled_rms(x), args.reps)
            print(f"  LayerNorm compile {format_stats(compiled_ln_t, m, n)}")
            print(f"  RMSNorm   compile {format_stats(compiled_rms_t, m, n)}")
        print()

    with open(os.path.join(args.outdir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")

    print("Each timed implementation has a separate output entry in manifest.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
