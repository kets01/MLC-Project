#!/usr/bin/env python3
"""PyTorch ATen baseline for LayerNorm / RMSNorm.

Each harness rule below exists because of a specific earlier failure; see
ROADMAP Sprint 7.5 and sprint6_errors.md §2.7-2.8 before changing this file.

  * NATIVE LAYOUT.  PyTorch normalizes the last, contiguous axis of a row-major
    tensor; our kernels are column-major over a strided feature axis.  Both are
    efficient in their own layout, so each is measured in its own — forcing
    either into the other's measures the mismatch, not the kernel.  That is how
    "we are 16-37x faster than Accelerate" happened when, in its own layout,
    that library was in fact faster than us.

  * ONE THREAD.  PyTorch defaults to all cores; our headline numbers are
    single-core, and the threaded comparison is a separate question.

  * SAME BYTE CONVENTION: useful bytes = 1 read + 1 write per element.

  * CORRECTNESS FIRST (decision B).  Outputs are dumped to raw f32 so the C++
    checker can verify them against our float64 reference BEFORE any throughput
    number is trusted.  This is what catches semantic mismatches — PyTorch
    LayerNorm uses the biased variance, and its RMSNorm applies eps inside the
    sqrt.  If the semantics differ, the throughput comparison is meaningless.
"""

import argparse
import os
import struct
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import torch

# Same generator the C++ side uses, so both see bit-identical input.
def make_input(m: int, n: int) -> torch.Tensor:
    idx = torch.arange(m * n, dtype=torch.float64)
    vals = 0.01 * (idx % 97.0)
    return vals.to(torch.float32).reshape(m, n)   # row-major: [rows, features]


def make_gamma_beta(n: int):
    j = torch.arange(n, dtype=torch.float64)
    gamma = (1.0 + 0.001 * (j % 13.0)).to(torch.float32)
    beta = (0.01 * (j % 7.0)).to(torch.float32)
    return gamma, beta


def dump(path: str, t: torch.Tensor) -> None:
    flat = t.detach().contiguous().flatten().to(torch.float32)
    with open(path, "wb") as f:
        f.write(flat.numpy().tobytes())


def best_of(fn, reps: int) -> float:
    fn()                       # warm-up (allocation, lazy init, compile)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def gibs(m: int, n: int, seconds: float) -> float:
    useful = 2.0 * m * n * 4.0          # 1 read + 1 write, FP32
    return (useful / (1024 ** 3)) / seconds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.set_num_threads(1)
    torch.manual_seed(0)

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print("byte convention: useful = 1R+1W per element; layout = row-major, "
          "normalized over the last (contiguous) axis\n")

    shapes = [(128, 64), (1024, 2048), (4096, 8192)]
    eps = 1e-5
    shapes_written: list = []

    hdr = (f"{'shape':>14}{'MiB':>9}{'LN eager':>11}{'LN compile':>13}"
           f"{'RMS eager':>12}{'RMS compile':>13}")
    print(hdr)
    print("-" * len(hdr))

    for (m, n) in shapes:
        x = make_input(m, n)
        gamma, beta = make_gamma_beta(n)
        mib = 2.0 * m * n * 4.0 / (1024 ** 2)

        ln = torch.nn.LayerNorm(n, eps=eps, dtype=torch.float32)
        with torch.no_grad():
            ln.weight.copy_(gamma)
            ln.bias.copy_(beta)

        rms = torch.nn.RMSNorm(n, eps=eps, dtype=torch.float32)
        with torch.no_grad():
            rms.weight.copy_(gamma)

        with torch.no_grad():
            ln_out = ln(x)
            rms_out = rms(x)

            t_ln = best_of(lambda: ln(x), args.reps)
            t_rms = best_of(lambda: rms(x), args.reps)

            # torch.compile is the fair "best PyTorch can do" number; it fuses
            # the op graph, which eager does not.  Reported separately rather
            # than blended, because they are different execution models.
            try:
                ln_c = torch.compile(ln)
                rms_c = torch.compile(rms)
                ln_c(x); rms_c(x)
                t_lnc = best_of(lambda: ln_c(x), args.reps)
                t_rmsc = best_of(lambda: rms_c(x), args.reps)
            except Exception as exc:            # noqa: BLE001
                print(f"  (torch.compile unavailable: {exc})", file=sys.stderr)
                t_lnc = t_rmsc = float("nan")

        print(f"{m}x{n:<9}{mib:9.1f}{gibs(m,n,t_ln):11.2f}{gibs(m,n,t_lnc):13.2f}"
              f"{gibs(m,n,t_rms):12.2f}{gibs(m,n,t_rmsc):13.2f}")

        # Sprint 8: dump EVERY shape, not just the smallest.  Verifying one
        # shape and then publishing three is an inference, not a check -- a
        # kernel can be right at 128x64 and wrong at a tail or a boundary, which
        # is exactly the class of bug this project has hit repeatedly.  The C++
        # checker runs over every dumped shape, so each row of the results
        # table has its own correctness gate.
        tag = f"{m}x{n}"
        dump(os.path.join(args.outdir, f"input_{tag}.f32"), x)
        dump(os.path.join(args.outdir, f"gamma_{tag}.f32"), gamma)
        dump(os.path.join(args.outdir, f"beta_{tag}.f32"), beta)
        dump(os.path.join(args.outdir, f"ln_{tag}.f32"), ln_out)
        dump(os.path.join(args.outdir, f"rms_{tag}.f32"), rms_out)
        shapes_written.append((m, n, eps))

    # Manifest: the checker reads this to know which shapes to verify, and it
    # doubles as the environment record so a rerun can be compared to this one
    # rather than merely resembling it.
    with open(os.path.join(args.outdir, "manifest.txt"), "w") as f:
        f.write(f"# framework torch {torch.__version__}\n")
        f.write(f"# python {sys.version.split()[0]}\n")
        f.write(f"# threads {torch.get_num_threads()}\n")
        for (m, n, e) in shapes_written:
            f.write(f"{m} {n} {e}\n")

    print("\nGiB/s counts useful bytes (1R+1W). Every shape above was dumped and is "
          "verified\nby the C++ checker against our float64 reference before any of "
          "these numbers is\nquoted (decision B) — not just the smallest shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
