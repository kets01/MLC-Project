#!/usr/bin/env python3
"""Generate the report figures.

    python3 docs/figures/make_figures.py

Writes SVGs into docs/source/_static/figures/.  No third-party dependencies.

PROVENANCE.  Every number below is transcribed from a single benchmark run,
captured verbatim in

    docs/source/_static/results/main_norm_submission.txt

produced by `./build-submission/src/norm/main_norm` (the `release-submission`
CMake preset) on the Apple M4, macOS 15.2, AppleClang 16.0.0.  That file opens
with the run's provenance header — git commit, build type, compiler, detected
FEAT_SME/FEAT_SME2 — so a figure here can always be traced to the run that
produced it.  Values are literals rather than parsed from the log on purpose:
parsing a human-readable table is brittle, and a wrong figure is worse than an
inconvenient one.  If the benchmark is re-run, update both files together.
"""

from pathlib import Path

from svgplot import Fig, PALETTE, MUTED, nice_ymax

OUT = Path(__file__).resolve().parent.parent / "source" / "_static" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data (submission run — see PROVENANCE above)
# --------------------------------------------------------------------------

# Bandwidth ceiling vs working-set size, both execution modes.
FOOTPRINT_MIB = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
NEON = [293.44, 225.64, 162.76, 126.01, 125.34, 124.34, 124.01,
        123.84, 118.67, 100.52, 85.01, 81.27, 79.28]
SSVE = [104.51, 97.66, 101.05, 117.21, 116.04, 116.03, 115.88,
        115.67, 115.35, 96.13, 63.11, 60.17, 59.45]

# Consolidated ablation at 4096x8192 (256 MiB, true DRAM), useful bytes.
ABL_STAGES = ["scalar ref", "V0 SSVE", "V4 (ILP)", "V6 (contig)",
              "V7 (SME2)", "ZA residency", "JIT (auto)"]
ABL_RMS = [0.67, 10.37, 10.75, 20.97, 24.76, 10.37, 24.76]
ABL_LN = [0.53, 7.40, 7.95, 13.21, 13.55, 7.51, 13.47]

# Sprint 7a: relative error of each FP32 variance formulation vs a float64
# oracle, as the input is shifted (kappa ~ 1 + shift^2).
SHIFTS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
ERR_NAIVE = [5.09e-7, 1.39e-4, 8.52e-3, 1.25e0, 1.59e2, 1.02e4, 6.80e6]
ERR_TWOPASS = [3.89e-7, 3.98e-7, 1.18e-7, 3.75e-7, 2.81e-5, 1.51e-7, 7.43e-7]
ERR_WELFORD = [2.07e-7, 1.98e-7, 4.16e-7, 4.73e-6, 5.41e-5, 8.94e-5, 6.71e-3]
NAIVE_NEGATIVE_FROM = 1e5      # returns a negative variance -> sqrt = NaN

# Sprint 7b: microseconds per call, RMSNorm at N=512, sweeping rows.
ROWS = [1, 2, 4, 8, 16, 64, 256]
T_SCALAR = [0.250, 1.041, 2.166, 4.375, 8.833, 66.666, 378.375]
T_V7 = [7.708, 6.834, 6.250, 6.208, 6.167, 6.208, 25.291]
GROUP_ROWS = 64                # 4 x VL, VL = 16 FP32 lanes at SVL = 512

# Sprint 7.5: external baselines, single-threaded, native layouts, 256 MiB.
BASE_LABELS = ["ours (V7)", "torch eager", "torch.compile",
               "ET portable", "ET XNNPACK"]
BASE_RMS = [24.76, 2.76, 5.90, 3.26, 4.19]
BASE_LN = [13.55, 7.35, 4.44, 5.92, 5.93]


# --------------------------------------------------------------------------
def fig_ceiling():
    """The project's central correction: the ceiling is a curve."""
    f = Fig(width=780, height=450)
    f.set_scales(0.055, 300, 0, 320, xlog=True)
    f.frame("Bandwidth ceiling depends on working-set size",
            "working set (MiB, log scale)", "GiB/s (useful bytes, 1R+1W)",
            "Apple M4, single core. Dividing a cache-resident kernel by the "
            "DRAM ceiling inflates its % of peak ~2x.")
    f.yticks([0, 50, 100, 150, 200, 250, 300])
    f.xticks([0.0625, 0.25, 1, 4, 16, 64, 256],
             ["64 KiB", "256 KiB", "1 MiB", "4 MiB", "16 MiB", "64 MiB", "256 MiB"])

    # The single constant every percentage used before Sprint 6.
    y = f.py(59.45)
    f.line(f.pl, y, f.w - f.pr, y, stroke="#b03060", width=1.4, dash="6,4")
    f.text(f.pl + 8, y + 17, "59.5 GiB/s — the single DRAM constant used as",
           size=10.5, fill="#b03060")
    f.text(f.pl + 8, y + 30, "THE denominator for every % of peak until Sprint 6",
           size=10.5, fill="#b03060")

    f.polyline(zip(FOOTPRINT_MIB, NEON), PALETTE[0])
    f.markers(list(zip(FOOTPRINT_MIB, NEON)), PALETTE[0])
    f.add_legend("NEON (non-streaming)", PALETTE[0])
    f.polyline(zip(FOOTPRINT_MIB, SSVE), PALETTE[1])
    f.markers(list(zip(FOOTPRINT_MIB, SSVE)), PALETTE[1])
    f.add_legend("SSVE (streaming) — the kernels' mode", PALETTE[1])

    # Where the two reported shapes actually sit.
    for mib, lab in ((16, "16 MiB shape:\nceiling 115.4, not 59.5"),
                     (256, "256 MiB shape:\nceiling 59.5")):
        x = f.px(mib)
        f.line(x, f.pt, x, f.h - f.pb, stroke=MUTED, width=1, dash="3,3")
    f.text(f.px(16) - 8, f.pt + 92, "16 MiB shape:", size=10.5, anchor="end", fill=MUTED)
    f.text(f.px(16) - 8, f.pt + 105, "ceiling 115.4, not 59.5", size=10.5,
           anchor="end", fill=MUTED)

    # The sub-MiB divergence is streaming-entry cost, not bandwidth.  Placed
    # low-left, clear of the NEON curve which starts near the top of the axis.
    ynote = f.py(30)
    f.text(f.pl + 8, ynote, "below ~512 KiB the SSVE figure is not pure",
           size=10.5, fill=MUTED)
    f.text(f.pl + 8, ynote + 13, "bandwidth: SMSTART is a visible share of",
           size=10.5, fill=MUTED)
    f.text(f.pl + 8, ynote + 26, "a sub-microsecond pass (Sprint 7b)",
           size=10.5, fill=MUTED)

    f.draw_legend(x=f.pl + 250, y=f.pt + 14)
    return f.save(OUT / "ceiling_curve.svg")


def fig_ablation():
    """What each lever bought, in the regime where it matters."""
    n = len(ABL_STAGES)
    f = Fig(width=780, height=430, pad_l=118, pad_b=58)
    top = nice_ymax(max(ABL_RMS + ABL_LN))
    f.set_scales(0, n, 0, top)
    f.frame("Ablation at 256 MiB (true DRAM), useful bytes",
            "", "GiB/s", "Every bar was correctness-gated against the float64 "
            "reference before it was timed (66/66 configurations).")
    f.yticks([0, 5, 10, 15, 20, 25])

    bw = (f.w - f.pl - f.pr) / n
    for i, stage in enumerate(ABL_STAGES):
        x0 = f.px(i)
        for j, (vals, col) in enumerate(((ABL_RMS, PALETTE[0]),
                                         (ABL_LN, PALETTE[1]))):
            bx = x0 + bw * (0.16 + 0.34 * j)
            by = f.py(vals[i])
            f.rect(bx, by, bw * 0.30, f.py(0) - by, col)
            f.text(bx + bw * 0.15, by - 5, f"{vals[i]:.1f}", size=10,
                   anchor="middle")
        f.text(x0 + bw / 2, f.h - f.pb + 17, stage, size=10.5, anchor="middle")

    f.add_legend("RMSNorm (2 traversals, 2R+1W)", PALETTE[0])
    f.add_legend("LayerNorm (3 traversals, 3R+1W)", PALETTE[1])
    f.draw_legend(x=f.pl + 12, y=f.pt + 8)

    f.text(f.pl + 12, f.pt + 56,
           "V6 (access density) is the big lever; V7 (SME2 multi-vector) adds "
           "+17% on RMSNorm; ZA loses to both.", size=10.5, fill=MUTED)
    return f.save(OUT / "ablation_dram.svg")


def fig_stability():
    """Where each variance formulation breaks."""
    import math
    f = Fig(width=780, height=440)
    f.set_scales(1, 1e6, -7.5, 7.5, xlog=True)
    f.frame("FP32 variance: relative error vs input shift",
            "input shift (condition number kappa ~ 1 + shift²)",
            "log10 relative error vs float64 oracle",
            "N=512. Naive single-pass tracks kappa and fails catastrophically; "
            "two-pass is flat across 12 orders.")
    f.yticks([-6, -4, -2, 0, 2, 4, 6], fmt="1e{:.0f}")
    f.xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
             ["0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6"])

    for vals, col, lab in ((ERR_NAIVE, PALETTE[1], "naive E[x²]−mean² (1 pass)"),
                           (ERR_WELFORD, PALETTE[3], "Welford (1 pass)"),
                           (ERR_TWOPASS, PALETTE[0], "centred two-pass (ours)")):
        pts = [(s, math.log10(v)) for s, v in zip(SHIFTS, vals)]
        f.polyline(pts, col)
        f.markers(pts, col)
        f.add_legend(lab, col)

    # Mark where naive stops being merely inaccurate and becomes unusable.
    x = f.px(NAIVE_NEGATIVE_FROM)
    f.line(x, f.pt, x, f.h - f.pb, stroke="#b03060", width=1.3, dash="5,4")
    f.text(x - 8, f.pt + 22, "naive returns NEGATIVE variance", size=11,
           anchor="end", fill="#b03060", weight="bold")
    f.text(x - 8, f.pt + 36, "→ sqrt = NaN, not just inaccurate", size=11,
           anchor="end", fill="#b03060")

    f.draw_legend(x=f.pl + 12, y=f.pt + 8)
    return f.save(OUT / "stability.svg")


def fig_crossover():
    """When does the SME kernel win, and why not sooner."""
    f = Fig(width=780, height=430)
    f.set_scales(1, 256, 0, 400, xlog=True)
    f.frame("When does the SME kernel beat the scalar reference?",
            "rows M (N=512, log scale)", "microseconds per call",
            "Crossover at M≈16 — but the cause is group granularity, not "
            "streaming-mode overhead.")
    f.yticks([0, 100, 200, 300, 400])
    f.xticks(ROWS, [str(r) for r in ROWS])

    f.polyline(zip(ROWS, T_SCALAR), PALETTE[1])
    f.markers(list(zip(ROWS, T_SCALAR)), PALETTE[1])
    f.add_legend("scalar reference", PALETTE[1])
    f.polyline(zip(ROWS, T_V7), PALETTE[0])
    f.markers(list(zip(ROWS, T_V7)), PALETTE[0])
    f.add_legend("SME kernel (V7)", PALETTE[0])

    x = f.px(GROUP_ROWS)
    f.line(x, f.pt, x, f.h - f.pb, stroke=MUTED, width=1.2, dash="4,4")
    f.text(x + 8, f.pt + 24, f"one group = {GROUP_ROWS} rows (4 × VL)", size=11,
           fill=MUTED)
    f.text(x + 8, f.pt + 38, "V7 is flat from M=1 to M=64: a partial", size=10.5,
           fill=MUTED)
    f.text(x + 8, f.pt + 51, "group costs the same as a full one", size=10.5,
           fill=MUTED)
    f.text(f.pl + 12, f.h - f.pb - 22,
           "A 9 ns SMSTART cannot explain a 6 µs plateau; the group can, exactly.",
           size=11, fill=MUTED)

    f.draw_legend(x=f.pl + 12, y=f.pt + 8)
    return f.save(OUT / "crossover.svg")


def fig_baselines():
    """Against two general-purpose frameworks, same function, same shapes."""
    n = len(BASE_LABELS)
    f = Fig(width=780, height=430, pad_l=100, pad_b=76)
    f.set_scales(0, n, 0, nice_ymax(max(BASE_RMS + BASE_LN)))
    f.frame("External baselines at 256 MiB, single-threaded, native layouts",
            "", "GiB/s (useful bytes)",
            "PyTorch 2.13.0 / ExecuTorch 1.4.1. Every shape verified to compute "
            "the same function before timing.")
    f.yticks([0, 5, 10, 15, 20, 25])

    bw = (f.w - f.pl - f.pr) / n
    for i, lab in enumerate(BASE_LABELS):
        x0 = f.px(i)
        for j, (vals, col) in enumerate(((BASE_RMS, PALETTE[0]),
                                         (BASE_LN, PALETTE[1]))):
            bx = x0 + bw * (0.16 + 0.34 * j)
            by = f.py(vals[i])
            f.rect(bx, by, bw * 0.30, f.py(0) - by, col)
            f.text(bx + bw * 0.15, by - 5, f"{vals[i]:.1f}", size=10,
                   anchor="middle")
        f.text(x0 + bw / 2, f.h - f.pb + 17, lab, size=10.5, anchor="middle")

    f.add_legend("RMSNorm", PALETTE[0])
    f.add_legend("LayerNorm", PALETTE[1])
    f.draw_legend(x=f.pl + 12, y=f.pt + 8)

    f.text(f.pl + 12, f.h - f.pb + 44,
           "Note the inversion: our RMSNorm beats our LayerNorm; PyTorch's "
           "LayerNorm beats its RMSNorm —", size=10.5, fill=MUTED)
    f.text(f.pl + 12, f.h - f.pb + 58,
           "because its CPU RMSNorm decomposes into elementwise ops instead of "
           "running a fused kernel.", size=10.5, fill=MUTED)
    return f.save(OUT / "baselines.svg")


def fig_traversals():
    """The structural difference the whole project rests on."""
    f = Fig(width=780, height=300, pad_l=150, pad_t=64, pad_b=54)
    f.frame("Why RMSNorm is faster: one fewer reduction stage",
            "", "", "Reduction stages and input traversals — the terms "
            "'two-pass' and 'single-pass' conflate these.")

    rows = [
        ("RMSNorm", ["read x → Σx²", "read x → normalise, scale γ, write y"],
         PALETTE[0], "2 traversals · 2R+1W"),
        ("LayerNorm", ["read x → mean", "read x → variance (centred)",
                       "read x → normalise, scale γ, shift β, write y"],
         PALETTE[1], "3 traversals · 3R+1W"),
    ]
    y = f.pt + 18
    for name, steps, col, summary in rows:
        f.text(f.pl - 14, y + 16, name, size=13, anchor="end", weight="bold")
        x = f.pl
        for s in steps:
            w = 172
            f.rect(x, y, w - 8, 30, col, opacity=0.85)
            f.text(x + (w - 8) / 2, y + 19, s, size=9.5, anchor="middle",
                   fill="white")
            x += w
        f.text(x + 4, y + 19, summary, size=11, fill=MUTED)
        y += 74

    f.text(f.pl - 14, y + 4, "ratio", size=12, anchor="end", fill=MUTED)
    f.text(f.pl, y + 4, "3R+1W ÷ 2R+1W = 1.33× more traffic per element for "
           "LayerNorm — the structural source of the gap.", size=11.5)
    return f.save(OUT / "traversals.svg")


if __name__ == "__main__":
    for fn in (fig_ceiling, fig_ablation, fig_stability, fig_crossover,
               fig_baselines, fig_traversals):
        print("wrote", fn())
