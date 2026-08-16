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
FEAT_SME/FEAT_SME2 — so any figure here traces to the run that produced it.
Values are literals rather than parsed from the log on purpose: parsing a
human-readable table is brittle, and a wrong figure is worse than an
inconvenient one.  Re-running the benchmark means updating both files together.

DESIGN.  Each figure is built to make its point *visually* rather than assert it
in a caption — shaded regimes rather than a sentence about regimes, a shaded
error band rather than "the percentages were inflated", leader lines so
annotations sit in whitespace instead of on top of the data.
"""

from pathlib import Path

from svgplot import (Fig, BLUE, VERM, GREEN, ORANGE, PURPLE, INK, MUTED,
                     ACCENT, BAND, measure, nice_ymax)

OUT = Path(__file__).resolve().parent.parent / "source" / "_static" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data (submission run — see PROVENANCE above)
# --------------------------------------------------------------------------
FOOTPRINT_MIB = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
NEON = [244.14, 225.22, 158.43, 110.57, 109.53, 109.27, 108.76,
        108.82, 105.57, 92.26, 82.22, 79.35, 78.08]
SSVE = [104.69, 97.66, 117.21, 117.21, 116.04, 116.03, 115.74,
        115.60, 115.60, 91.25, 62.16, 59.46, 58.78]
DRAM_CONST = 58.78          # the single denominator used until Sprint 6

ABL_STAGES = ["scalar\nref", "V0\nSSVE", "V4\nILP", "V6\ncontiguity",
              "V7\nSME2", "ZA\nresidency", "JIT\nauto"]
ABL_RMS = [0.59, 10.04, 10.67, 21.05, 24.63, 10.29, 24.68]
ABL_LN = [0.47, 7.38, 7.69, 13.09, 13.52, 7.39, 13.49]

SHIFTS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
ERR_NAIVE = [5.09e-7, 1.39e-4, 8.52e-3, 1.25e0, 1.59e2, 1.02e4, 6.80e6]
ERR_TWOPASS = [3.89e-7, 3.98e-7, 1.18e-7, 3.75e-7, 2.81e-5, 1.51e-7, 7.43e-7]
ERR_WELFORD = [2.07e-7, 1.98e-7, 4.16e-7, 4.73e-6, 5.41e-5, 8.94e-5, 6.71e-3]
NAIVE_NEG_FROM = 1e5

ROWS = [1, 2, 4, 8, 16, 64, 256]
T_SCALAR = [0.250, 1.166, 2.416, 4.916, 9.916, 74.833, 422.208]
T_V7 = [9.291, 6.708, 6.291, 6.209, 6.250, 6.166, 25.250]
GROUP_ROWS = 64

BASE_LABELS = ["ours (V7)", "torch\neager", "torch\ncompile",
               "ExecuTorch\nportable", "ExecuTorch\nXNNPACK"]
BASE_RMS = [24.63, 2.82, 6.04, 1.85, 4.14]
BASE_LN = [13.52, 7.36, 4.45, 5.95, 5.98]


def _bar_group(f, i, n, values_colours, label, label_size=10.5,
               value_fmt="{:.1f}"):
    """One clustered bar group; returns the group's centre x."""
    gw = f.plot_w / n
    x0 = f.pl + i * gw
    k = len(values_colours)
    bw = gw * 0.62 / k
    left = x0 + gw * 0.19
    for j, (v, col) in enumerate(values_colours):
        bx = left + j * bw
        by = f.py(v)
        f.rect(bx, by, bw * 0.86, f.py(0) - by, col, rx=2)
        f.text(bx + bw * 0.43, by - 6, value_fmt.format(v), size=10,
               anchor="middle", weight="600")
        f.reserve(bx, by - 18, bw, f.py(0) - by + 18)
    for li, part in enumerate(label.split("\n")):
        f.text(x0 + gw / 2, f.h - f.pb + 19 + li * 13, part, size=label_size,
               anchor="middle", layer=f.fg)
    return x0 + gw / 2


# --------------------------------------------------------------------------
def fig_ceiling():
    """The central correction, shown rather than asserted: the gap between the
    old constant denominator and the real ceiling is shaded."""
    f = Fig(width=840, height=480, pad_l=86, pad_t=74, pad_b=74)
    f.set_scales(0.055, 300, 0, 320, xlog=True)

    # Regimes as bands, so "cache-resident vs DRAM" is visible, not asserted.
    f.band(0.055, 0.5, "SMSTART-\ncontaminated")
    f.band(0.5, 24, "cache-resident plateau", fill="#eef4fb")
    f.band(24, 300, "DRAM", fill="#fdf1ec")

    f.frame("The bandwidth ceiling is a curve, not a constant",
            "working set (log scale)", "GiB/s  (useful bytes, 1R+1W)",
            "Apple M4, single core. Every “% of peak” before Sprint 6 divided "
            "by one number — the DRAM figure.")
    f.yticks([0, 50, 100, 150, 200, 250, 300])
    f.xticks([0.0625, 0.25, 1, 4, 16, 64, 256],
             ["64 KiB", "256 KiB", "1 MiB", "4 MiB", "16 MiB", "64 MiB", "256 MiB"])

    # Shade the inflation: between the old constant and the true SSVE ceiling,
    # across the cache-resident range.  This region IS the ~2x error.
    pts_top = [(m, s) for m, s in zip(FOOTPRINT_MIB, SSVE) if 0.5 <= m <= 24]
    poly = " ".join(f"{f.px(m):.1f},{f.py(s):.1f}" for m, s in pts_top)
    poly += (f" {f.px(pts_top[-1][0]):.1f},{f.py(DRAM_CONST):.1f}"
             f" {f.px(pts_top[0][0]):.1f},{f.py(DRAM_CONST):.1f}")
    f.bg.append(f'<polygon points="{poly}" fill="{ACCENT}" opacity="0.10"/>')

    y = f.py(DRAM_CONST)
    f.line(f.pl, y, f.w - f.pr, y, stroke=ACCENT, width=1.6, dash="7,4")

    f.polyline(zip(FOOTPRINT_MIB, NEON), BLUE)
    f.markers(list(zip(FOOTPRINT_MIB, NEON)), BLUE)
    f.polyline(zip(FOOTPRINT_MIB, SSVE), VERM)
    f.markers(list(zip(FOOTPRINT_MIB, SSVE)), VERM)
    f.add_legend("NEON (non-streaming)", BLUE)
    f.add_legend("SSVE (streaming) — the kernels' mode", VERM)

    # Annotations in whitespace, tied to the data by leader lines.
    f.callout(["the shaded gap is the error:",
               "115.6 real ceiling vs 58.8 assumed",
               "→ cache-resident % of peak inflated ~2×"],
              tx=f.px(1.0), ty=f.py(78), target=(4, 87), fill=ACCENT,
              size=11, weight="600")
    # Anchored to the right margin so it cannot run off the canvas, and
    # placed BELOW the dashed line where the DRAM band is empty.
    f.text(f.w - f.pr - 4, f.py(DRAM_CONST) + 18,
           "58.8 GiB/s — the single DRAM constant", size=10.5, anchor="end",
           fill=ACCENT, layer=f.fg)
    # Lower-left is the only region both curves leave clear.
    f.callout(["below ~512 KiB this is not bandwidth:",
               "SMSTART is a visible share of a",
               "sub-microsecond pass (Sprint 7b)"],
              tx=f.px(0.058), ty=f.py(34), target=(0.125, 97.66), size=10.5)

    f.draw_legend(prefer=[(f.px(2.2), f.pt + 16)])
    f.footnote(["Bands mark the three regimes the measurements distinguish; "
                "the kernels' reported shapes sit at 16 MiB and 256 MiB."])
    return f.save(OUT / "ceiling_curve.svg")


def fig_ablation():
    """Grouped bars plus the two jumps that actually mattered, called out."""
    n = len(ABL_STAGES)
    f = Fig(width=900, height=470, pad_l=86, pad_t=64, pad_b=76)
    f.set_scales(0, n, 0, 28)
    f.frame("What each optimization bought — 256 MiB (true DRAM)",
            "", "GiB/s  (useful bytes)")
    f.yticks([0, 5, 10, 15, 20, 25])

    for i, stage in enumerate(ABL_STAGES):
        _bar_group(f, i, n, [(ABL_RMS[i], BLUE), (ABL_LN[i], VERM)], stage)

    f.add_legend("RMSNorm — 2 traversals (2R+1W)", BLUE)
    f.add_legend("LayerNorm — 3 traversals (3R+1W)", VERM)

    f.draw_legend(prefer=[(f.pl + 14, f.pt + 10)])
    f.footnote(["V6 (access density, 256 B per column touch) is the lever that "
                "moved the needle; V7 adds SME2 multi-vector accesses.",
                "ZA residency is a measured, explained negative — it lands back "
                "at V0 level despite saving a read."])
    return f.save(OUT / "ablation_dram.svg")


def fig_stability():
    """The danger zone is shaded, so 'catastrophic' is visible."""
    import math
    f = Fig(width=840, height=480, pad_l=92, pad_t=74, pad_b=72)
    f.set_scales(1, 2.2e6, -8.6, 7.6, xlog=True)

    # Everything above relative error 1 means the estimate is worthless.
    f.rect(f.pl, f.pt, f.plot_w, f.py(0) - f.pt, "#fdecea", layer=f.bg)
    # Past this shift the naive estimator returns a negative variance.
    f.rect(f.px(NAIVE_NEG_FROM), f.pt, f.w - f.pr - f.px(NAIVE_NEG_FROM),
           f.h - f.pb - f.pt, ACCENT, opacity=0.06, layer=f.bg)

    f.frame("Where each FP32 variance formulation breaks",
            "input shift   (condition number κ ≈ 1 + shift²)",
            "relative error vs float64 oracle",
            "N = 512. Shifting the data leaves the variance unchanged but "
            "scales the conditioning by ~shift².")
    f.yticks([-8, -6, -4, -2, 0, 2, 4, 6], fmt="1e{:.0f}")
    f.xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
             ["0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6"])

    for vals, col, lab in ((ERR_NAIVE, VERM, "naive  E[x²] − mean²   (1 stage)"),
                           (ERR_WELFORD, ORANGE, "Welford  (1 stage)"),
                           (ERR_TWOPASS, BLUE, "centred two-pass  (ours)")):
        pts = [(s, math.log10(v)) for s, v in zip(SHIFTS, vals)]
        f.polyline(pts, col)
        f.markers(pts, col)
        f.add_legend(lab, col)

    f.line(f.pl, f.py(0), f.w - f.pr, f.py(0), stroke="#c0392b", width=1.2,
           dash="4,4", layer=f.bg)
    f.text(f.pl + 8, f.py(0) - 8, "error ≥ the quantity being estimated",
           size=10.5, fill="#c0392b", layer=f.bg)

    f.callout(["naive returns a NEGATIVE variance here",
               "→ sqrt = NaN. Not inaccurate — unusable."],
              tx=f.px(1.1e3), ty=f.py(6.4), target=(1e5, math.log10(1.02e4)),
              fill=ACCENT, size=11, weight="600")
    # Below every curve: the blue trace bottoms out near 1e-7, so -8.0 is clear.
    f.callout(["two-pass stays below ~3e-5 across the sweep — that",
               "insensitivity is what LayerNorm's third traversal buys"],
              tx=f.px(1.15), ty=f.py(-7.9), target=(1e3, math.log10(3.75e-7)),
              size=10.5, fill=BLUE)

    f.draw_legend(prefer=[(f.px(1.15), f.pt + 12)])
    f.footnote(["RMSNorm never forms a mean, so it never performs this "
                "subtraction: its error stays below ~1.8e-5 across the sweep."])
    return f.save(OUT / "stability.svg")


def fig_crossover():
    """The plateau is bracketed, so 'group granularity' is legible."""
    f = Fig(width=840, height=460, pad_l=86, pad_t=74, pad_b=70)
    f.set_scales(1, 256, 0, 400, xlog=True)

    f.band(1, GROUP_ROWS, "one group — 64 rows (4 × VL)", fill="#eef4fb")
    f.frame("When is the SME kernel worth calling?",
            "rows M   (N = 512, log scale)", "microseconds per call",
            "Crossover at M ≈ 16 — but the cause is group granularity, not "
            "streaming-mode overhead.")
    f.yticks([0, 100, 200, 300, 400])
    f.xticks(ROWS, [str(r) for r in ROWS])

    f.polyline(zip(ROWS, T_SCALAR), VERM)
    f.markers(list(zip(ROWS, T_SCALAR)), VERM)
    f.polyline(zip(ROWS, T_V7), BLUE)
    f.markers(list(zip(ROWS, T_V7)), BLUE)
    f.add_legend("scalar reference", VERM)
    f.add_legend("SME kernel (V7)", BLUE)

    # Bracket the flat region: that flatness IS the finding.
    yb = f.py(46)
    f.line(f.px(1), yb, f.px(GROUP_ROWS), yb, stroke=GREEN, width=1.4)
    for xv in (1, GROUP_ROWS):
        f.line(f.px(xv), yb - 5, f.px(xv), yb + 5, stroke=GREEN, width=1.4)
    f.text((f.px(1) + f.px(GROUP_ROWS)) / 2, yb - 10,
           "V7 flat: a partial group costs a full one", size=10.5,
           anchor="middle", fill=GREEN, weight="600")

    f.callout(["crossover", "M ≈ 16"], tx=f.px(17), ty=f.py(150),
              target=(16, 8.833), size=11, weight="600", fill=INK)
    f.callout(["at M=1 the kernel does 64 rows'",
               "work for 1 row of result"],
              tx=f.px(1.15), ty=f.py(330), target=(1, 7.708), size=10.5)

    f.draw_legend(prefer=[(f.px(2.6), f.pt + 12)])
    f.footnote(["A 9 ns SMSTART cannot explain a 6 µs plateau; a 64-row group "
                "explains it exactly. M=256 is 4 groups → 4× the time."])
    return f.save(OUT / "crossover.svg")


def fig_baselines():
    """The inversion is the point, so it is marked on the figure."""
    n = len(BASE_LABELS)
    f = Fig(width=840, height=470, pad_l=86, pad_t=74, pad_b=76)
    f.set_scales(0, n, 0, 28)
    f.frame("Against general-purpose frameworks — 256 MiB, single-threaded",
            "", "GiB/s  (useful bytes)",
            "PyTorch 2.13.0 / ExecuTorch 1.4.1, each in its native layout. "
            "Every shape verified to compute the same function before timing.")
    f.yticks([0, 5, 10, 15, 20, 25])

    for i, lab in enumerate(BASE_LABELS):
        _bar_group(f, i, n, [(BASE_RMS[i], BLUE), (BASE_LN[i], VERM)], lab)

    f.add_legend("RMSNorm", BLUE)
    f.add_legend("LayerNorm", VERM)

    # Both callouts live in the empty band above the framework bars (which top
    # out at 7.3) and to the right of our tall bar, so neither covers a value.
    f.callout(["ours follows the traffic ratio:",
               "RMSNorm (2 traversals) wins"],
              tx=f.pl + f.plot_w * 0.245, ty=f.py(25.6), size=10.5, fill=BLUE,
              weight="600", leader=False)
    f.callout(["PyTorch inverts it — its CPU RMSNorm decomposes",
               "into mul/pow/sum/div instead of running a fused kernel"],
              tx=f.pl + f.plot_w * 0.245, ty=f.py(19.6), size=10.5, fill=VERM,
              weight="600", leader=False)

    f.draw_legend(prefer=[(f.pl + f.plot_w * 0.80, f.pt + 12)])
    f.footnote(["A fused implementation of the expensive norm beats a "
                "decomposed implementation of the cheap one — the clearest",
                "argument in this report for writing the kernel at all."])
    return f.save(OUT / "traversals_tmp.svg") if False else f.save(
        OUT / "baselines.svg")


def fig_traversals():
    """Structure diagram: boxes sized to their text, plus a traffic bar."""
    f = Fig(width=840, height=330, pad_l=132, pad_t=64, pad_b=40)
    f.text(f.pl - 86, 28, "Why RMSNorm is faster: one fewer reduction stage",
           size=17, weight="700", layer=f.fg)
    f.text(f.pl - 86, 46, "“Two-pass” and “single-pass” name the reduction "
           "stages; what costs bandwidth is the number of input traversals.",
           size=11.5, fill=MUTED, layer=f.fg)

    rows = [
        ("RMSNorm", BLUE,
         ["read x → Σx²", "read x → normalise · γ → write y"],
         "2 traversals · 2R+1W"),
        ("LayerNorm", VERM,
         ["read x → mean", "read x → variance (centred)",
          "read x → normalise · γ · β → write y"],
         "3 traversals · 3R+1W"),
    ]

    size = 10
    pad = 18
    gap = 10
    y = f.pt + 6
    for name, col, steps, summary in rows:
        f.text(f.pl - 14, y + 20, name, size=13, anchor="end", weight="700",
               layer=f.fg)
        x = f.pl
        for s in steps:
            bw = measure(s, size) + pad * 2          # sized to content
            f.rect(x, y, bw, 32, col, rx=4, layer=f.fg)
            f.text(x + bw / 2, y + 21, s, size=size, anchor="middle",
                   fill="white", weight="600", layer=f.fg)
            x += bw + gap
        f.text(x + 4, y + 21, summary, size=11, fill=MUTED, layer=f.fg)
        y += 62

    # Traffic bar: 3 units vs 4 units per element, drawn to scale.
    y += 6
    unit = 42
    f.text(f.pl - 14, y + 14, "traffic", size=11.5, anchor="end", fill=MUTED,
           layer=f.fg)
    for i, (name, col, units) in enumerate((("RMSNorm", BLUE, 3),
                                            ("LayerNorm", VERM, 4))):
        yy = y + i * 26
        for u in range(units):
            f.rect(f.pl + u * (unit + 4), yy, unit, 18, col,
                   opacity=0.35 if u < units - 1 else 0.85, rx=3, layer=f.fg)
        lbl = "R R W" if units == 3 else "R R R W"
        f.text(f.pl + units * (unit + 4) + 8, yy + 14,
               f"{lbl}   =  {units} units", size=10.5, fill=MUTED, layer=f.fg)

    f.text(f.pl, y + 74,
           "4 ÷ 3 = 1.33× more traffic per element for LayerNorm — the "
           "structural source of the gap.", size=11.5, weight="600",
           layer=f.fg)
    return f.save(OUT / "traversals.svg")


if __name__ == "__main__":
    for fn in (fig_ceiling, fig_ablation, fig_stability, fig_crossover,
               fig_baselines, fig_traversals):
        print("wrote", fn())
