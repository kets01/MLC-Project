"""A very small dependency-free SVG plotting helper.

Why not matplotlib: CLAUDE.md keeps the project's dependency surface minimal
(Catch2 is the only third-party dependency in the build), and the figures for
this report are simple — line plots on a log axis and grouped bars.  A ~200-line
generator that runs under plain `python3` keeps the figures reproducible from a
clean checkout with no environment to recreate, which matters more here than
plotting flexibility.

SVG (rather than PNG) because Sphinx embeds it directly, it stays sharp at any
zoom, and it diffs as text.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# Palette: colour-blind safe, and each series stays distinguishable in
# greyscale by ordering light-to-dark.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GRID = "#d9d9d9"
AXIS = "#333333"
TEXT = "#222222"
MUTED = "#777777"


class Fig:
    def __init__(self, width=760, height=430, pad_l=78, pad_r=26,
                 pad_t=52, pad_b=62):
        self.w, self.h = width, height
        self.pl, self.pr, self.pt, self.pb = pad_l, pad_r, pad_t, pad_b
        self.parts: list[str] = []
        self.legend: list[tuple[str, str]] = []

    # -- coordinate space ---------------------------------------------------
    def set_scales(self, xmin, xmax, ymin, ymax, xlog=False):
        self.xlog = xlog
        self.xmin = math.log10(xmin) if xlog else xmin
        self.xmax = math.log10(xmax) if xlog else xmax
        self.ymin, self.ymax = ymin, ymax

    def px(self, x):
        v = math.log10(x) if self.xlog else x
        f = (v - self.xmin) / (self.xmax - self.xmin)
        return self.pl + f * (self.w - self.pl - self.pr)

    def py(self, y):
        f = (y - self.ymin) / (self.ymax - self.ymin)
        return self.h - self.pb - f * (self.h - self.pt - self.pb)

    # -- primitives ---------------------------------------------------------
    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", fill=TEXT, weight="normal"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" '
                 f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
                 f'font-weight="{weight}">{_esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=AXIS, width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, opacity=1.0):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" '
                 f'height="{max(h,0):.1f}" fill="{fill}" opacity="{opacity}"/>')

    def polyline(self, pts: Iterable[tuple[float, float]], stroke, width=2.2,
                 dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in pts)
        self.add(f'<polyline points="{s}" fill="none" stroke="{stroke}" '
                 f'stroke-width="{width}" stroke-linejoin="round"{d}/>')

    def markers(self, pts, fill, r=3.4):
        for x, y in pts:
            self.add(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" '
                     f'r="{r}" fill="{fill}"/>')

    # -- chrome -------------------------------------------------------------
    def frame(self, title, xlabel, ylabel, subtitle=None):
        self.text(self.pl, 26, title, size=16, weight="bold")
        if subtitle:
            self.text(self.pl, 43, subtitle, size=11.5, fill=MUTED)
        self.line(self.pl, self.h - self.pb, self.w - self.pr, self.h - self.pb)
        self.line(self.pl, self.pt, self.pl, self.h - self.pb)
        self.text((self.pl + self.w - self.pr) / 2, self.h - 16, xlabel,
                  size=12.5, anchor="middle")
        self.add(f'<text x="18" y="{(self.pt + self.h - self.pb)/2:.1f}" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="12.5" '
                 f'text-anchor="middle" fill="{TEXT}" '
                 f'transform="rotate(-90 18 {(self.pt + self.h - self.pb)/2:.1f})">'
                 f'{_esc(ylabel)}</text>')

    def yticks(self, values, fmt="{:.0f}"):
        for v in values:
            y = self.py(v)
            self.line(self.pl, y, self.w - self.pr, y, stroke=GRID)
            self.text(self.pl - 8, y + 4, fmt.format(v), size=11.5, anchor="end")

    def xticks(self, values, labels=None, rotate=False):
        labels = labels or [str(v) for v in values]
        for v, lab in zip(values, labels):
            x = self.px(v)
            self.line(x, self.h - self.pb, x, self.h - self.pb + 5)
            if rotate:
                self.add(f'<text x="{x:.1f}" y="{self.h - self.pb + 18:.1f}" '
                         f'font-family="Helvetica,Arial,sans-serif" font-size="11" '
                         f'text-anchor="end" fill="{TEXT}" '
                         f'transform="rotate(-40 {x:.1f} {self.h - self.pb + 18:.1f})">'
                         f'{_esc(lab)}</text>')
            else:
                self.text(x, self.h - self.pb + 18, lab, size=11, anchor="middle")

    def add_legend(self, label, colour):
        self.legend.append((label, colour))

    def draw_legend(self, x=None, y=None):
        x = self.pl + 12 if x is None else x
        y = self.pt + 6 if y is None else y
        for i, (lab, col) in enumerate(self.legend):
            yy = y + i * 18
            self.rect(x, yy - 9, 22, 4, col)
            self.text(x + 30, yy - 4, lab, size=12)

    def save(self, path):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
               f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
               f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n'
               f'{body}\n</svg>\n')
        with open(path, "w") as f:
            f.write(svg)
        return path


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def nice_ymax(v, steps=(1, 2, 2.5, 5, 10)):
    """Round an axis maximum up to a readable value."""
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for s in steps:
        if v / mag <= s:
            return s * mag
    return 10 * mag
