"""A small dependency-free SVG plotting helper.

Why not matplotlib: CLAUDE.md keeps the project's dependency surface minimal
(Catch2 is the only third-party dependency in the build), and these figures are
simple.  A self-contained generator that runs under plain `python3` keeps them
reproducible from a clean checkout with no environment to recreate.

SVG rather than PNG: Sphinx embeds it directly, it stays sharp at any zoom, and
it diffs as text.

Design notes for the primitives here — each exists because a first draft of the
figures had the corresponding defect:

* `measure()` estimates rendered text width, so boxes can be sized to their
  contents instead of clipping them.
* `callout()` puts annotation text in whitespace and draws a leader line to the
  thing it describes, instead of dropping text on top of a curve.
* `band()` shades a region of the x-axis, which turns "regimes" from something
  the caption asserts into something the reader can see.
* `reserve()` / `free_box()` track occupied rectangles so legends and labels can
  be placed where the data is not.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# Palette: colour-blind safe (blue/vermillion/green/orange from Okabe-Ito),
# and ordered so the series stay distinguishable in greyscale.
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#7B52AB"
PALETTE = [BLUE, VERM, GREEN, ORANGE, PURPLE]

INK = "#1a1a1a"
MUTED = "#6b7280"
GRID = "#e6e6e6"
BAND = "#f4f6f8"
ACCENT = "#B03060"
FONT = "Inter,Helvetica Neue,Helvetica,Arial,sans-serif"

# Average glyph width as a fraction of font size, for this font stack.
_W = 0.53


def measure(text: str, size: float) -> float:
    """Approximate rendered width in px.  Deliberately slightly generous."""
    narrow = sum(1 for c in text if c in "iljI.,:;'|! ")
    wide = sum(1 for c in text if c in "mwMW—")
    n = len(text)
    return size * (_W * (n - narrow - wide) + 0.30 * narrow + 0.85 * wide)


class Fig:
    def __init__(self, width=820, height=470, pad_l=82, pad_r=30,
                 pad_t=62, pad_b=66):
        self.w, self.h = width, height
        self.pl, self.pr, self.pt, self.pb = pad_l, pad_r, pad_t, pad_b
        self.bg: list[str] = []      # drawn first (bands, grid)
        self.parts: list[str] = []   # data
        self.fg: list[str] = []      # drawn last (annotations, legend)
        self.legend: list[tuple[str, str]] = []
        self._taken: list[tuple[float, float, float, float]] = []
        self.xlog = False

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

    @property
    def plot_w(self):
        return self.w - self.pl - self.pr

    # -- occupancy ----------------------------------------------------------
    def reserve(self, x, y, w, h):
        self._taken.append((x, y, x + w, y + h))

    def _hits(self, x, y, w, h, margin=6):
        for (a, b, c, d) in self._taken:
            if not (x + w + margin < a or x > c + margin
                    or y + h + margin < b or y > d + margin):
                return True
        return False

    def free_box(self, w, h, prefer: Sequence[tuple[float, float]]):
        """First preferred position whose rectangle is unoccupied."""
        for (x, y) in prefer:
            if not self._hits(x, y, w, h):
                return x, y
        return prefer[-1]

    # -- primitives ---------------------------------------------------------
    def _t(self, layer, s):
        layer.append(s)

    def text(self, x, y, s, size=13, anchor="start", fill=INK, weight="400",
             layer=None, italic=False):
        layer = self.parts if layer is None else layer
        st = ' font-style="italic"' if italic else ""
        self._t(layer, f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
                       f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
                       f'font-weight="{weight}"{st}>{_esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0, dash=None, layer=None):
        layer = self.parts if layer is None else layer
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._t(layer, f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                       f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"'
                       f'{d} stroke-linecap="round"/>')

    def rect(self, x, y, w, h, fill, opacity=1.0, rx=0, layer=None,
             stroke=None, stroke_width=1.0):
        layer = self.parts if layer is None else layer
        sk = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
        self._t(layer, f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" '
                       f'height="{max(h,0):.1f}" rx="{rx}" fill="{fill}" '
                       f'opacity="{opacity}"{sk}/>')

    def polyline(self, pts: Iterable[tuple[float, float]], stroke, width=2.6,
                 dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in pts)
        self._t(self.parts, f'<polyline points="{s}" fill="none" '
                            f'stroke="{stroke}" stroke-width="{width}" '
                            f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def markers(self, pts, fill, r=3.8):
        for x, y in pts:
            self._t(self.parts, f'<circle cx="{self.px(x):.1f}" '
                                f'cy="{self.py(y):.1f}" r="{r}" fill="white" '
                                f'stroke="{fill}" stroke-width="2.2"/>')

    def band(self, x0, x1, label=None, fill=BAND):
        """Shade an x-range; label sits at the top of the plot area."""
        a, b = self.px(x0), self.px(x1)
        self.rect(a, self.pt, b - a, self.h - self.pb - self.pt, fill,
                  layer=self.bg)
        if label:
            self.text((a + b) / 2, self.pt - 8, label, size=11,
                      anchor="middle", fill=MUTED, layer=self.bg)

    def callout(self, lines, tx, ty, target=None, size=11, fill=MUTED,
                anchor="start", weight="400", leader=True):
        """Text block in whitespace, with an optional leader line to `target`
        given in DATA coordinates."""
        w = max(measure(s, size) for s in lines)
        h = len(lines) * (size + 3)
        if target is not None and leader:
            gx, gy = self.px(target[0]), self.py(target[1])
            ax = tx + (w if anchor == "start" else 0)
            ax = tx if anchor == "start" else tx
            self.line(ax, ty - size + 2, gx, gy, stroke=MUTED, width=0.9,
                      dash="2,3", layer=self.fg)
            self._t(self.fg, f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="2.6" '
                             f'fill="{MUTED}"/>')
        for i, s in enumerate(lines):
            self.text(tx, ty + i * (size + 3), s, size=size, fill=fill,
                      anchor=anchor, weight=weight, layer=self.fg)
        self.reserve(tx if anchor == "start" else tx - w, ty - size, w, h)

    # -- chrome -------------------------------------------------------------
    def frame(self, title, xlabel="", ylabel="", subtitle=None):
        self.text(self.pl - 46, 28, title, size=17, weight="700", layer=self.fg)
        if subtitle:
            self.text(self.pl - 46, 46, subtitle, size=11.5, fill=MUTED,
                      layer=self.fg)
        self.line(self.pl, self.h - self.pb, self.w - self.pr, self.h - self.pb,
                  stroke="#9aa0a6", layer=self.fg)
        if xlabel:
            # Positioned relative to the axis, not the canvas bottom, so it
            # cannot collide with a footnote (it did in the first draft).
            self.text((self.pl + self.w - self.pr) / 2, self.h - self.pb + 44,
                      xlabel, size=12.5, anchor="middle", layer=self.fg)
        if ylabel:
            cy = (self.pt + self.h - self.pb) / 2
            self._t(self.fg, f'<text x="20" y="{cy:.1f}" font-family="{FONT}" '
                             f'font-size="12.5" text-anchor="middle" fill="{INK}" '
                             f'transform="rotate(-90 20 {cy:.1f})">'
                             f'{_esc(ylabel)}</text>')

    def yticks(self, values, fmt="{:.0f}"):
        for v in values:
            y = self.py(v)
            self.line(self.pl, y, self.w - self.pr, y, stroke=GRID,
                      layer=self.bg)
            self.text(self.pl - 9, y + 4, fmt.format(v), size=11.5,
                      anchor="end", fill=MUTED, layer=self.bg)

    def xticks(self, values, labels=None):
        labels = labels or [str(v) for v in values]
        for v, lab in zip(values, labels):
            x = self.px(v)
            self.line(x, self.h - self.pb, x, self.h - self.pb + 5,
                      stroke="#9aa0a6", layer=self.fg)
            self.text(x, self.h - self.pb + 19, lab, size=11,
                      anchor="middle", layer=self.fg)

    def add_legend(self, label, colour):
        self.legend.append((label, colour))

    def draw_legend(self, prefer=None, size=12):
        if not self.legend:
            return
        w = max(measure(l, size) for l, _ in self.legend) + 34
        h = len(self.legend) * 19 + 10
        prefer = prefer or [
            (self.pl + 14, self.pt + 12),
            (self.w - self.pr - w - 14, self.pt + 12),
            (self.w - self.pr - w - 14, self.h - self.pb - h - 14),
            (self.pl + 14, self.h - self.pb - h - 14),
        ]
        x, y = self.free_box(w, h, prefer)
        self.rect(x - 8, y - 6, w, h, "white", opacity=0.88, rx=4,
                  layer=self.fg, stroke="#e2e5e9")
        for i, (lab, col) in enumerate(self.legend):
            yy = y + 12 + i * 19
            self.rect(x, yy - 8, 20, 4, col, rx=2, layer=self.fg)
            self.text(x + 28, yy - 3, lab, size=size, layer=self.fg)
        self.reserve(x - 8, y - 6, w, h)

    def footnote(self, lines, size=10.5):
        y = self.h - 8 - (len(lines) - 1) * 13
        for i, s in enumerate(lines):
            self.text(self.pl - 46, y + i * 13, s, size=size, fill=MUTED,
                      layer=self.fg)

    def save(self, path):
        body = "\n".join(self.bg + self.parts + self.fg)
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
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for s in steps:
        if v / mag <= s:
            return s * mag
    return 10 * mag
