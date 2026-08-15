#!/usr/bin/env python3
"""Convert the report figures from SVG to PDF, for the LaTeX paper.

    python3 docs/figures/svg2pdf.py

Reads docs/source/_static/figures/*.svg and writes paper/figures/*.pdf.

Why this exists: Sphinx embeds SVG directly, pdflatex cannot, and the usual
converters (rsvg-convert, Inkscape, cairosvg) are external dependencies this
project deliberately avoids -- svgplot.py is dependency-free for the same
reason.  Both outputs are therefore generated from the same figure source, so a
figure in the paper and the same figure in the HTML report cannot drift apart.

This is NOT a general SVG converter.  It handles exactly the primitives
svgplot.py emits: rect, line, polyline, polygon, circle and text, with fill,
opacity, stroke, stroke-width, stroke-dasharray, font-size/weight/anchor and a
rotate() transform.  Anything else is ignored rather than approximated.

Fonts are the PDF base-14 set, so nothing is embedded: Helvetica and
Helvetica-Bold for text, and Symbol for the glyphs WinAnsiEncoding lacks
(the arrow, approx/greater-equal, minus sign and Greek letters).  Text runs are
split per font so those glyphs render as themselves rather than as ASCII
substitutes.
"""

from __future__ import annotations

import glob
import os
import re

# --- font metrics ----------------------------------------------------------
# Standard Helvetica AFM widths (1/1000 em) for the characters these figures
# use.  Only needed so that text-anchor middle/end lands in the right place.
_HELV = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667,
    "'": 191, '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333,
    '.': 278, '/': 278, ':': 278, ';': 278, '<': 584, '=': 584, '>': 584,
    '?': 556, '@': 1015, '[': 278, '\\': 278, ']': 278, '^': 469, '_': 556,
    '`': 333, '{': 334, '|': 260, '}': 334, '~': 584,
}
for _c in '0123456789':
    _HELV[_c] = 556
for _c, _w in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                  [667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556,
                   833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667,
                   667, 611]):
    _HELV[_c] = _w
for _c, _w in zip('abcdefghijklmnopqrstuvwxyz',
                  [556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222,
                   833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500,
                   500, 500]):
    _HELV[_c] = _w
# WinAnsi extras that svgplot.py emits.
_HELV.update({'—': 1000, '×': 584, 'µ': 556, '“': 333, '”': 333, '²': 333,
              '·': 278, '÷': 584})

# Glyphs WinAnsiEncoding cannot represent -> (Symbol byte, width).
_SYMBOL = {
    '≈': (0xBB, 549), '≥': (0xB3, 549), '→': (0xAE, 987), '−': (0x2D, 549),
    'Σ': (0x53, 603), 'γ': (0x67, 411), 'β': (0x62, 549), 'κ': (0x6B, 549),
}


def _text_runs(s: str):
    """Split text into (font_key, string) runs, where font_key is 'T' (the
    requested text font) or 'S' (Symbol)."""
    runs, cur, cur_sym = [], [], None
    for ch in s:
        sym = ch in _SYMBOL
        if cur_sym is None or sym == cur_sym:
            cur.append(ch); cur_sym = sym
        else:
            runs.append(('S' if cur_sym else 'T', ''.join(cur)))
            cur, cur_sym = [ch], sym
    if cur:
        runs.append(('S' if cur_sym else 'T', ''.join(cur)))
    return runs


def _width(s: str, size: float) -> float:
    total = 0
    for ch in s:
        if ch in _SYMBOL:
            total += _SYMBOL[ch][1]
        else:
            total += _HELV.get(ch, 556)
    return total * size / 1000.0


# --- PDF plumbing ----------------------------------------------------------

def _esc_winansi(s: str) -> bytes:
    out = bytearray()
    for ch in s:
        try:
            b = ch.encode('cp1252')
        except UnicodeEncodeError:
            b = b'?'
        for byte in b:
            if byte in (0x28, 0x29, 0x5C):      # ( ) \
                out += b'\\' + bytes([byte])
            else:
                out.append(byte)
    return bytes(out)


def _esc_symbol(s: str) -> bytes:
    out = bytearray()
    for ch in s:
        byte = _SYMBOL[ch][0]
        if byte in (0x28, 0x29, 0x5C):
            out += b'\\' + bytes([byte])
        else:
            out.append(byte)
    return bytes(out)


def _rgb(colour: str):
    c = (colour or '#000000').strip()
    if c == 'none':
        return None
    if c == 'white':
        return (1.0, 1.0, 1.0)
    if not c.startswith('#'):
        return (0.0, 0.0, 0.0)
    c = c[1:]
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _attrs(tag: str) -> dict:
    return dict(re.findall(r'([\w-]+)\s*=\s*"([^"]*)"', tag))


def convert(svg_path: str, pdf_path: str) -> None:
    src = open(svg_path, encoding='utf-8').read()
    head = _attrs(re.search(r'<svg[^>]*>', src).group(0))
    W, H = float(head['width']), float(head['height'])

    def fy(y):                                  # SVG y-down -> PDF y-up
        return H - float(y)

    ops = []
    gstates = {}                                # opacity -> /GS name

    def use_alpha(a):
        a = round(float(a), 3)
        if a >= 0.999:
            if gstates.get('__cur') != 1.0:
                ops.append('/GS100 gs'); gstates['__cur'] = 1.0
            gstates.setdefault(1.0, 'GS100')
            return
        name = gstates.get(a)
        if name is None:
            name = f'GS{len(gstates)}'
            gstates[a] = name
        ops.append(f'/{name} gs'); gstates['__cur'] = a

    for m in re.finditer(r'<(rect|line|polyline|polygon|circle|text)\b([^>]*?)(/?)>'
                         r'(?:([^<]*)</text>)?', src):
        kind, raw, _, body = m.group(1), m.group(2), m.group(3), m.group(4)
        a = _attrs(raw)
        use_alpha(a.get('opacity', 1.0))

        if kind == 'rect':
            col = _rgb(a.get('fill', '#000000'))
            if col is None:
                continue
            # The full-canvas background rect carries no x/y.
            x, y = float(a.get('x', 0)), float(a.get('y', 0))
            w, h = float(a.get('width', 0)), float(a.get('height', 0))
            if w <= 0 or h <= 0:
                continue
            ops.append(f'{col[0]:.4f} {col[1]:.4f} {col[2]:.4f} rg')
            ops.append(f'{x:.2f} {fy(y + h):.2f} {w:.2f} {h:.2f} re f')
            if a.get('stroke'):
                sc = _rgb(a['stroke'])
                ops.append(f'{sc[0]:.4f} {sc[1]:.4f} {sc[2]:.4f} RG')
                ops.append(f'{float(a.get("stroke-width", 1)):.2f} w')
                ops.append(f'{x:.2f} {fy(y + h):.2f} {w:.2f} {h:.2f} re S')

        elif kind in ('line', 'polyline', 'polygon'):
            sc = _rgb(a.get('stroke', 'none'))
            fc = _rgb(a.get('fill', 'none'))
            ops.append(f'{float(a.get("stroke-width", 1)):.2f} w')
            dash = a.get('stroke-dasharray')
            ops.append(f'[{dash.replace(",", " ")}] 0 d' if dash else '[] 0 d')
            ops.append('1 J' if a.get('stroke-linecap') == 'round' else '0 J')
            if kind == 'line':
                ops.append(f'{float(a["x1"]):.2f} {fy(a["y1"]):.2f} m '
                           f'{float(a["x2"]):.2f} {fy(a["y2"]):.2f} l')
            else:
                pts = [p.split(',') for p in a['points'].split()]
                ops.append(f'{float(pts[0][0]):.2f} {fy(pts[0][1]):.2f} m')
                for px, py in pts[1:]:
                    ops.append(f'{float(px):.2f} {fy(py):.2f} l')
            if kind == 'polygon' and fc is not None:
                ops.append(f'{fc[0]:.4f} {fc[1]:.4f} {fc[2]:.4f} rg')
                ops.append('h f')
            elif sc is not None:
                ops.append(f'{sc[0]:.4f} {sc[1]:.4f} {sc[2]:.4f} RG')
                ops.append('S')

        elif kind == 'circle':
            cx, cy, r = float(a['cx']), fy(a['cy']), float(a['r'])
            k = r * 0.5523
            fc = _rgb(a.get('fill', 'none'))
            path = (f'{cx + r:.2f} {cy:.2f} m '
                    f'{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c '
                    f'{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c '
                    f'{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c '
                    f'{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c')
            ops.append(path)
            if fc is not None:
                ops.append(f'{fc[0]:.4f} {fc[1]:.4f} {fc[2]:.4f} rg')
            sc = _rgb(a.get('stroke', 'none'))
            if sc is not None:
                ops.append(f'{sc[0]:.4f} {sc[1]:.4f} {sc[2]:.4f} RG')
                ops.append(f'{float(a.get("stroke-width", 1)):.2f} w')
                ops.append('B' if fc is not None else 'S')
            elif fc is not None:
                ops.append('f')

        elif kind == 'text':
            txt = (body or '').replace('&amp;', '&').replace('&lt;', '<') \
                              .replace('&gt;', '>')
            if not txt:
                continue
            size = float(a.get('font-size', 12))
            bold = float(a.get('font-weight', 400) or 400) >= 600
            col = _rgb(a.get('fill', '#000000')) or (0, 0, 0)
            x0, y0 = float(a['x']), fy(a['y'])
            w = _width(txt, size)
            anchor = a.get('text-anchor', 'start')
            shift = -w / 2.0 if anchor == 'middle' else (-w if anchor == 'end' else 0.0)

            rot = re.match(r'rotate\(\s*(-?[\d.]+)', a.get('transform', ''))
            ops.append('q')
            if rot:
                import math
                ang = float(rot.group(1))
                c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                # svgplot always rotates about the text's own anchor point, so
                # translate to THAT point (not the anchor-adjusted one), rotate,
                # then apply the anchor shift inside the rotated frame.  SVG
                # rotates in a y-down frame and PDF in a y-up one, hence -s.
                ops.append(f'1 0 0 1 {x0:.2f} {y0:.2f} cm')
                ops.append(f'{c:.5f} {-s:.5f} {s:.5f} {c:.5f} 0 0 cm')
                x, y = shift, 0.0
            else:
                x, y = x0 + shift, y0
            ops.append(f'{col[0]:.4f} {col[1]:.4f} {col[2]:.4f} rg')
            ops.append('BT')
            cx = x
            for key, run in _text_runs(txt):
                if key == 'S':
                    ops.append(f'/F3 {size:.2f} Tf')
                    payload = _esc_symbol(run)
                else:
                    ops.append(f'/{"F2" if bold else "F1"} {size:.2f} Tf')
                    payload = _esc_winansi(run)
                ops.append(f'1 0 0 1 {cx:.2f} {y:.2f} Tm '
                           f'({payload.decode("latin-1")}) Tj')
                cx += _width(run, size)
            ops.append('ET')
            ops.append('Q')

    content = ('\n'.join(ops)).encode('latin-1')

    alphas = {v: k for k, v in gstates.items() if isinstance(k, float)}
    objs = []
    objs.append(b'<</Type/Catalog/Pages 2 0 R>>')
    objs.append(b'<</Type/Pages/Kids[3 0 R]/Count 1>>')

    gs_entries = ''.join(f'/{name} {8 + i} 0 R'
                         for i, name in enumerate(sorted(alphas)))
    gs_entries += '/GS100 %d 0 R' % (8 + len(alphas))
    page = (f'<</Type/Page/Parent 2 0 R/MediaBox[0 0 {W:.2f} {H:.2f}]'
            f'/Resources<</Font<</F1 5 0 R/F2 6 0 R/F3 7 0 R>>'
            f'/ExtGState<<{gs_entries}>>>>/Contents 4 0 R>>')
    objs.append(page.encode('latin-1'))
    objs.append(b'<</Length %d>>stream\n' % len(content) + content + b'\nendstream')
    objs.append(b'<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>')
    objs.append(b'<</Type/Font/Subtype/Type1/BaseFont/Helvetica-Bold/Encoding/WinAnsiEncoding>>')
    objs.append(b'<</Type/Font/Subtype/Type1/BaseFont/Symbol>>')
    for name in sorted(alphas):
        objs.append(b'<</Type/ExtGState/ca %.3f/CA %.3f>>'
                    % (alphas[name], alphas[name]))
    objs.append(b'<</Type/ExtGState/ca 1/CA 1>>')

    out = bytearray(b'%PDF-1.4\n')
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b'%d 0 obj' % i + body + b'endobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n' % (len(objs) + 1)
    out += b'0000000000 65535 f \n'
    for off in offsets:
        out += b'%010d 00000 n \n' % off
    out += (b'trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n'
            % (len(objs) + 1, xref))

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    open(pdf_path, 'wb').write(bytes(out))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    src_dir = os.path.join(root, 'docs', 'source', '_static', 'figures')
    out_dir = os.path.join(root, 'paper', 'figures')
    for svg in sorted(glob.glob(os.path.join(src_dir, '*.svg'))):
        pdf = os.path.join(out_dir,
                           os.path.basename(svg)[:-4] + '.pdf')
        convert(svg, pdf)
        print(f'{os.path.basename(svg)} -> {os.path.relpath(pdf, root)} '
              f'({os.path.getsize(pdf)} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
