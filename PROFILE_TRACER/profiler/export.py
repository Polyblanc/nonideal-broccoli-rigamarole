"""Paths -> SVG / DXF.

A "path" is {"points": [[x, y], ...], "closed": bool}. Every function also
accepts a single bare polygon ([[x, y], ...]), which is treated as one closed path.

Units: points arrive in source-image pixels (x right, y down). Pass
``mm_per_px`` to get real-world millimetres. SVG keeps y-down (SVG native);
DXF flips to y-up (CAD native) so the part isn't mirrored when imported.

``curves=True`` exports smooth curves through the points instead of straight
segments: cubic Béziers in SVG, fit-point SPLINEs in DXF.
"""

from __future__ import annotations

import io

import numpy as np


def normalize_paths(paths) -> list[dict]:
    if paths is None or len(paths) == 0:
        raise ValueError("Nothing to export")
    first = paths[0]
    if isinstance(first, dict):
        out = [{"points": np.asarray(p["points"], float), "closed": bool(p.get("closed", True))}
               for p in paths]
    else:  # bare polygon
        out = [{"points": np.asarray(paths, float), "closed": True}]
    out = [p for p in out if len(p["points"]) >= (3 if p["closed"] else 2)]
    if not out:
        raise ValueError("Nothing to export")
    return out


def _all_points(paths):
    return np.concatenate([p["points"] for p in paths])


def mm_per_px_from_width(paths, width_mm: float | None = None, height_mm: float | None = None):
    """Scale such that the bounding box of everything has the given real size."""
    pts = _all_points(normalize_paths(paths))
    span = pts.max(axis=0) - pts.min(axis=0)
    if width_mm:
        return width_mm / span[0]
    if height_mm:
        return height_mm / span[1]
    return None


def mm_per_px_from_points(a, b, distance_mm: float) -> float:
    """Scale from two clicked points a known real distance apart."""
    d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if d <= 0:
        raise ValueError("The two scale points are on top of each other")
    return distance_mm / d


def _prep(paths, mm_per_px, flip_y, origin):
    paths = normalize_paths(paths)
    pts = [p["points"].copy() for p in paths]
    k = mm_per_px or 1.0
    for q in pts:
        q *= k
        if flip_y:
            q[:, 1] = -q[:, 1]
    if origin == "min":  # move the bounding box corner to (0, 0)
        mn = np.concatenate(pts).min(axis=0)
        pts = [q - mn for q in pts]
    return [{"points": q, "closed": p["closed"]} for q, p in zip(pts, paths)]


def _bezier_segments(p: np.ndarray, closed: bool):
    """Catmull-Rom through the points, as cubic Bézier control points."""
    n = len(p)
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = p[(i - 1) % n] if closed or i > 0 else p[i]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if closed or i + 2 < n else p2
        out.append((p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2))
    return out


def _svg_d(p: np.ndarray, closed: bool, curves: bool) -> str:
    f = lambda v: f"{v[0]:.4f},{v[1]:.4f}"
    if curves and len(p) >= 3:
        segs = _bezier_segments(p, closed)
        d = f"M {f(p[0])} " + " ".join(f"C {f(a)} {f(b)} {f(c)}" for a, b, c in segs)
    else:
        d = "M " + " L ".join(f(v) for v in p)
    return d + (" Z" if closed else "")


def to_svg(paths, mm_per_px: float | None = None, *, image_size=None, stroke="#000",
           stroke_width=None, origin="min", curves=False) -> str:
    """One <path> per piece. With mm_per_px, the document is sized in mm so it
    imports at true scale into Inkscape/Fusion/etc. Without it, units are image
    pixels; with origin='image' and image_size the viewBox matches the photo."""
    ps = _prep(paths, mm_per_px, flip_y=False, origin=origin)
    allp = np.concatenate([p["points"] for p in ps])
    if origin == "image" and image_size is not None:
        w, h = image_size
        if mm_per_px:
            w, h = w * mm_per_px, h * mm_per_px
        minx, miny = 0.0, 0.0
    else:
        minx, miny = allp.min(axis=0)
        w, h = np.maximum(allp.max(axis=0) - allp.min(axis=0), 1e-6)
    unit = "mm" if mm_per_px else ""
    sw = stroke_width if stroke_width is not None else max(w, h) / 400
    body = "\n".join(
        f'  <path d="{_svg_d(p["points"], p["closed"], curves)}" fill="none" stroke="{stroke}" '
        f'stroke-width="{sw:.4f}" stroke-linejoin="round" stroke-linecap="round"/>'
        for p in ps)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.4f}{unit}" height="{h:.4f}{unit}" '
        f'viewBox="{minx:.4f} {miny:.4f} {w:.4f} {h:.4f}">\n{body}\n</svg>\n'
    )


def to_dxf(paths, mm_per_px: float | None = None, *, origin="min", curves=False) -> str:
    """Closed outlines go on layer PROFILE, open centerlines on layer CENTERLINE.
    Units are mm when scaled, else unitless (pixels)."""
    import ezdxf
    from ezdxf import units

    ps = _prep(paths, mm_per_px, flip_y=True, origin=origin)
    doc = ezdxf.new("R2010")
    doc.units = units.MM if mm_per_px else units.UNITLESS
    doc.layers.add("PROFILE", color=7)
    doc.layers.add("CENTERLINE", color=1)
    msp = doc.modelspace()
    for p in ps:
        layer = "PROFILE" if p["closed"] else "CENTERLINE"
        pts = [tuple(map(float, v)) for v in p["points"]]
        if curves and len(pts) >= 3:
            fit = pts + [pts[0]] if p["closed"] else pts
            msp.add_spline(fit_points=[(x, y, 0.0) for x, y in fit], dxfattribs={"layer": layer})
        else:
            msp.add_lwpolyline(pts, close=p["closed"], dxfattribs={"layer": layer})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue()
