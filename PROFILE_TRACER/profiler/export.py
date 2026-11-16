"""Polygon -> SVG / DXF.

Units: polygons arrive in source-image pixels (x right, y down). Pass
``mm_per_px`` to get real-world millimetres. SVG keeps y-down (SVG native);
DXF flips to y-up (CAD native) so the part isn't mirrored when imported.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np


def _prep(polygon: Sequence[Sequence[float]], mm_per_px: float | None, flip_y: bool, origin: str):
    p = np.asarray(polygon, float)
    if len(p) < 3:
        raise ValueError("Polygon needs at least 3 points")
    if mm_per_px:
        p = p * mm_per_px
    if flip_y:
        p[:, 1] = -p[:, 1]
    if origin == "min":  # move bounding box corner to (0, 0)
        p = p - p.min(axis=0)
    return p


def mm_per_px_from_width(polygon, width_mm: float | None = None, height_mm: float | None = None):
    """Scale such that the outline's bounding box has the given real size."""
    p = np.asarray(polygon, float)
    span = p.max(axis=0) - p.min(axis=0)
    if width_mm:
        return width_mm / span[0]
    if height_mm:
        return height_mm / span[1]
    return None


def to_svg(polygon, mm_per_px: float | None = None, *, image_size=None, stroke="#000",
           stroke_width=None, origin="min") -> str:
    """Closed path SVG. With mm_per_px, the document is sized in mm so it imports at
    true scale into Inkscape/Fusion/etc. Without it, units are image pixels and, if
    ``image_size`` is given with origin='image', the viewBox matches the photo."""
    p = _prep(polygon, mm_per_px, flip_y=False, origin=origin)
    if origin == "image" and image_size is not None:
        w, h = image_size
        if mm_per_px:
            w, h = w * mm_per_px, h * mm_per_px
        minx, miny = 0.0, 0.0
    else:
        minx, miny = p.min(axis=0)
        w, h = p.max(axis=0) - p.min(axis=0)
    unit = "mm" if mm_per_px else ""
    sw = stroke_width if stroke_width is not None else max(w, h) / 400
    d = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in p) + " Z"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.4f}{unit}" height="{h:.4f}{unit}" '
        f'viewBox="{minx:.4f} {miny:.4f} {w:.4f} {h:.4f}">\n'
        f'  <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw:.4f}" '
        'stroke-linejoin="round"/>\n'
        '</svg>\n'
    )


def to_dxf(polygon, mm_per_px: float | None = None, *, layer="PROFILE", origin="min") -> str:
    """Single closed LWPOLYLINE. Units set to mm when scaled, else unitless (px)."""
    import ezdxf
    from ezdxf import units

    p = _prep(polygon, mm_per_px, flip_y=True, origin=origin)
    doc = ezdxf.new("R2010")
    doc.units = units.MM if mm_per_px else units.UNITLESS
    doc.layers.add(layer, color=7)
    doc.modelspace().add_lwpolyline([tuple(map(float, pt)) for pt in p], close=True,
                                    dxfattribs={"layer": layer})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue()
