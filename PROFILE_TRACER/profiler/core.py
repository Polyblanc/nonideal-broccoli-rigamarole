"""Image -> outline(s) or centerline(s).

Pipeline
--------
1. Load image (alpha channel is used directly if it carries a real cutout).
2. Downscale to a working resolution ("detail").
3. Build several candidate foreground masks with different classic methods
   (Otsu on gray, colour distance from background, Canny edges + fill, GrabCut,
   and for thin lines an adaptive threshold).
4. Clean each mask: apply lasso constraints and click hints, pick the right
   piece(s), drop specks/spurs, and round off pixel jaggies.
5. Score every candidate (edge alignment, consensus with the others, border
   contact, lasso fit, agreement with clicks) and pick the best.
6. Convert to vector paths:
   - shape="outline": outer contour of each piece -> smoothed, simplified,
     guaranteed simple (no self-intersections) closed polygon.
   - shape="centerline": skeleton of the shape -> open polylines through the
     middle of thin features (wires, springs, drawn lines), with short
     side-branches pruned.

All coordinates returned are in *source image pixels*, x right, y down.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

METHODS = ("otsu", "color", "edges", "grabcut")
THIN_METHODS = ("lines", "otsu", "adaptive")   # used for centerlines
LASSO_MODES = ("loose", "tight")
SHAPES = ("outline", "centerline")
PIECES = ("largest", "all")

Points = Sequence[Sequence[float]]


# --------------------------------------------------------------------------
# Parameters / results
# --------------------------------------------------------------------------

@dataclass
class TraceParams:
    method: str = "auto"            # "auto" or one of METHODS / THIN_METHODS
    shape: str = "outline"          # "outline": closed profile; "centerline": open middle lines
    pieces: str = "largest"         # "largest": one piece; "all": every piece found
    include: Points | None = None   # clicks ON the object, [[x, y], ...] source px
    exclude: Points | None = None   # clicks on BACKGROUND / things to ignore
    lasso: Points | None = None     # [[x, y], ...] in source px
    lasso_mode: str = "loose"       # "loose": object is somewhere inside the lasso
                                    # "tight": lasso is a rough hand trace of the outline
    snap_px: float | None = None    # tight mode: how far (source px) the result may
                                    # deviate from the hand trace. None = auto
    work_size: int = 1000           # longest side used for processing; 0 = full size
    smooth: float = 1.5             # smoothing sigma, in working px (0 = off)
    min_feature: float = 3.0        # specks/spurs/notches smaller than this (working px) are removed
    epsilon: float = 0.75           # polyline simplification tolerance, source px
    invert: bool | None = None      # force polarity for otsu (None = auto)


@dataclass
class Candidate:
    method: str
    score: float
    metrics: dict
    paths: list[dict] = field(default_factory=list)       # [{"points": [[x,y],...], "closed": bool}]
    polygon: list[list[float]] = field(default_factory=list)  # first closed path (convenience)


@dataclass
class TraceResult:
    polygon: list[list[float]]      # first closed path of the chosen candidate (outline mode)
    paths: list[dict]               # all paths of the chosen candidate
    method: str
    width: int
    height: int
    candidates: list[Candidate]
    warnings: list[str]
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_image(src: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Return BGR or BGRA uint8 image."""
    if isinstance(src, np.ndarray):
        img = src
    else:
        data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not decode image")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# --------------------------------------------------------------------------
# Mask helpers
# --------------------------------------------------------------------------

def _poly_mask(shape, poly: np.ndarray) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return m


def _disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray, max_area: float | None = None) -> np.ndarray:
    """Fill enclosed holes (all of them, or only those smaller than max_area)."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = mask.copy()
    if hier is None:
        return out
    for c, h in zip(cnts, hier[0]):
        is_hole = h[3] >= 0
        if is_hole and (max_area is None or cv2.contourArea(c) < max_area):
            cv2.drawContours(out, [c], -1, 255, cv2.FILLED)
    return out


def _component_at(labels: np.ndarray, pt, radius: float) -> int:
    """Label of the component under pt, or the nearest one within radius (0 = none)."""
    h, w = labels.shape
    x, y = int(round(pt[0])), int(round(pt[1]))
    if not (0 <= x < w and 0 <= y < h):
        return 0
    if labels[y, x]:
        return int(labels[y, x])
    r = max(1, int(radius))
    y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
    nz = np.argwhere(labels[y0:y1, x0:x1] > 0)
    if not len(nz):
        return 0
    d = (nz[:, 0] + y0 - y) ** 2 + (nz[:, 1] + x0 - x) ** 2
    yy, xx = nz[int(np.argmin(d))]
    return int(labels[yy + y0, xx + x0]) if d.min() <= r * r else 0


def _select(mask, includes, excludes, pieces, min_area):
    """Choose which connected pieces to keep.

    - Pieces with an exclude click (and no include click) are dropped.
    - With include clicks: keep the pieces that were clicked (or are nearest).
    - Otherwise: the largest piece, or every piece bigger than min_area.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    reach = 0.03 * max(h, w)
    areas = stats[:, cv2.CC_STAT_AREA]
    inc = {_component_at(labels, p, reach) for p in includes} - {0}
    exc = {_component_at(labels, p, 2) for p in excludes} - {0} - inc
    keep = [i for i in range(1, n) if areas[i] >= min_area and i not in exc]
    if inc:
        chosen = set(inc) | (set(keep) if pieces == "all" else set())
    elif not keep:
        return np.zeros_like(mask)
    elif pieces == "all":
        chosen = set(keep)
    else:
        chosen = {max(keep, key=lambda i: areas[i])}
    return np.isin(labels, list(chosen)).astype(np.uint8) * 255


def _clean(mask, min_feature, includes, excludes, pieces):
    """Solid outlines: fill holes, keep chosen pieces, drop spurs, de-jag."""
    h, w = mask.shape
    min_area = max(0.0005 * h * w, (3 * min_feature) ** 2)
    mask = (mask > 127).astype(np.uint8) * 255
    mask = _fill_holes(mask)
    mask = _select(mask, includes, excludes, pieces, min_area)
    if min_feature > 0 and mask.any():
        # Kernel diameter strictly greater than min_feature, so features up to
        # min_feature wide are removed (edge/grabcut candidates thicken hairlines).
        k = _disk(np.ceil(min_feature / 2 + 0.01))
        # Opening kills thin spurs/islands bridged to the body; closing fills
        # thin notches/cracks. Re-select afterwards since opening can split a neck.
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if opened.any():
            mask = _select(opened, includes, excludes, pieces, min_area)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = _fill_holes(mask)
    # Round off staircase jaggies: blur then re-threshold.
    if mask.any():
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        mask = (mask > 127).astype(np.uint8) * 255
        mask = _select(mask, includes, excludes, pieces, min_area)
    return mask


def _clean_thin(mask, min_feature, includes, excludes, pieces):
    """Thin lines: never erode (that would delete the wire). Bridge 1px gaps,
    fill pinholes, drop specks."""
    h, w = mask.shape
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk(1))
    mask = _fill_holes(mask, max_area=(2 * max(min_feature, 1.5)) ** 2)
    min_area = max(0.00005 * h * w, (2 * min_feature) ** 2)
    return _select(mask, includes, excludes, pieces, min_area)


# --------------------------------------------------------------------------
# Candidate generators
# --------------------------------------------------------------------------

def _bg_sample(shape, lasso_mask, excludes) -> np.ndarray:
    """Boolean mask of pixels we believe are background."""
    h, w = shape[:2]
    sample = None
    if lasso_mask is not None:
        ring = cv2.dilate(lasso_mask, _disk(max(4, 0.03 * max(h, w)))) > 0
        sample = ring & (lasso_mask == 0)
        if sample.sum() <= 50:
            sample = None
    if sample is None:
        b = max(2, int(0.02 * max(h, w)))
        sample = np.zeros((h, w), bool)
        sample[:b, :] = sample[-b:, :] = sample[:, :b] = sample[:, -b:] = True
    if excludes:
        m = np.zeros((h, w), np.uint8)
        for x, y in excludes:
            cv2.circle(m, (int(x), int(y)), max(3, int(0.01 * max(h, w))), 255, -1)
        sample |= m > 0
    return sample


def _polarity(th, bg, includes):
    """True if the threshold image must be inverted so the object is white."""
    h, w = th.shape
    if includes:
        vals = [th[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))] for x, y in includes]
        return np.mean(np.asarray(vals) > 0) < 0.5
    return (th[bg] > 0).mean() > 0.5


def _cand_otsu(gray, bg, includes, invert):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert is None:
        invert = _polarity(th, bg, includes)
    return 255 - th if invert else th


def _cand_adaptive(gray, bg, includes, invert):
    """Local threshold: good for thin dark/bright lines under uneven lighting."""
    h, w = gray.shape
    block = int(0.04 * max(h, w)) | 1
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    # Pick polarity from a global Otsu first (local threshold alone can't tell)
    _, glob = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = _polarity(glob, bg, includes) if invert is None else invert
    mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    c = 8 if inv else -8          # require a real contrast step, not noise
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, max(3, block), c)


def _cand_lines(gray, bg, includes, invert):
    """Thin-stroke detector. A morphological black-hat (closing minus image)
    responds to dark strokes narrower than the kernel and ignores broad,
    soft areas such as pencil shading, smudges and uneven light. Hysteresis
    keeps faint stretches only where they connect to clearly strong line."""
    h, w = gray.shape
    k = max(7, int(0.006 * max(h, w))) | 1
    blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
    _, glob = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = _polarity(glob, bg, includes) if invert is None else invert
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT if dark else cv2.MORPH_TOPHAT, se)
    t, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), 8.0)
    weak = (resp >= t).astype(np.uint8)
    strong = resp >= max(1.8 * t, float(np.percentile(resp, 99)))
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    good = np.zeros(n, bool)
    good[np.unique(labels[strong])] = True
    good[0] = False
    return good[labels].astype(np.uint8) * 255


def _cand_color(lab, bg):
    pix = lab[bg].reshape(-1, 3).astype(np.float32)
    # Robust background colour: median, plus spread so textured bg is tolerated.
    med = np.median(pix, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - med, axis=2)
    bg_d = np.linalg.norm(pix - med, axis=1)
    noise = np.percentile(bg_d, 95) + 1e-3
    d8 = np.clip(dist / (noise * 6) * 255, 0, 255).astype(np.uint8)
    d8 = cv2.GaussianBlur(d8, (0, 0), 1.0)
    t, th = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Never accept a threshold inside the background noise band.
    floor = int(255 / 6)
    if t < floor:
        _, th = cv2.threshold(d8, floor, 255, cv2.THRESH_BINARY)
    return th


def _cand_edges(gray, grad):
    """Strong edges -> bridge gaps by dilating -> fill -> erode back."""
    blur = cv2.bilateralFilter(gray, 9, 50, 9)
    # Thresholds from the gradient distribution, so surface texture/noise is ignored
    hi = float(np.percentile(grad, 97))
    e = cv2.Canny(blur, 0.4 * hi, hi, L2gradient=True) if hi > 0 else np.zeros_like(gray)
    h, w = gray.shape
    r = max(2, 0.008 * max(h, w))
    e = cv2.dilate(e, _disk(r))
    filled = _fill_holes(e)
    return cv2.erode(filled, _disk(r))


def _cand_grabcut(bgr, lasso_mask, tight_masks, includes, excludes, iters=4, size=360):
    """GrabCut is O(pixels*iters) and slow, so run it on a small copy and upsample
    the result. The cleanup + smoothing stages absorb the resolution loss."""
    H0, W0 = bgr.shape[:2]
    s = min(1.0, size / max(H0, W0))
    if s < 1:
        small = lambda m, interp=cv2.INTER_NEAREST: None if m is None else cv2.resize(
            m, (round(W0 * s), round(H0 * s)), interpolation=interp)
        bgr = small(bgr, cv2.INTER_AREA)
        lasso_mask = small(lasso_mask)
        if tight_masks is not None:
            tight_masks = tuple(small(m) for m in tight_masks)
    out = _grabcut(bgr, lasso_mask, tight_masks,
                   [(x * s, y * s) for x, y in includes], [(x * s, y * s) for x, y in excludes], iters)
    if s < 1:
        out = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)
        out = (cv2.GaussianBlur(out, (0, 0), 1 / s / 2) > 127).astype(np.uint8) * 255
    return out


def _grabcut(bgr, lasso_mask, tight_masks, includes, excludes, iters):
    h, w = bgr.shape[:2]
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    if tight_masks is not None:
        inner, outer = tight_masks
        gc[outer > 0] = cv2.GC_PR_BGD
        gc[lasso_mask > 0] = cv2.GC_PR_FGD
        gc[inner > 0] = cv2.GC_FGD
    elif lasso_mask is not None:
        gc[lasso_mask > 0] = cv2.GC_PR_FGD
    else:
        m = int(0.03 * max(h, w))
        gc[m:h - m, m:w - m] = cv2.GC_PR_FGD
    r = max(2, int(0.012 * max(h, w)))
    for x, y in includes:
        cv2.circle(gc, (int(x), int(y)), r, int(cv2.GC_FGD), -1)
    for x, y in excludes:
        cv2.circle(gc, (int(x), int(y)), r, int(cv2.GC_BGD), -1)
    if not (gc == cv2.GC_BGD).any() or not ((gc == cv2.GC_PR_FGD) | (gc == cv2.GC_FGD)).any():
        return np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _hits(mask, pts):
    if not pts:
        return None
    h, w = mask.shape
    return float(np.mean([mask[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))] > 0
                          for x, y in pts]))


def _metrics(mask, grad, lasso_mask, includes, excludes):
    h, w = mask.shape
    area = float((mask > 0).sum())
    m = {"area_frac": area / (h * w)}
    if area == 0:
        return m
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.concatenate([k.reshape(-1, 2) for k in cnts])
    on_border = (c[:, 0] <= 1) | (c[:, 1] <= 1) | (c[:, 0] >= w - 2) | (c[:, 1] >= h - 2)
    m["border_frac"] = float(on_border.mean())
    # Edge alignment: gradient along the contour vs the strongest edges in the
    # image. Contour pixels sit up to ~1px off the true edge, so take a local max.
    gmax = cv2.dilate(grad, _disk(2))
    g = gmax[c[:, 1], c[:, 0]]
    ref = float(np.percentile(grad[grad > 0], 99.5)) if (grad > 0).any() else 1.0
    m["edge_strength"] = float(np.clip(np.mean(np.minimum(g / (ref + 1e-6), 1.0)), 0, 1))
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    m["roughness"] = float(cv2.arcLength(big, True) / max(cv2.arcLength(hull, True), 1e-6))
    m["pieces"] = len(cnts)
    if lasso_mask is not None:
        m["lasso_iou"] = _iou(mask, lasso_mask)
    if includes:
        m["include_hit"] = _hits(mask, includes)
    if excludes:
        m["exclude_hit"] = _hits(mask, excludes)
    return m


def _score(m, consensus, lasso_mode, thin):
    if m.get("area_frac", 0) < (0.0002 if thin else 0.002):
        return 0.0
    s = 0.45 * m["edge_strength"] + 0.35 * consensus
    s -= 0.6 * m["border_frac"]
    if not thin:
        s -= 0.05 * max(0.0, m["roughness"] - 1.5)
    if m["area_frac"] > (0.6 if thin else 0.97):
        s -= 0.5
    if "lasso_iou" in m and not thin:
        # tight: result should hug the trace; loose: object only needs to be plausible
        s += (0.4 if lasso_mode == "tight" else 0.1) * m["lasso_iou"]
    if "include_hit" in m:
        s += 0.3 * m["include_hit"] - 0.6 * (1 - m["include_hit"])
    if "exclude_hit" in m:
        s -= 0.6 * m["exclude_hit"]
    return float(s)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _gauss_kernel(sigma, n):
    r = min(int(3 * sigma), n // 2 - 1)
    if sigma <= 0 or r < 1:
        return None
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    k = _gauss_kernel(sigma, len(pts)) if len(pts) >= 8 else None
    if k is None:
        return pts.astype(float)
    r = len(k) // 2
    padded = np.concatenate([pts[-r:], pts, pts[:r]]).astype(float)
    return np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)], axis=1)


def _smooth_open(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth an open polyline; endpoints stay put."""
    k = _gauss_kernel(sigma, len(pts)) if len(pts) >= 5 else None
    if k is None:
        return pts.astype(float)
    r = len(k) // 2
    p = pts.astype(float)
    # Point-reflect around the ends so the curve keeps heading the same way.
    head = 2 * p[0] - p[1:r + 1][::-1]
    tail = 2 * p[-1] - p[-r - 1:-1][::-1]
    padded = np.concatenate([head, p, tail])
    out = np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)], axis=1)
    out[0], out[-1] = p[0], p[-1]
    return out


def _segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def is_simple(poly: np.ndarray) -> bool:
    """True if a closed polygon has no self-intersections (O(n^2), fine for n<~5k
    since we bbox-prune with numpy)."""
    n = len(poly)
    if n < 4:
        return n == 3
    a = poly
    b = np.roll(poly, -1, axis=0)
    mnx, mxx = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    mny, mxy = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if not len(j):
            continue
        hit = j[(mnx[j] <= mxx[i]) & (mxx[j] >= mnx[i]) & (mny[j] <= mxy[i]) & (mxy[j] >= mny[i])]
        for k in hit:
            if _segments_intersect(a[i], b[i], a[k], b[k]):
                return False
    return True


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _contour_to_polygon(raw, scale, smooth, epsilon):
    raw = raw.astype(float)
    # +0.5: findContours returns pixel indices; the boundary sits on pixel centres.
    poly = None
    for s in (smooth, smooth / 2, 0):
        sm = (_smooth_closed(raw, s) + 0.5) / scale
        eps = epsilon
        for _ in range(4):
            cand = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            if len(cand) >= 3 and is_simple(cand):
                poly = cand.astype(float)
                break
            eps /= 2
        if poly is not None:
            break
    if poly is None:
        poly = (raw + 0.5) / scale
    # Normalise orientation: counter-clockwise when viewed y-up (CAD convention).
    # In y-down image space that means negative signed area.
    if _signed_area(poly) > 0:
        poly = poly[::-1]
    return poly


def _mask_to_outlines(mask, scale, smooth, epsilon):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 4:
            continue
        poly = _contour_to_polygon(c.reshape(-1, 2), scale, smooth, epsilon)
        out.append({"points": np.round(poly, 3).tolist(), "closed": True})
    return out


# --------------------------------------------------------------------------
# Skeleton -> centerlines
# --------------------------------------------------------------------------

def _neighbours(P):
    """8-neighbour views of a padded binary image, in order
    N, NE, E, SE, S, SW, W, NW."""
    return (P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:],
            P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2])


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, then removal of redundant staircase pixels so the
    result is a clean 1-px, 8-connected skeleton. Returns bool array."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            n, ne, e, se, s, sw, w, nw = _neighbours(img)
            c = img[1:-1, 1:-1]
            B = n + ne + e + se + s + sw + w + nw
            seq = (n, ne, e, se, s, sw, w, nw, n)
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                m = (n * e * s == 0) & (e * s * w == 0)
            else:
                m = (n * e * w == 0) & (n * s * w == 0)
            rem = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rem.any():
                c[rem] = 0
                changed = True
        if not changed:
            break
    # Remove "simple" pixels (8-connectivity Yokoi number == 1) that are not
    # endpoints: only staircase corners qualify on an already-thin skeleton.
    # Done per 2x2 sub-field so no two removed pixels are neighbours.
    yy, xx = np.mgrid[0:img.shape[0] - 2, 0:img.shape[1] - 2]
    while True:
        changed = False
        for py in (0, 1):
            for px in (0, 1):
                n, ne, e, se, s, sw, w, nw = _neighbours(img)
                c = img[1:-1, 1:-1]
                x = [1 - v.astype(np.int8) for v in (e, ne, n, nw, w, sw, s, se)]
                x.append(x[0])
                x.append(x[1])
                yokoi = sum(x[k] - x[k] * x[k + 1] * x[k + 2] for k in (0, 2, 4, 6))
                B = n + ne + e + se + s + sw + w + nw
                rem = (c == 1) & (yokoi == 1) & (B >= 2) & ((yy % 2) == py) & ((xx % 2) == px)
                if rem.any():
                    c[rem] = 0
                    changed = True
        if not changed:
            break
    return img[1:-1, 1:-1] > 0


_OFFS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _skeleton_graph(sk: np.ndarray):
    """Trace a skeleton into pixel paths between nodes.

    Returns (paths, node_cluster) where each path is (list[(y,x)], closed) and
    node_cluster maps node pixels to a cluster id (adjacent junction pixels form
    one cluster)."""
    P = np.pad(sk, 1)
    H, W = sk.shape
    deg = np.zeros((H, W), np.uint8)
    for dy, dx in _OFFS:
        deg += P[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    deg = deg * sk

    def nbrs(y, x):
        return [(y + dy, x + dx) for dy, dx in _OFFS
                if 0 <= y + dy < H and 0 <= x + dx < W and sk[y + dy, x + dx]]

    node_mask = sk & (deg != 2)
    ncl, cl_labels = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    nodes = list(zip(*np.nonzero(node_mask)))
    used = np.zeros_like(sk)
    seen_steps = set()
    paths = []
    for node in nodes:
        for nb in nbrs(*node):
            if (node, nb) in seen_steps:
                continue
            path = [node]
            prev, cur = node, nb
            while True:
                path.append(cur)
                if node_mask[cur]:
                    break
                used[cur] = True
                nxt = [q for q in nbrs(*cur) if q != prev and q != path[-2]]
                nxt = [q for q in nxt if not used[q] or node_mask[q]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            seen_steps.add((node, nb))
            seen_steps.add((path[-1], path[-2]))
            paths.append((path, False))
    # Closed loops with no nodes at all (e.g. a ring)
    for y, x in zip(*np.nonzero(sk & ~used & ~node_mask)):
        if used[y, x]:
            continue
        start = (y, x)
        path = [start]
        used[start] = True
        prev, cur = None, start
        while True:
            nxt = [q for q in nbrs(*cur) if q != prev and not used[q]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            used[cur] = True
            path.append(cur)
        if len(path) > 2:
            paths.append((path, True))
    return paths, cl_labels


def _prune_and_merge(paths, cl_labels, sk_deg1, prune_len):
    """Remove short side-branches ("spurs") and join paths that pass straight
    through a node, so a spring comes out as one long curve."""
    def cl(p):
        return int(cl_labels[p])

    # Drop tiny links between pixels of the same junction cluster
    edges = []
    for path, closed in paths:
        if closed:
            edges.append([path, True])
            continue
        a, b = cl(path[0]), cl(path[-1])
        if a and a == b and len(path) <= 3:
            continue
        edges.append([path, False])

    def degree():
        d = {}
        for path, closed in edges:
            if closed:
                continue
            for end in (path[0], path[-1]):
                c = cl(end)
                if c:
                    d[c] = d.get(c, 0) + 1
        return d

    # Prune spurs: short edges from a free end (endpoint) into a junction
    for _ in range(3):
        d = degree()
        keep = []
        removed = False
        for path, closed in edges:
            if not closed and len(path) < prune_len:
                ends = [path[0], path[-1]]
                free = [sk_deg1[e] for e in ends]
                junction = [d.get(cl(e), 0) >= 3 for e in ends]
                if (free[0] and junction[1]) or (free[1] and junction[0]):
                    removed = True
                    continue
            keep.append([path, closed])
        edges = keep
        if not removed:
            break

    # Merge chains through clusters that now have exactly two edges
    changed = True
    while changed:
        changed = False
        d = degree()
        for c, k in d.items():
            if k != 2:
                continue
            idx = [i for i, (p, cl_) in enumerate(edges) if not cl_ and c in (cl(p[0]), cl(p[-1]))]
            if len(idx) != 2:
                continue
            i, j = idx
            p, q = edges[i][0], edges[j][0]
            if cl(p[0]) == c:
                p = p[::-1]
            if cl(q[-1]) == c:
                q = q[::-1]
            merged = p + q[1:]
            closed = cl(merged[0]) == c and cl(merged[-1]) == c and len(merged) > 3
            edges = [e for t, e in enumerate(edges) if t not in (i, j)] + [[merged, closed]]
            changed = True
            break
    # At real junctions (3+ edges, e.g. a pencil stroke crossing the wire), join
    # the two edges that continue most straight through it; leave the others.
    changed = True
    while changed:
        changed = False
        for c, k in degree().items():
            if k < 3:
                continue
            idx = [(i, cl(p[0]) == c) for i, (p, cl_) in enumerate(edges)
                   if not cl_ and c in (cl(p[0]), cl(p[-1]))]
            best = None
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    (i, si), (j, sj) = idx[x], idx[y]
                    if i == j:
                        continue
                    dot = float(_end_dir(edges[i][0], si) @ _end_dir(edges[j][0], sj))
                    if dot < -0.7 and (best is None or dot < best[0]):
                        best = (dot, i, si, j, sj)
            if best:
                _, i, si, j, sj = best
                p = edges[i][0][::-1] if si else edges[i][0]      # p ends at the cluster
                q = edges[j][0] if sj else edges[j][0][::-1]      # q starts at the cluster
                merged = p + _link(p[-1], q[0]) + q[1:] if p[-1] != q[0] else p + q[1:]
                edges = [e for t, e in enumerate(edges) if t not in (i, j)] + [[merged, False]]
                changed = True
                break
    # A single edge that starts and ends in the same cluster is a loop
    for e in edges:
        if not e[1] and cl(e[0][0]) and cl(e[0][0]) == cl(e[0][-1]) and len(e[0]) > 3:
            e[1] = True
    return edges


def _end_dir(path, at_start, k=8):
    """Unit vector pointing OUT of a pixel path at one of its ends (x, y)."""
    if len(path) < 2:
        return np.zeros(2)
    a = path[0] if at_start else path[-1]
    b = path[min(k, len(path) - 1)] if at_start else path[max(0, len(path) - 1 - k)]
    v = np.array([a[1] - b[1], a[0] - b[0]], float)
    n = np.hypot(*v)
    return v / n if n else v


def _link(p, q):
    """Pixels on a straight line from p to q (exclusive of p), (y, x) tuples."""
    n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1])))
    return [(int(round(p[0] + (q[0] - p[0]) * t / n)), int(round(p[1] + (q[1] - p[1]) * t / n)))
            for t in range(1, n + 1)] if n else []


def _join_gaps(edges, gap, cos_min=0.7, far_gap=0.0, lw=2.0):
    """Bridge breaks in a line: connect two open ends that point at each other
    (so neighbouring coils of a spring, which run side by side, are never
    cross-connected). Two tiers:
      - short breaks (<= gap) with loosely matching directions;
      - longer breaks (<= far_gap) only when both ends point almost straight
        at each other with very little sideways offset."""
    open_ = [e for e in edges if not e[1] and len(e[0]) >= 2]
    rest = [e for e in edges if e[1] or len(e[0]) < 2]
    n = len(open_)
    if n < 2:
        return edges
    ends = []  # (path index, at_start, point yx, outward dir xy)
    for i, (path, _) in enumerate(open_):
        for at_start in (True, False):
            ends.append((i, at_start, path[0] if at_start else path[-1], _end_dir(path, at_start)))
    P = np.array([[e[2][1], e[2][0]] for e in ends], float)
    D = np.array([e[3] for e in ends])
    diff = P[None, :, :] - P[:, None, :]                      # from end a to end b
    dist = np.hypot(diff[..., 0], diff[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        u = diff / dist[..., None]
    ca = np.einsum("abk,ak->ab", u, D)                        # a points toward b
    cb = np.einsum("abk,bk->ab", -u, D)                       # b points toward a
    owner = np.array([e[0] for e in ends])
    lateral = np.abs(diff[..., 0] * D[:, None, 1] - diff[..., 1] * D[:, None, 0])
    near = (dist <= gap) & (ca >= cos_min) & (cb >= cos_min)
    far = (dist <= far_gap) & (ca >= 0.9) & (cb >= 0.9) & (lateral <= np.maximum(3 * lw, 0.12 * dist))
    ok = (near | far) & (dist > 0) & (owner[:, None] != owner[None, :])
    a_idx, b_idx = np.nonzero(np.triu(ok))
    order = np.argsort(dist[a_idx, b_idx] * (2.5 - ca[a_idx, b_idx] - cb[a_idx, b_idx]))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    link = {}
    for t in order:
        a, b = int(a_idx[t]), int(b_idx[t])
        if a in link or b in link:
            continue
        ra, rb = find(owner[a]), find(owner[b])
        if ra == rb:
            continue
        parent[ra] = rb
        link[a], link[b] = b, a
    if not link:
        return edges
    # Walk each chain of linked paths from a free end
    used = [False] * n
    chains = []
    for start_end in range(len(ends)):
        i = owner[start_end]
        if used[i] or start_end in link:
            continue
        pix, e = [], start_end
        while True:
            i = owner[e]
            used[i] = True
            path = open_[i][0] if ends[e][1] else open_[i][0][::-1]   # walk away from end e
            if pix:
                pix += _link(pix[-1], path[0])
            pix += list(path)
            far = e + 1 if ends[e][1] else e - 1                       # the path's other end
            if far not in link:
                break
            e = link[far]
        chains.append([pix, False])
    return rest + chains + [open_[i] for i in range(n) if not used[i]]


def _path_len(path):
    a = np.asarray(path, float)
    return float(np.hypot(*np.diff(a, axis=0).T).sum()) if len(a) > 1 else 0.0


def _mask_to_centerlines(mask, scale, smooth, epsilon, prune_len, includes=(), excludes=(),
                         pieces="all"):
    sk = thin(mask)
    if not sk.any():
        return [], {}
    P = np.pad(sk, 1)
    H, W = sk.shape
    deg = np.zeros((H, W), np.uint8)
    for dy, dx in _OFFS:
        deg += P[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    deg1 = sk & (deg == 1)
    paths, cl_labels = _skeleton_graph(sk)
    edges = _prune_and_merge(paths, cl_labels, deg1, prune_len)

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    widths = 2 * dist[sk]
    lw = float(np.median(widths)) if len(widths) else 2.0
    edges = _join_gaps(edges, gap=max(8.0, 5 * lw), far_gap=0.05 * max(mask.shape), lw=lw)

    # Which curves to keep: drop those near a background click; then those
    # near an object click, else the longest, else all
    reach = 0.02 * max(mask.shape)

    def gap_to(path, x, y):
        a = np.asarray(path, float)
        return float(np.min(np.hypot(a[:, 1] - x, a[:, 0] - y)))

    def near(path, pts):
        return any(gap_to(path, x, y) <= reach for x, y in pts)
    # A background click removes only the single closest curve, so clicking a
    # stray stroke beside a coil doesn't also delete the coil.
    for x, y in excludes:
        if edges:
            d = [gap_to(e[0], x, y) for e in edges]
            i = int(np.argmin(d))
            if d[i] <= reach:
                edges = edges[:i] + edges[i + 1:]
    if includes:
        picked = [e for e in edges if near(e[0], includes)]
        edges = picked or edges
    elif pieces == "largest" and edges:
        edges = [max(edges, key=lambda e: _path_len(e[0]))]

    out = []
    for path, closed in sorted(edges, key=lambda e: -len(e[0])):
        if len(path) < 3 or (not closed and len(path) < prune_len / 2):
            continue
        pts = np.array([(x, y) for y, x in path], float)
        if closed and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        sm = _smooth_closed(pts, smooth) if closed else _smooth_open(pts, smooth)
        sm = (sm + 0.5) / scale
        simp = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2)
        if len(simp) < 2:
            continue
        out.append({"points": np.round(simp, 3).tolist(), "closed": bool(closed)})

    info = {"line_width_px": round(lw / scale, 2), "line_width_work_px": round(lw, 2)}
    return out, info


# --------------------------------------------------------------------------
# Snapping (for the scale tool)
# --------------------------------------------------------------------------

class SnapIndex:
    """Precomputed corner strength and edge map for snapping clicks."""

    def __init__(self, image):
        img = load_image(image)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        self.gray = gray
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        self.eig = cv2.cornerMinEigenVal(blur, blockSize=5, ksize=3)
        nz = self.eig[self.eig > 0]
        self.eig_thresh = float(np.percentile(nz, 99.5)) * 0.3 if nz.size else np.inf
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
        grad = cv2.magnitude(gx, gy)
        hi = float(np.percentile(grad, 97))
        self.edges = cv2.Canny(blur, 0.4 * hi, hi, L2gradient=True) if hi > 0 else np.zeros_like(gray)

    def snap(self, pt, radius):
        """Nearest strong corner within radius, else nearest edge pixel, else pt."""
        h, w = self.gray.shape
        x, y = float(pt[0]), float(pt[1])
        r = max(2, int(round(radius)))
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return [x, y], "none"
        eig = self.eig[y0:y1, x0:x1]
        peaks = (eig >= self.eig_thresh) & (eig == cv2.dilate(eig, _disk(2)))
        cand = np.argwhere(peaks)
        if len(cand):
            d = (cand[:, 1] + x0 - x) ** 2 + (cand[:, 0] + y0 - y) ** 2
            ok = d <= r * r
            if ok.any():
                cy, cx = cand[ok][int(np.argmin(d[ok]))]
                p = np.array([[[cx + x0, cy + y0]]], np.float32)
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                p = cv2.cornerSubPix(self.gray, p, (3, 3), (-1, -1), crit)
                return [round(float(p[0, 0, 0]), 2), round(float(p[0, 0, 1]), 2)], "corner"
        e = np.argwhere(self.edges[y0:y1, x0:x1] > 0)
        if len(e):
            d = (e[:, 1] + x0 - x) ** 2 + (e[:, 0] + y0 - y) ** 2
            ok = d <= r * r
            if ok.any():
                ey, ex = e[ok][int(np.argmin(d[ok]))]
                return [float(ex + x0), float(ey + y0)], "edge"
        return [x, y], "none"


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def trace(image, params: TraceParams | None = None, *, return_masks=False):
    p = params or TraceParams()
    if p.shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    if p.pieces not in PIECES:
        raise ValueError(f"pieces must be one of {PIECES}")
    thin_mode = p.shape == "centerline"
    img = load_image(image)
    H, W = img.shape[:2]
    warnings: list[str] = []

    scale = min(1.0, p.work_size / max(H, W)) if p.work_size else 1.0
    work = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    h, w = work.shape[:2]
    includes = [(x * scale, y * scale) for x, y in (p.include or [])]
    excludes = [(x * scale, y * scale) for x, y in (p.exclude or [])]

    alpha = None
    if work.shape[2] == 4:
        a = work[:, :, 3]
        if (a < 128).mean() > 0.01 and (a >= 128).mean() > 0.01:
            alpha = (a >= 128).astype(np.uint8) * 255
        # Composite onto mid-grey so transparent areas don't become black objects
        af = a[:, :, None].astype(np.float32) / 255
        work = (work[:, :, :3] * af + 128 * (1 - af)).astype(np.uint8)

    lasso_mask = None
    tight_masks = None
    if p.lasso is not None and len(p.lasso) >= 3:
        lasso_w = np.asarray(p.lasso, float) * scale
        lasso_mask = _poly_mask(work.shape, lasso_w)
        if p.lasso_mode == "tight" and not thin_mode:
            snap = (p.snap_px * scale) if p.snap_px else max(4.0, 0.025 * max(h, w))
            inner = cv2.erode(lasso_mask, _disk(snap))
            outer = cv2.dilate(lasso_mask, _disk(snap))
            tight_masks = (inner, outer)
    elif p.lasso is not None and len(p.lasso):
        warnings.append("Lasso ignored: needs at least 3 points")

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    gblur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(gblur, cv2.CV_32F, 1, 0), cv2.Sobel(gblur, cv2.CV_32F, 0, 1))
    bg = _bg_sample(work.shape, lasso_mask, excludes)

    raw: dict[str, np.ndarray] = {}
    all_methods = THIN_METHODS if thin_mode else METHODS
    wanted = all_methods if p.method == "auto" else (p.method,)
    if alpha is not None and p.method in ("auto", "alpha"):
        raw["alpha"] = alpha
    for name in wanted:
        if name == "otsu":
            raw[name] = _cand_otsu(gray, bg, includes, p.invert)
        elif name == "color":
            raw[name] = _cand_color(lab, bg)
        elif name == "edges":
            raw[name] = _cand_edges(gray, grad)
        elif name == "grabcut":
            raw[name] = _cand_grabcut(work, lasso_mask, tight_masks, includes, excludes)
        elif name == "lines":
            raw[name] = _cand_lines(gray, bg, includes, p.invert)
        elif name == "adaptive":
            raw[name] = _cand_adaptive(gray, bg, includes, p.invert)
        elif name != "alpha":
            raise ValueError(f"Unknown method {name!r}; use auto or one of "
                             f"{sorted(set(METHODS) | set(THIN_METHODS))}")

    masks: dict[str, np.ndarray] = {}
    for name, m in raw.items():
        if tight_masks is not None:
            inner, outer = tight_masks
            m = cv2.bitwise_or(cv2.bitwise_and(m, outer), inner)
        elif lasso_mask is not None:
            m = cv2.bitwise_and(m, lasso_mask)
        if thin_mode:
            masks[name] = _clean_thin(m, p.min_feature, [], [], "all")
        else:
            masks[name] = _clean(m, p.min_feature, includes, excludes, p.pieces)

    names = list(masks)
    cands: list[Candidate] = []
    infos: dict[str, dict] = {}
    for name in names:
        others = [masks[o] for o in names if o != name and masks[o].any()]
        consensus = float(np.mean([_iou(masks[name], o) for o in others])) if others else 0.5
        # In centerline mode clicks act on traced curves, not on the pixel mask
        met = _metrics(masks[name], grad, lasso_mask, includes, [] if thin_mode else excludes)
        met["consensus"] = consensus
        sc = 1.0 if name == "alpha" else _score(met, consensus, p.lasso_mode, thin_mode)
        paths: list[dict] = []
        if masks[name].any():
            if thin_mode:
                prune = max(10.0, 3 * p.min_feature)
                paths, info = _mask_to_centerlines(masks[name], scale, p.smooth, p.epsilon, prune,
                                                   includes=includes, excludes=excludes,
                                                   pieces=p.pieces)
                infos[name] = info
            else:
                paths = _mask_to_outlines(masks[name], scale, p.smooth, p.epsilon)
        first_closed = next((q["points"] for q in paths if q["closed"]), [])
        cands.append(Candidate(name, round(sc, 4), {k: round(v, 4) for k, v in met.items()},
                               paths, first_closed))
    cands.sort(key=lambda c: c.score, reverse=True)
    cands = [c for c in cands if c.paths] + [c for c in cands if not c.paths]

    best = cands[0] if cands and cands[0].paths else None
    if best is None:
        raise RuntimeError("Nothing found. Click on the object, or draw a lasso around it.")

    if tight_masks is not None:
        # Tight mode assumes everything well inside the hand trace is object. Check
        # that against an unconstrained (loose) trace: if the core contains a lot of
        # background, the trace was too far from the real edge -> use loose result.
        loose_res, loose_masks, _ = trace(image, replace(p, lasso_mode="loose"), return_masks=True)
        inner = tight_masks[0]
        lm = loose_masks.get(loose_res.method)
        if inner.any() and lm is not None:
            core_fill = float(((lm > 0) & (inner > 0)).sum() / (inner > 0).sum())
            if core_fill < 0.9:
                loose_res.warnings.insert(0, (
                    f"Tight trace encloses background ({1 - core_fill:.0%} of its core); "
                    "fell back to loose lasso mode. Trace closer to the edge or raise snap distance."))
                return (loose_res, loose_masks, scale) if return_masks else loose_res

    info = dict(infos.get(best.method, {}))
    info["pieces"] = len(best.paths)
    info["points"] = sum(len(q["points"]) for q in best.paths)
    if best.metrics.get("border_frac", 0) > 0.15:
        warnings.append("The result runs along the image edge; the object may be cut off, "
                        "or the background was picked. Click on the object or draw a lasso.")
    if best.score < 0.25:
        warnings.append("Low confidence. Click on the object (and on background to exclude), "
                        "or compare the other guesses.")
    if includes and best.metrics.get("include_hit", 1) < 1:
        warnings.append("Some of your object clicks are outside the result. Try another guess "
                        "from the table, or add a lasso.")
    if excludes and best.metrics.get("exclude_hit", 0) > 0:
        warnings.append("Some background clicks are still inside the result.")
    if len(cands) > 1 and cands[1].paths and cands[0].score - cands[1].score < 0.03 \
            and cands[0].metrics.get("consensus", 1) < 0.8:
        warnings.append(f"Top guesses '{cands[0].method}' and '{cands[1].method}' scored "
                        "nearly equal but disagree; check both.")
    if thin_mode and 0 < info.get("line_width_work_px", 99) < 2.5 and scale < 1:
        warnings.append("Lines are only ~2 px wide at this detail level and may break up. "
                        "Set Detail to High or Full.")

    res = TraceResult(best.polygon, best.paths, best.method, W, H, cands, warnings, info)
    if return_masks:
        return res, masks, scale
    return res


def render_overlay(image, result: TraceResult, lasso=None, show_all=True,
                   include=None, exclude=None) -> np.ndarray:
    """Debug image: chosen result in green, other candidates thin, lasso red,
    clicks as green/red dots."""
    img = load_image(image)[:, :, :3].copy()
    th = max(1, round(max(img.shape[:2]) / 500))
    palette = [(255, 128, 0), (0, 160, 255), (255, 0, 200), (0, 220, 220), (128, 128, 255)]

    def draw(paths, color, t):
        for q in paths:
            pts = np.round(np.asarray(q["points"])).astype(np.int32)
            cv2.polylines(img, [pts], q["closed"], color, t, cv2.LINE_AA)

    if show_all:
        for i, c in enumerate(result.candidates[1:]):
            draw(c.paths, palette[i % len(palette)], th)
    if lasso is not None and len(lasso) >= 3:
        cv2.polylines(img, [np.round(np.asarray(lasso)).astype(np.int32)], True, (0, 0, 255), th, cv2.LINE_AA)
    draw(result.paths, (0, 220, 0), th * 2)
    for pts, col in ((include or [], (0, 200, 0)), (exclude or [], (0, 0, 230))):
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), th * 4, col, -1, cv2.LINE_AA)
    return img
