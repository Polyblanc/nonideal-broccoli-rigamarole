"""Image -> single closed, non-self-intersecting outline.

Pipeline
--------
1. Load image (alpha channel is used directly if it carries a real cutout).
2. Downscale to a working resolution.
3. Build several candidate foreground masks with different classic methods
   (Otsu on gray, colour distance from background, Canny edges + fill, GrabCut).
4. Clean each mask: apply lasso constraints, fill holes, keep the largest
   blob, drop thin spurs, and round off pixel jaggies.
5. Score every candidate (edge alignment, consensus with the others, border
   contact, lasso fit) and pick the best.
6. Trace the outer contour, smooth it, rescale to source pixels, simplify, and
   guarantee the final polygon is simple (no self-intersections).

All coordinates returned are in *source image pixels*, x right, y down.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

METHODS = ("otsu", "color", "edges", "grabcut")
LASSO_MODES = ("loose", "tight")


# --------------------------------------------------------------------------
# Parameters / results
# --------------------------------------------------------------------------

@dataclass
class TraceParams:
    method: str = "auto"            # "auto" or one of METHODS
    lasso: Sequence[Sequence[float]] | None = None  # [[x, y], ...] in source px
    lasso_mode: str = "loose"       # "loose": object is somewhere inside the lasso
                                    # "tight": lasso is a rough hand trace of the outline
    snap_px: float | None = None    # tight mode: how far (source px) the result may
                                    # deviate from the hand trace. None = auto
    work_size: int = 1000           # longest side used for processing
    smooth: float = 1.5             # contour smoothing sigma, in working px (0 = off)
    min_feature: float = 3.0        # spurs/notches thinner than this (working px) are removed
    epsilon: float = 0.75           # polyline simplification tolerance, source px
    invert: bool | None = None      # force polarity for otsu (None = auto)


@dataclass
class Candidate:
    method: str
    score: float
    metrics: dict
    polygon: list[list[float]] = field(default_factory=list)  # source px


@dataclass
class TraceResult:
    polygon: list[list[float]]      # chosen outline, source px, CCW in y-up space
    method: str
    width: int
    height: int
    candidates: list[Candidate]
    warnings: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


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


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill all enclosed holes: we only care about the outer profile."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _clean(mask: np.ndarray, min_feature: float) -> np.ndarray:
    mask = (mask > 127).astype(np.uint8) * 255
    mask = _fill_holes(mask)
    mask = _largest_component(mask)
    if min_feature > 0 and mask.any():
        # Kernel diameter strictly greater than min_feature, so features up to
        # min_feature wide are removed (edge/grabcut candidates thicken hairlines).
        k = _disk(np.ceil(min_feature / 2 + 0.01))
        # Opening kills thin spurs/islands bridged to the body; closing fills
        # thin notches/cracks. Re-pick the largest blob afterwards since opening
        # can split a neck.
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if opened.any():
            mask = _largest_component(opened)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = _fill_holes(mask)
    # Round off staircase jaggies: blur then re-threshold.
    if mask.any():
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        mask = (mask > 127).astype(np.uint8) * 255
        mask = _largest_component(mask)
    return mask


# --------------------------------------------------------------------------
# Candidate generators
# --------------------------------------------------------------------------

def _bg_sample(shape, lasso_mask: np.ndarray | None) -> np.ndarray:
    """Boolean mask of pixels we believe are background."""
    h, w = shape[:2]
    if lasso_mask is not None:
        ring = cv2.dilate(lasso_mask, _disk(max(4, 0.03 * max(h, w)))) > 0
        sample = ring & (lasso_mask == 0)
        if sample.sum() > 50:
            return sample
    b = max(2, int(0.02 * max(h, w)))
    sample = np.zeros((h, w), bool)
    sample[:b, :] = sample[-b:, :] = sample[:, :b] = sample[:, -b:] = True
    return sample


def _cand_otsu(gray, bg, lasso_mask, invert):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert is None:
        # Foreground = the class the background sample mostly is NOT.
        invert = (th[bg] > 0).mean() > 0.5
    return 255 - th if invert else th


def _cand_color(lab, bg, lasso_mask):
    pix = lab[bg].reshape(-1, 3).astype(np.float32)
    # Robust background colour: median, plus spread so textured bg is tolerated.
    med = np.median(pix, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - med, axis=2)
    bg_d = np.linalg.norm(pix - med, axis=1)
    noise = np.percentile(bg_d, 95) + 1e-3
    d8 = np.clip(dist / (noise * 6) * 255, 0, 255).astype(np.uint8)
    d8 = cv2.GaussianBlur(d8, (0, 0), 1.5)
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


def _cand_grabcut(bgr, lasso_mask, tight_masks, iters=4, size=360):
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
    out = _grabcut(bgr, lasso_mask, tight_masks, iters)
    if s < 1:
        out = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)
        out = (cv2.GaussianBlur(out, (0, 0), 1 / s / 2) > 127).astype(np.uint8) * 255
    return out


def _grabcut(bgr, lasso_mask, tight_masks, iters):
    h, w = bgr.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    if tight_masks is not None:
        inner, outer = tight_masks
        gc[:] = cv2.GC_BGD
        gc[outer > 0] = cv2.GC_PR_BGD
        gc[lasso_mask > 0] = cv2.GC_PR_FGD
        gc[inner > 0] = cv2.GC_FGD
    elif lasso_mask is not None:
        gc[:] = cv2.GC_BGD
        gc[lasso_mask > 0] = cv2.GC_PR_FGD
    else:
        m = int(0.03 * max(h, w))
        gc[:] = cv2.GC_BGD
        gc[m:h - m, m:w - m] = cv2.GC_PR_FGD
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


def _metrics(mask, grad, lasso_mask):
    h, w = mask.shape
    area = float((mask > 0).sum())
    m = {"area_frac": area / (h * w)}
    if area == 0:
        return m
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    on_border = (c[:, 0] <= 1) | (c[:, 1] <= 1) | (c[:, 0] >= w - 2) | (c[:, 1] >= h - 2)
    m["border_frac"] = float(on_border.mean())
    # Edge alignment: gradient along the contour vs the strongest edges in the
    # image. Contour pixels sit up to ~1px off the true edge, so take a local max.
    gmax = cv2.dilate(grad, _disk(2))
    g = gmax[c[:, 1], c[:, 0]]
    ref = float(np.percentile(grad[grad > 0], 99.5)) if (grad > 0).any() else 1.0
    m["edge_strength"] = float(np.clip(np.mean(np.minimum(g / (ref + 1e-6), 1.0)), 0, 1))
    hull = cv2.convexHull(c.reshape(-1, 1, 2))
    perim, hperim = cv2.arcLength(c.reshape(-1, 1, 2), True), cv2.arcLength(hull, True)
    m["roughness"] = float(perim / max(hperim, 1e-6))
    if lasso_mask is not None:
        m["lasso_iou"] = _iou(mask, lasso_mask)
    return m


def _score(m, consensus, lasso_mode):
    if m.get("area_frac", 0) < 0.002:
        return 0.0
    s = 0.45 * m["edge_strength"] + 0.35 * consensus
    s -= 0.6 * m["border_frac"]
    s -= 0.05 * max(0.0, m["roughness"] - 1.5)
    if m["area_frac"] > 0.97:
        s -= 0.5
    if "lasso_iou" in m:
        # tight: result should hug the trace; loose: object only needs to be plausible
        s += (0.4 if lasso_mode == "tight" else 0.1) * m["lasso_iou"]
    return float(s)


# --------------------------------------------------------------------------
# Contour -> polygon
# --------------------------------------------------------------------------

def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(pts) < 8:
        return pts.astype(float)
    r = int(3 * sigma)
    r = min(r, len(pts) // 2 - 1)
    if r < 1:
        return pts.astype(float)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([pts[-r:], pts, pts[:r]]).astype(float)
    out = np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)], axis=1)
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


def _mask_to_polygon(mask, scale, smooth, epsilon):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    raw = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    # +0.5: findContours returns pixel indices; the boundary sits on pixel centres.
    raw_src = (raw + 0.5) / scale
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
        poly = raw_src
    # Normalise orientation: counter-clockwise when viewed y-up (CAD convention).
    # In y-down image space that means negative signed area.
    if _signed_area(poly) > 0:
        poly = poly[::-1]
    return poly


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def trace(image, params: TraceParams | None = None, *, return_masks=False):
    p = params or TraceParams()
    img = load_image(image)
    H, W = img.shape[:2]
    warnings: list[str] = []

    scale = min(1.0, p.work_size / max(H, W))
    work = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    h, w = work.shape[:2]

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
        if p.lasso_mode == "tight":
            snap = (p.snap_px * scale) if p.snap_px else max(4.0, 0.025 * max(h, w))
            inner = cv2.erode(lasso_mask, _disk(snap))
            outer = cv2.dilate(lasso_mask, _disk(snap))
            tight_masks = (inner, outer)
    elif p.lasso is not None:
        warnings.append("Lasso ignored: needs at least 3 points")

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    gx = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    bg = _bg_sample(work.shape, lasso_mask)

    raw: dict[str, np.ndarray] = {}
    wanted = METHODS if p.method == "auto" else (p.method,)
    if alpha is not None and p.method in ("auto", "alpha"):
        raw["alpha"] = alpha
    for name in wanted:
        if name == "otsu":
            raw[name] = _cand_otsu(gray, bg, lasso_mask, p.invert)
        elif name == "color":
            raw[name] = _cand_color(lab, bg, lasso_mask)
        elif name == "edges":
            raw[name] = _cand_edges(gray, grad)
        elif name == "grabcut":
            raw[name] = _cand_grabcut(work, lasso_mask, tight_masks)
        elif name != "alpha":
            raise ValueError(f"Unknown method {name!r}; use auto or one of {METHODS}")

    masks: dict[str, np.ndarray] = {}
    for name, m in raw.items():
        if tight_masks is not None:
            inner, outer = tight_masks
            m = cv2.bitwise_or(cv2.bitwise_and(m, outer), inner)
        elif lasso_mask is not None:
            m = cv2.bitwise_and(m, lasso_mask)
        masks[name] = _clean(m, p.min_feature)

    names = list(masks)
    cands: list[Candidate] = []
    for name in names:
        others = [masks[o] for o in names if o != name and masks[o].any()]
        consensus = float(np.mean([_iou(masks[name], o) for o in others])) if others else 0.5
        met = _metrics(masks[name], grad, lasso_mask)
        met["consensus"] = consensus
        sc = 1.0 if name == "alpha" else _score(met, consensus, p.lasso_mode)
        poly = _mask_to_polygon(masks[name], scale, p.smooth, p.epsilon) if masks[name].any() else np.zeros((0, 2))
        cands.append(Candidate(name, round(sc, 4), {k: round(v, 4) for k, v in met.items()},
                               np.round(poly, 3).tolist()))
    cands.sort(key=lambda c: c.score, reverse=True)
    cands = [c for c in cands if c.polygon] + [c for c in cands if not c.polygon]

    best = cands[0] if cands and cands[0].polygon else None
    if best is None:
        raise RuntimeError("No outline found. Try a lasso around the object.")

    if tight_masks is not None:
        # Tight mode assumes everything well inside the hand trace is object. Check
        # that against an unconstrained (loose) trace: if the core contains a lot of
        # background, the trace was too far from the real edge -> use loose result.
        from dataclasses import replace
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
    if best.metrics.get("border_frac", 0) > 0.15:
        warnings.append("Outline runs along the image border; the object may be cropped "
                        "or the background was picked. Try a lasso.")
    if best.score < 0.25:
        warnings.append("Low confidence result; compare candidates or draw a lasso.")
    if len(cands) > 1 and cands[1].polygon and cands[0].score - cands[1].score < 0.03 \
            and cands[0].metrics.get("consensus", 1) < 0.8:
        warnings.append(f"Top candidates '{cands[0].method}' and '{cands[1].method}' "
                        "scored nearly equal but disagree; check both.")

    res = TraceResult(best.polygon, best.method, W, H, cands, warnings)
    if return_masks:
        return res, masks, scale
    return res


def render_overlay(image, result: TraceResult, lasso=None, show_all=True) -> np.ndarray:
    """Debug image: chosen outline in green, other candidates thin, lasso dashed-ish."""
    img = load_image(image)[:, :, :3].copy()
    th = max(1, round(max(img.shape[:2]) / 500))
    palette = [(255, 128, 0), (0, 160, 255), (255, 0, 200), (0, 220, 220), (128, 128, 255)]
    if show_all:
        for i, c in enumerate(result.candidates[1:]):
            if c.polygon:
                pts = np.round(np.asarray(c.polygon)).astype(np.int32)
                cv2.polylines(img, [pts], True, palette[i % len(palette)], th, cv2.LINE_AA)
    if lasso is not None and len(lasso) >= 3:
        cv2.polylines(img, [np.round(np.asarray(lasso)).astype(np.int32)], True, (0, 0, 255), th, cv2.LINE_AA)
    pts = np.round(np.asarray(result.polygon)).astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 220, 0), th * 2, cv2.LINE_AA)
    return img
