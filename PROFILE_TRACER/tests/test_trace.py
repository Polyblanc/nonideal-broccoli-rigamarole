"""Synthetic 'photos' with known ground truth. Run: python -m pytest tests -q"""

import io

import cv2
import ezdxf
import numpy as np
import pytest

from profiler import TraceParams, trace, to_svg, to_dxf, is_simple, mm_per_px_from_width

RNG = np.random.default_rng(0)
W, H = 900, 640


def wrench_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (230, 290), (690, 350), 255, -1)           # handle
    cv2.circle(m, (200, 320), 85, 255, -1)                      # head
    cv2.rectangle(m, (90, 295), (210, 345), 0, -1)              # jaw slot (concavity)
    cv2.circle(m, (700, 320), 55, 255, -1)                      # ring end
    return m


def pcb_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (180, 140), (720, 500), 255, -1)
    cv2.rectangle(m, (420, 470), (500, 500), 0, -1)             # edge notch
    cv2.circle(m, (180, 140), 30, 0, -1)                        # corner relief
    return m


def photo(mask, fg=(60, 60, 70), bg=(200, 195, 185), shadow=True, noise=12):
    img = np.empty((H, W, 3), np.float32)
    img[:] = bg
    yy, xx = np.mgrid[0:H, 0:W]
    img *= (0.8 + 0.3 * xx / W)[..., None]                      # lighting gradient
    if shadow:
        sh = cv2.GaussianBlur(np.roll(np.roll(mask, 12, 0), 12, 1).astype(np.float32) / 255, (0, 0), 8)
        img *= (1 - 0.35 * sh)[..., None]
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.2)[..., None]  # soft edge
    tex = np.asarray(fg, np.float32) + 25 * np.sin(xx / 7.0)[..., None]          # machined texture
    img = img * (1 - m) + tex * m
    img += RNG.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def poly_iou(poly, mask):
    pm = np.zeros_like(mask)
    cv2.fillPoly(pm, [np.round(np.asarray(poly)).astype(np.int32)], 255)
    a, b = pm > 0, mask > 0
    return (a & b).sum() / (a | b).sum()


@pytest.mark.parametrize("maskfn,fg,bg", [
    (wrench_mask, (60, 60, 70), (200, 195, 185)),     # dark tool on light paper
    (wrench_mask, (210, 210, 215), (40, 45, 40)),     # bright part on dark mat
    (pcb_mask, (40, 120, 40), (150, 170, 190)),       # green PCB on similar-luma blue
])
def test_auto_trace(maskfn, fg, bg):
    mask = maskfn()
    res = trace(photo(mask, fg, bg))
    assert poly_iou(res.polygon, mask) > 0.95, res.candidates
    assert is_simple(np.asarray(res.polygon))


def test_lasso_isolates_object_among_distractors():
    target = wrench_mask()
    img = photo(target)
    # distractor: another dark blob in the corner, plus a pen line
    cv2.circle(img, (820, 90), 50, (50, 50, 60), -1)
    cv2.line(img, (30, 600), (400, 560), (40, 40, 40), 6)
    lasso = [[60, 200], [800, 200], [800, 440], [60, 440]]
    res = trace(img, TraceParams(lasso=lasso))
    assert poly_iou(res.polygon, target) > 0.95


def test_tight_lasso_snaps_to_edges():
    target = pcb_mask()
    img = photo(target, (40, 120, 40), (150, 170, 190))
    # sloppy hand trace: true outline + ~8px wobble
    cnt = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)[::15]
    wobble = cnt + RNG.normal(0, 6, cnt.shape)
    res = trace(img, TraceParams(lasso=wobble.tolist(), lasso_mode="tight"))
    assert poly_iou(res.polygon, target) > 0.96


def test_alpha_cutout_used():
    mask = wrench_mask()
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = mask
    res = trace(rgba)
    assert res.method == "alpha"
    assert poly_iou(res.polygon, mask) > 0.97


def test_spur_and_island_removed():
    mask = pcb_mask()
    img = photo(mask, (40, 120, 40), (150, 170, 190), shadow=False)
    cv2.line(img, (720, 300), (850, 300), (40, 120, 40), 2)     # 2px hair attached to edge
    cv2.circle(img, (60, 60), 6, (40, 120, 40), -1)            # speck
    res = trace(img, TraceParams(min_feature=5))
    xs = np.asarray(res.polygon)[:, 0]
    assert xs.max() < 735 and xs.min() > 170


def test_exports_scale_and_orientation():
    poly = [[0, 0], [100, 0], [100, 50], [0, 50]]
    mmpp = mm_per_px_from_width(poly, width_mm=200)
    assert mmpp == pytest.approx(2.0)
    svg = to_svg(poly, mmpp)
    assert 'width="200.0000mm"' in svg and 'height="100.0000mm"' in svg
    doc = ezdxf.read(io.StringIO(to_dxf(poly, mmpp)))
    pl = doc.modelspace().query("LWPOLYLINE")[0]
    pts = np.array([p[:2] for p in pl.get_points()])
    assert pl.closed
    assert np.allclose(pts.max(0) - pts.min(0), [200, 100])


def test_tight_mode_with_sloppy_trace_falls_back_to_loose():
    target = wrench_mask()
    img = photo(target)
    a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    far = np.stack([420 + 400 * np.cos(a), 320 + 200 * np.sin(a)], 1)  # nowhere near the edge
    res = trace(img, TraceParams(lasso=far.tolist(), lasso_mode="tight"))
    assert any("fell back" in w for w in res.warnings)
    assert poly_iou(res.polygon, target) > 0.95
