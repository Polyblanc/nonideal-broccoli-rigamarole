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


# ---------------------------------------------------------------- clicks / pieces

def _two_objects():
    big, small = wrench_mask(), np.zeros((H, W), np.uint8)
    cv2.circle(small, (780, 120), 60, 255, -1)
    return big, small, photo(big | small)


def test_click_picks_the_clicked_object():
    big, small, img = _two_objects()
    res = trace(img, TraceParams(include=[[780, 120]]))
    assert poly_iou(res.polygon, small) > 0.93
    res = trace(img, TraceParams(include=[[450, 320]]))
    assert poly_iou(res.polygon, big) > 0.95


def test_pieces_all_and_exclude():
    big, small, img = _two_objects()
    res = trace(img, TraceParams(pieces="all"))
    assert len(res.paths) == 2 and all(p["closed"] for p in res.paths)
    res = trace(img, TraceParams(pieces="all", exclude=[[780, 120]]))
    assert len(res.paths) == 1
    assert poly_iou(res.paths[0]["points"], big) > 0.95


# ---------------------------------------------------------------- centerlines

def _wire_photo(draw):
    img = np.full((800, 800, 3), 205, np.float32)
    xx = np.mgrid[0:800, 0:800][1]
    img *= (0.85 + 0.25 * xx / 800)[..., None]
    m = np.zeros((800, 800), np.uint8)
    draw(m)
    img = img * (1 - m[..., None] / 255 * 0.75)
    return np.clip(img + RNG.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


def _dense(points, closed=False):
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    return np.concatenate([np.linspace(p[i], p[i + 1], 20) for i in range(len(p) - 1)])


def test_centerline_spiral_spring():
    t = np.linspace(0.6, 7 * np.pi, 4000)
    r = 18 + 14 * t
    truth = np.stack([400 + r * np.cos(t), 400 + r * np.sin(t)], 1)
    img = _wire_photo(lambda m: cv2.polylines(m, [truth.astype(np.int32)], False, 255, 4, cv2.LINE_AA))
    res = trace(img, TraceParams(shape="centerline"))
    assert len(res.paths) == 1 and not res.paths[0]["closed"]      # one continuous open curve
    path = _dense(res.paths[0]["points"])
    d_path = np.sqrt(((path[:, None] - truth[None, ::4]) ** 2).sum(-1)).min(1)
    d_truth = np.sqrt(((truth[::4, None] - path[None, ::3]) ** 2).sum(-1)).min(1)
    assert d_path.mean() < 1.5 and d_path.max() < 5        # on the wire
    assert d_truth.max() < 6                                 # covers the whole wire
    assert 3 < res.info["line_width_px"] < 7


def test_centerline_ring_is_closed_and_burr_pruned():
    def draw(m):
        cv2.circle(m, (400, 400), 250, 255, 5, cv2.LINE_AA)
        cv2.line(m, (650, 400), (658, 400), 255, 3)           # tiny burr on the outside
    res = trace(_wire_photo(draw), TraceParams(shape="centerline"))
    assert len(res.paths) == 1 and res.paths[0]["closed"]
    r = np.hypot(*(np.asarray(res.paths[0]["points"]) - 400).T)
    assert np.abs(r - 250).max() < 2.5


# ---------------------------------------------------------------- snapping / export / web

def test_snap_to_corner_then_edge():
    from profiler import SnapIndex
    img = np.full((300, 300, 3), 220, np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (40, 40, 40), -1)
    idx = SnapIndex(img)
    pt, kind = idx.snap([106, 95], 12)
    assert kind == "corner" and np.hypot(pt[0] - 100, pt[1] - 100) < 2
    pt, kind = idx.snap([150, 92], 12)
    assert kind == "edge" and abs(pt[1] - 100) <= 1.5
    assert idx.snap([30, 30], 8)[1] == "none"


def test_export_open_paths_and_curves():
    from profiler import mm_per_px_from_points
    paths = [{"points": [[0, 0], [50, 10], [100, 0]], "closed": False},
             {"points": [[0, 20], [100, 20], [100, 60], [0, 60]], "closed": True}]
    assert mm_per_px_from_points([0, 0], [30, 40], 10) == pytest.approx(0.2)
    svg = to_svg(paths)
    assert svg.count("<path") == 2 and svg.count(" Z") == 1
    assert " C " in to_svg(paths, curves=True)
    doc = ezdxf.read(io.StringIO(to_dxf(paths, 0.5)))
    lw = doc.modelspace().query("LWPOLYLINE")
    assert {e.dxf.layer for e in lw} == {"PROFILE", "CENTERLINE"}
    doc = ezdxf.read(io.StringIO(to_dxf(paths, 0.5, curves=True)))
    assert len(doc.modelspace().query("SPLINE")) == 2


def test_web_api_roundtrip():
    import base64, json, threading, urllib.error, urllib.request
    from profiler.web import make_server
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    try:
        assert b"Profile Tracer" in urllib.request.urlopen(base + "/").read()
        mask = pcb_mask()
        ok, png = cv2.imencode(".png", photo(mask, (40, 120, 40), (150, 170, 190)))
        img = post("/api/load", {"image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()})
        assert img["width"] == W
        res = post("/api/trace", {"image_id": img["id"], "include": [[400, 300]], "work_size": 1000})
        assert poly_iou(res["polygon"], mask) > 0.95
        snap = post("/api/snap", {"image_id": img["id"], "point": [715, 495], "radius": 15})
        assert snap["kind"] in ("corner", "edge")
        out = post("/api/export", {"paths": res["paths"], "format": "dxf", "mm_per_px": 0.1})
        assert "LWPOLYLINE" in out["text"]
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/trace", {"image_id": "nope"})
        assert e.value.code == 410
    finally:
        srv.shutdown()


def test_thin_strokes_ignore_pencil_shading_and_exclude_strays():
    t = np.linspace(0.6, 6 * np.pi, 4000)
    r = 20 + 13 * t
    truth = np.stack([400 + r * np.cos(t), 400 + r * np.sin(t)], 1)

    def draw(m):
        cv2.polylines(m, [truth.astype(np.int32)], False, 255, 3, cv2.LINE_AA)
    img = _wire_photo(draw).astype(np.float32)
    # broad, soft pencil shading across the coils + a stray hatch stroke
    yy, xx = np.mgrid[0:800, 0:800]
    for cx, cy in ((520, 520), (300, 330)):
        img *= (1 - 0.35 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 70.0 ** 2)))[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.line(img, (40, 760), (160, 700), (60, 60, 60), 2)
    res = trace(img, TraceParams(shape="centerline", pieces="all", method="lines"))
    pts = np.concatenate([_dense(p["points"]) for p in res.paths])
    d = np.sqrt(((pts[:, None] - truth[None, ::4]) ** 2).sum(-1)).min(1)
    stray = np.hypot(pts[:, 0] - 100, pts[:, 1] - 730) < 80
    assert np.percentile(d[~stray], 98) < 4                 # shading did not become lines
    res2 = trace(img, TraceParams(shape="centerline", pieces="all", method="lines", exclude=[[100, 730]]))
    pts2 = np.concatenate([_dense(p["points"]) for p in res2.paths])
    assert not (np.hypot(pts2[:, 0] - 100, pts2[:, 1] - 730) < 60).any()   # stray removed
    assert len(res2.paths) >= 1
