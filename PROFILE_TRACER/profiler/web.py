"""Tiny stdlib HTTP server for the browser UI. No framework needed.

POST /api/load    {image: dataURL}                         -> {id, width, height}
POST /api/trace   {image_id, include?, exclude?, lasso?, lasso_mode, shape, pieces,
                   method, smooth, epsilon, min_feature, work_size}
POST /api/snap    {image_id, point: [x, y], radius}         -> {point, kind}
POST /api/export  {paths, format: svg|dxf, curves?, mm_per_px? | width_mm? | height_mm?}
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import SnapIndex, TraceParams, load_image, trace
from .export import mm_per_px_from_width, to_dxf, to_svg

STATIC = Path(__file__).parent / "static"
MAX_BODY = 80 * 1024 * 1024


class UnknownImage(Exception):
    pass


class ImageCache:
    """Keeps the last few uploaded images (decoded) so each click doesn't
    re-upload and re-decode a 12-megapixel photo."""

    def __init__(self, size=6):
        self.size = size
        self.items: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()

    def add(self, data: bytes) -> tuple[str, dict]:
        key = hashlib.sha1(data).hexdigest()[:16]
        with self.lock:
            if key not in self.items:
                self.items[key] = {"image": load_image(data)}
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
            return key, self.items[key]

    def get(self, key: str) -> dict:
        with self.lock:
            if key not in self.items:
                raise UnknownImage(key)
            self.items.move_to_end(key)
            return self.items[key]


CACHE = ImageCache()


def _num(v, default, cast=float):
    return default if v in (None, "") else cast(v)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quieter
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            return self._json(413, {"error": "image too large"})
        routes = {"/api/load": self._load, "/api/trace": self._trace,
                  "/api/snap": self._snap, "/api/export": self._export}
        try:
            req = json.loads(self.rfile.read(n))
            if self.path not in routes:
                return self._json(404, {"error": "unknown endpoint"})
            return self._json(200, routes[self.path](req))
        except UnknownImage:
            self._json(410, {"error": "unknown_image"})
        except Exception as e:  # report to UI rather than dropping the connection
            self._json(400, {"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _image(req):
        if req.get("image_id"):
            return CACHE.get(req["image_id"])
        return CACHE.add(base64.b64decode(req["image"].split(",", 1)[-1]))[1]

    @staticmethod
    def _load(req):
        key, item = CACHE.add(base64.b64decode(req["image"].split(",", 1)[-1]))
        h, w = item["image"].shape[:2]
        return {"id": key, "width": w, "height": h}

    @classmethod
    def _trace(cls, req):
        item = cls._image(req)
        params = TraceParams(
            method=req.get("method") or "auto",
            shape=req.get("shape") or "outline",
            pieces=req.get("pieces") or "largest",
            include=req.get("include") or None,
            exclude=req.get("exclude") or None,
            lasso=req.get("lasso") or None,
            lasso_mode=req.get("lasso_mode") or "loose",
            snap_px=_num(req.get("snap_px"), None),
            smooth=_num(req.get("smooth"), 1.5),
            epsilon=_num(req.get("epsilon"), 0.75),
            min_feature=_num(req.get("min_feature"), 3.0),
            work_size=_num(req.get("work_size"), 1000, int),
        )
        return trace(item["image"], params).to_dict()

    @classmethod
    def _snap(cls, req):
        item = cls._image(req)
        if "snap" not in item:
            item["snap"] = SnapIndex(item["image"])
        pt, kind = item["snap"].snap(req["point"], float(req.get("radius", 12)))
        return {"point": pt, "kind": kind}

    @staticmethod
    def _export(req):
        paths = req.get("paths") or req.get("polygon")
        mmpp = req.get("mm_per_px") or mm_per_px_from_width(paths, req.get("width_mm"), req.get("height_mm"))
        curves = bool(req.get("curves"))
        fmt = req.get("format", "svg")
        text = to_dxf(paths, mmpp, curves=curves) if fmt == "dxf" else to_svg(paths, mmpp, curves=curves)
        return {"text": text, "mm_per_px": mmpp}


def make_server(host="127.0.0.1", port=8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host="127.0.0.1", port=8765):
    httpd = make_server(host, port)
    print(f"profiler UI on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
