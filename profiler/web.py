"""Tiny stdlib HTTP server for the lasso UI. No framework needed.

POST /api/trace   {image: dataURL, lasso?, lasso_mode, method, smooth, epsilon, min_feature}
POST /api/export  {polygon, format: svg|dxf, mm_per_px?, width_mm?, height_mm?}
"""

from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import TraceParams, trace
from .export import to_svg, to_dxf, mm_per_px_from_width

STATIC = Path(__file__).parent / "static"
MAX_BODY = 60 * 1024 * 1024


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
        try:
            req = json.loads(self.rfile.read(n))
            if self.path == "/api/trace":
                return self._json(200, self._trace(req))
            if self.path == "/api/export":
                return self._json(200, self._export(req))
            self._json(404, {"error": "unknown endpoint"})
        except Exception as e:  # report to UI rather than dropping the connection
            self._json(400, {"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _trace(req):
        data = req["image"].split(",", 1)[-1]
        params = TraceParams(
            method=req.get("method", "auto"),
            lasso=req.get("lasso") or None,
            lasso_mode=req.get("lasso_mode", "loose"),
            snap_px=req.get("snap_px") or None,
            smooth=float(req.get("smooth", 1.5)),
            epsilon=float(req.get("epsilon", 0.75)),
            min_feature=float(req.get("min_feature", 3.0)),
        )
        return trace(base64.b64decode(data), params).to_dict()

    @staticmethod
    def _export(req):
        poly = req["polygon"]
        mmpp = req.get("mm_per_px") or mm_per_px_from_width(poly, req.get("width_mm"), req.get("height_mm"))
        fmt = req.get("format", "svg")
        text = to_dxf(poly, mmpp) if fmt == "dxf" else to_svg(poly, mmpp)
        return {"text": text, "mm_per_px": mmpp}


def serve(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"profiler UI on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
