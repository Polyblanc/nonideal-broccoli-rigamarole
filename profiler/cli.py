"""Command line: ``python -m profiler trace photo.jpg -o part.svg``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .core import METHODS, LASSO_MODES, TraceParams, trace, render_overlay
from .export import to_svg, to_dxf, mm_per_px_from_width


def parse_lasso(arg: str | None):
    """Lasso from a JSON file ([[x,y],...] or {"points": [...]}) or an inline
    string "x,y x,y x,y"."""
    if not arg:
        return None
    path = Path(arg)
    if path.suffix.lower() == ".json" and path.exists():
        data = json.loads(path.read_text())
        return data["points"] if isinstance(data, dict) else data
    return [[float(v) for v in pair.split(",")] for pair in arg.split()]


def _scale(args, polygon):
    if args.mm_per_px:
        return args.mm_per_px
    return mm_per_px_from_width(polygon, args.width_mm, args.height_mm)


def cmd_trace(args) -> int:
    lasso = parse_lasso(args.lasso)
    params = TraceParams(method=args.method, lasso=lasso, lasso_mode=args.lasso_mode,
                         snap_px=args.snap_px, smooth=args.smooth, epsilon=args.epsilon,
                         min_feature=args.min_feature, work_size=args.work_size)
    res = trace(args.image, params)

    polygon = res.polygon
    if args.pick:
        match = [c for c in res.candidates if c.method == args.pick and c.polygon]
        if not match:
            print(f"No usable candidate '{args.pick}'", file=sys.stderr)
            return 2
        polygon = match[0].polygon

    mmpp = _scale(args, polygon)
    outs = list(args.output or [])
    if not outs:
        outs = [str(Path(args.image).with_suffix(".svg"))]
    for out in outs:
        o = Path(out)
        if o.suffix.lower() == ".dxf":
            o.write_text(to_dxf(polygon, mmpp))
        elif o.suffix.lower() == ".svg":
            o.write_text(to_svg(polygon, mmpp, image_size=(res.width, res.height)))
        elif o.suffix.lower() == ".json":
            o.write_text(json.dumps(res.to_dict(), indent=1))
        else:
            print(f"Unknown output type: {o}", file=sys.stderr)
            return 2
    if args.debug:
        cv2.imwrite(args.debug, render_overlay(args.image, res, lasso))

    summary = {
        "chosen": args.pick or res.method,
        "points": len(polygon),
        "mm_per_px": mmpp,
        "candidates": [{"method": c.method, "score": c.score, **c.metrics} for c in res.candidates],
        "warnings": res.warnings,
        "outputs": outs,
    }
    print(json.dumps(summary, indent=1))
    return 0


def cmd_serve(args) -> int:
    from .web import serve
    serve(args.host, args.port)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="profiler", description="Photo -> outline profile (SVG/DXF)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="trace an image")
    t.add_argument("image")
    t.add_argument("-o", "--output", action="append",
                   help="output file(s): .svg, .dxf, .json (repeatable). Default: <image>.svg")
    t.add_argument("--method", default="auto", choices=("auto",) + METHODS)
    t.add_argument("--pick", help="export this candidate instead of the top-scored one")
    t.add_argument("--lasso", help='JSON file or "x,y x,y ..." in image pixels')
    t.add_argument("--lasso-mode", default="loose", choices=LASSO_MODES,
                   help="loose: object is inside the lasso; tight: lasso is a rough trace to snap")
    t.add_argument("--snap-px", type=float, help="tight mode: max deviation from trace (px)")
    t.add_argument("--smooth", type=float, default=1.5, help="contour smoothing sigma (px)")
    t.add_argument("--epsilon", type=float, default=0.75, help="simplification tolerance (px)")
    t.add_argument("--min-feature", type=float, default=3.0,
                   help="remove spurs/notches thinner than this (working px)")
    t.add_argument("--work-size", type=int, default=1000)
    g = t.add_mutually_exclusive_group()
    g.add_argument("--mm-per-px", type=float)
    g.add_argument("--width-mm", type=float, help="real width of the outline's bounding box")
    g.add_argument("--height-mm", type=float, help="real height of the outline's bounding box")
    t.add_argument("--debug", help="write overlay PNG showing all candidates")
    t.set_defaults(func=cmd_trace)

    s = sub.add_parser("serve", help="browser UI with lasso tool")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
