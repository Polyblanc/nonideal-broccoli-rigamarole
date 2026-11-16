"""Command line: ``profiler trace photo.jpg -o part.svg``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .core import LASSO_MODES, METHODS, PIECES, SHAPES, THIN_METHODS, TraceParams, render_overlay, trace
from .export import mm_per_px_from_points, mm_per_px_from_width, to_dxf, to_svg


def parse_points(arg: str | None):
    """Points from a JSON file ([[x,y],...] or {"points": [...]}) or an inline
    string "x,y x,y x,y"."""
    if not arg:
        return None
    path = Path(arg)
    if path.suffix.lower() == ".json" and path.exists():
        data = json.loads(path.read_text())
        return data["points"] if isinstance(data, dict) else data
    return [[float(v) for v in pair.split(",")] for pair in arg.split()]


parse_lasso = parse_points  # backwards compatible name


def _scale(args, paths):
    if args.mm_per_px:
        return args.mm_per_px
    if args.scale_points:
        pts = parse_points(args.scale_points[0])
        if not pts or len(pts) != 2:
            raise SystemExit('--scale-points needs exactly two points: "x1,y1 x2,y2" DISTANCE_MM')
        return mm_per_px_from_points(pts[0], pts[1], float(args.scale_points[1]))
    return mm_per_px_from_width(paths, args.width_mm, args.height_mm)


def cmd_trace(args) -> int:
    lasso = parse_points(args.lasso)
    include, exclude = parse_points(args.include), parse_points(args.exclude)
    params = TraceParams(method=args.method, shape=args.shape, pieces=args.pieces,
                         include=include, exclude=exclude,
                         lasso=lasso, lasso_mode=args.lasso_mode,
                         snap_px=args.snap_px, smooth=args.smooth, epsilon=args.epsilon,
                         min_feature=args.min_feature, work_size=args.work_size)
    res = trace(args.image, params)

    paths = res.paths
    if args.pick:
        match = [c for c in res.candidates if c.method == args.pick and c.paths]
        if not match:
            print(f"No usable candidate '{args.pick}'", file=sys.stderr)
            return 2
        paths = match[0].paths

    mmpp = _scale(args, paths)
    outs = list(args.output or []) or [str(Path(args.image).with_suffix(".svg"))]
    for out in outs:
        o = Path(out)
        suffix = o.suffix.lower()
        if suffix == ".dxf":
            o.write_text(to_dxf(paths, mmpp, curves=args.curves))
        elif suffix == ".svg":
            o.write_text(to_svg(paths, mmpp, image_size=(res.width, res.height), curves=args.curves))
        elif suffix == ".json":
            o.write_text(json.dumps(res.to_dict(), indent=1))
        else:
            print(f"Unknown output type: {o}", file=sys.stderr)
            return 2
    if args.debug:
        cv2.imwrite(args.debug, render_overlay(args.image, res, lasso, include=include, exclude=exclude))

    summary = {
        "chosen": args.pick or res.method,
        "paths": [{"points": len(p["points"]), "closed": p["closed"]} for p in paths],
        "info": res.info,
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
    ap = argparse.ArgumentParser(prog="profiler", description="Photo -> outline / centerline (SVG/DXF)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="trace an image")
    t.add_argument("image")
    t.add_argument("-o", "--output", action="append",
                   help="output file(s): .svg, .dxf, .json (repeatable). Default: <image>.svg")
    t.add_argument("--shape", default="outline", choices=SHAPES,
                   help="outline: closed outer edge; centerline: middle of thin wires/springs/lines")
    t.add_argument("--pieces", default="largest", choices=PIECES,
                   help="largest: one object; all: every separate object")
    t.add_argument("--include", help='clicks ON the object: "x,y x,y" or JSON file (image px)')
    t.add_argument("--exclude", help='clicks on background to ignore: "x,y x,y" or JSON file')
    t.add_argument("--method", default="auto", choices=("auto",) + tuple(sorted(set(METHODS) | set(THIN_METHODS))))
    t.add_argument("--pick", help="export this candidate instead of the top-scored one")
    t.add_argument("--lasso", help='JSON file or "x,y x,y ..." in image pixels')
    t.add_argument("--lasso-mode", default="loose", choices=LASSO_MODES,
                   help="loose: object is inside the lasso; tight: lasso is a rough trace to snap")
    t.add_argument("--snap-px", type=float, help="tight mode: max deviation from trace (px)")
    t.add_argument("--smooth", type=float, default=1.5, help="smoothing sigma (px)")
    t.add_argument("--epsilon", type=float, default=0.75, help="simplification tolerance (px)")
    t.add_argument("--min-feature", type=float, default=3.0,
                   help="ignore specks/spurs/notches thinner than this (working px)")
    t.add_argument("--work-size", type=int, default=1000,
                   help="longest side used for processing (0 = full photo size)")
    t.add_argument("--curves", action="store_true", help="export smooth curves (Bezier/spline)")
    g = t.add_mutually_exclusive_group()
    g.add_argument("--mm-per-px", type=float)
    g.add_argument("--width-mm", type=float, help="real width of the result's bounding box")
    g.add_argument("--height-mm", type=float, help="real height of the result's bounding box")
    g.add_argument("--scale-points", nargs=2, metavar=("POINTS", "DISTANCE_MM"),
                   help='two image points a known distance apart: "x1,y1 x2,y2" 25.4')
    t.add_argument("--debug", help="write overlay PNG showing all candidates")
    t.set_defaults(func=cmd_trace)

    s = sub.add_parser("serve", help="browser UI with pick, lasso, zoom and scale tools")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
