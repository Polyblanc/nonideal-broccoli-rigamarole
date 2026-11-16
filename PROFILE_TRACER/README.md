# PROFILE_TRACER — photo → outline or centerline (SVG / DXF)

Give it a photo of a machine part, PCB, tool or side profile and it returns a clean
closed outline (holes filled, no self-intersections). For thin things such as wire,
springs or drawn lines, it returns open **centerlines** instead.

## The browser page in 30 seconds
1. Open a photo. It traces straight away.
2. If the guess is wrong, pick **➕ Pick** and click on the object. Shift-click or
   right-click on anything it should ignore. It re-traces after every click.
3. Still wrong? Use **◯ Lasso** to fence the object off, or try another row in the
   **Guesses** table.
4. Scroll to zoom. Drag with **✋ Pan**, Space, or the middle mouse button to move around.
5. **📏 Scale**: click two points a known distance apart (they snap to corners and
   edges; hold Alt to place a point exactly), then type the real distance.
6. Download the SVG or DXF.

Every setting has a **?** you can hover over or tap for a plain-language explanation.

## Use it

Once installed as a command (see the [top-level README](../README.md)):

```bash
profiler serve                       # browser page → http://127.0.0.1:8765
profiler trace part.jpg -o part.svg -o part.dxf --width-mm 85 --debug overlay.png
profiler trace part.jpg --include "820,410" --exclude "100,90" -o part.dxf   # "click" hints
profiler trace spring.jpg --shape centerline --work-size 2000 --curves -o spring.dxf
profiler trace parts.jpg --pieces all --scale-points "120,40 980,44" 100 -o sheet.dxf
```

Without installing, from inside this folder:

```bash
uv run profiler serve
```

## How it guesses
It runs several classic OpenCV segmentations and scores each one, rather than relying on a single threshold:

| candidate | good for |
|---|---|
| `otsu`    | high-contrast silhouettes (dark part on paper, backlit) |
| `color`   | coloured objects on a plain background whose brightness is similar (green PCB on a blue mat) |
| `edges`   | sharp-edged parts on a uniform background |
| `grabcut` | textured or cluttered backgrounds, specular highlights |
| `alpha`   | PNGs that already have a transparent background (used directly) |
| `adaptive`| thin lines under uneven light (centerline mode only) |

Clicks guide every method. A click on the object decides which piece is kept, and the
Smart cut treats it as a sure "this is object" seed. A background click works the opposite way.

Every mask is then cleaned up: holes are filled, only the clicked or largest piece is kept (or all pieces), spurs and
notches thinner than `--min-feature` are removed, and pixel jaggies are rounded off. Each
candidate is scored on how well it lines up with image edges, how much it agrees with the
other candidates, how much of it touches the image border and, if you drew one, how well
it fits the lasso. The contour is smoothed and simplified, and then checked for
self-intersections. If it has any, the tool retries with less simplification, and as a
last resort it uses the raw contour.

The JSON summary printed by `profiler trace` lists every candidate's score and metrics,
plus warnings such as "touches border" or "top two disagree". This lets a person, or an
LLM reading the summary and the `--debug` overlay, decide to pick another candidate
(`--pick grabcut`) or add a lasso.

## Lasso / hand-drawn hull
- **loose** (default): draw roughly around the object. Anything outside the lasso is
  treated as background, so distractors and shadows outside it are ignored.
- **tight**: draw a rough trace of the outline itself. The result snaps to real edges
  within `--snap-px` of your line. If your trace turns out to enclose a lot of background,
  the tool falls back to loose mode and tells you.

Command-line lasso: `--lasso lasso.json` (`[[x,y],...]` in image pixels) or
`--lasso "10,10 400,12 390,300 15,310"`.

## Centerlines (wires, springs, lines)
`--shape centerline` thins the detected shape down to a 1-pixel skeleton. It then traces
that skeleton into curves, removes short side-branches (burrs), and joins curves that run
through a junction. A spiral hairspring comes out as one continuous open curve. The
summary reports the measured line thickness. For very thin wire, raise
`--work-size` (the **Detail** setting) so the wire stays a few pixels wide.

## Scale & output
- `--scale-points "x1,y1 x2,y2" DISTANCE_MM`, `--width-mm`, `--height-mm` (real size of the
  result's bounding box), or `--mm-per-px`.
- SVG: one `<path>` per piece, sized in mm when a scale is given.
- DXF: R2010, Y-up, units mm (or unitless when no scale is given). Closed outlines go on
  layer `PROFILE`, open centerlines on layer `CENTERLINE`.
- `--curves`: smooth curves through the points (Bézier in SVG, SPLINE in DXF) instead of
  straight segments.

## Photo tips
- Shoot straight down at the part. Perspective and lens distortion are not corrected.
- Use a plain background that contrasts with the part, and even light to keep shadows soft.

## Tests
```bash
uv run pytest -q
```
Synthetic scenes (lighting gradient, shadow, texture, noise, distractors, sloppy traces),
each scored against ground-truth IoU.
