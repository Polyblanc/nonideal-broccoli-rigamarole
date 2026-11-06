# nonideal-broccoli-rigamarole
We all make mistakes sometimes, this one included

## profiler — photo → outline profile (SVG / DXF)

Give it a photo of a machine part, PCB, tool or side profile and it returns one
closed, non-self-intersecting outline. Only the outer profile is kept: holes are filled.

```bash
pip install -r requirements.txt

# CLI
python -m profiler trace part.jpg -o part.svg -o part.dxf --width-mm 85 --debug overlay.png

# Browser UI with lasso tool (http://127.0.0.1:8765)
python -m profiler serve
```

### How it guesses
It runs several classic OpenCV segmentations and scores each one, rather than relying on a single threshold:

| candidate | good for |
|---|---|
| `otsu`    | high-contrast silhouettes (dark part on paper, backlit) |
| `color`   | coloured objects on a plain background whose brightness is similar (green PCB on a blue mat) |
| `edges`   | sharp-edged parts on a uniform background |
| `grabcut` | textured or cluttered backgrounds, specular highlights |
| `alpha`   | PNGs that already have a transparent background (used directly) |

Every mask is then cleaned up: holes are filled, only the largest blob is kept, spurs and
notches thinner than `--min-feature` are removed, and pixel jaggies are rounded off. Each
candidate is scored on how well it lines up with image edges, how much it agrees with the
other candidates, how much of it touches the image border and, if you drew one, how well
it fits the lasso. The contour is smoothed and simplified, and then checked for
self-intersections. If it has any, the tool retries with less simplification, and as a
last resort it uses the raw contour.

The JSON summary printed by the CLI lists every candidate's score and metrics, plus
warnings such as "touches border" or "top two disagree". This lets a person, or an LLM
reading the summary and the `--debug` overlay, decide to pick another candidate
(`--pick grabcut`) or add a lasso.

### Lasso / hand-drawn hull
- **loose** (default): draw roughly around the object. Anything outside the lasso is
  treated as background, so distractors and shadows outside it are ignored.
- **tight**: draw a rough trace of the outline itself. The result snaps to real edges
  within `--snap-px` of your line. If your trace turns out to enclose a lot of background,
  the tool falls back to loose mode and tells you.

CLI lasso: `--lasso lasso.json` (`[[x,y],...]` in image pixels) or `--lasso "10,10 400,12 390,300 15,310"`.

### Scale & output
- `--width-mm`, `--height-mm` (real size of the outline's bounding box) or `--mm-per-px`.
- SVG: one closed `<path>`, sized in mm when a scale is given.
- DXF: R2010, one closed `LWPOLYLINE` on layer `PROFILE`, Y-up, units mm (or unitless when no scale is given).

Note: a photo only gives a true profile when the camera looks straight at the face being
traced. Perspective and lens distortion are not corrected.

### Tests
`python -m pytest tests -q`: synthetic scenes (lighting gradient, shadow, texture, noise,
distractors, sloppy traces), each scored against ground-truth IoU.
