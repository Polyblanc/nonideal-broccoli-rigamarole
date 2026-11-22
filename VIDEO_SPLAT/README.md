# VIDEO_SPLAT — single-camera video → Gaussian splats

Extracts frames from a video, predicts a 3D Gaussian splat for each frame with
[SHARP](https://github.com/apple/ml-sharp), and renders the sequence through
[gsplat](https://github.com/nerfstudio-project/gsplat) along a virtual camera path.

```
video ──ffmpeg──▶ frames/*.png ──SHARP──▶ splats/*.ply ──gsplat──▶ renders/*.png ──ffmpeg──▶ render.mp4
```

## What this does and doesn't do

- **No point-cloud stage.** SHARP regresses 3D Gaussians directly from one image
  (about 1.18M per frame), so it goes image to splat in one feed-forward pass.
- **gsplat is the renderer, not a trainer.** SHARP already depends on gsplat to rasterize.
  Optimizing splats with gsplat would need multiple viewpoints, and a fixed camera gives only one.
- **Per-frame, not truly 4D.** Frames are predicted independently, so the splats share no
  identity over time. Expect flicker in regions SHARP has to guess (disocclusions, thin
  structures). The output is a splat *sequence* that you can play back.
- **Nearby views only.** SHARP is trained for small camera motion around the input viewpoint.
  Large orbits will show stretched or empty regions.
- **License.** The SHARP weights are released for non-commercial research use only
  (see `LICENSE_MODEL` in the SHARP repo).

## Install (on the computer with the NVIDIA GPU)

You need [uv](../README.md#one-time-setup-on-your-computer) and **ffmpeg** (`sudo apt install ffmpeg`,
`brew install ffmpeg`, or `winget install ffmpeg`). Then, from the top of this repo:

```bash
uv tool install --editable "./VIDEO_SPLAT[sharp]"
```

The `[sharp]` part pulls SHARP from GitHub together with torch and gsplat. That is a multi-gigabyte
download. To install only the frame extractor, leave it out: `uv tool install --editable ./VIDEO_SPLAT`.

Things that commonly go wrong with the GPU parts (I could not test these on a GPU machine):

- **Windows:** the default PyPI build of torch is CPU-only. Check the first log lines of a run: if they say
  `SHARP device: cpu` on a machine with an NVIDIA card, you need a CUDA build of torch from
  <https://pytorch.org/get-started/locally/>.
- **gsplat** typically compiles its CUDA kernels the first time it renders, which needs the CUDA toolkit
  (`nvcc`) and a C++ compiler, and takes a few minutes. Prediction still works without CUDA (slowly on CPU);
  rendering is skipped with a warning.

## Use it

```bash
# Try a short clip first
video-splat path/to/folder -o out --start 10 --duration 6

# Full run
video-splat path/to/folder -o out --focal-35mm 26
```

Without installing: `cd VIDEO_SPLAT && uv run video-splat path/to/folder -o out --frames-only`
(add `--extra sharp` to `uv run` for the full pipeline).

`input` is a video file, or a folder containing exactly one video. Output layout:

| Path | Contents |
| --- | --- |
| `out/frames/%06d.png` | extracted frames |
| `out/splats/%06d.ply` | SHARP splats, OpenCV convention (x right, y down, z forward) |
| `out/camera.json` | the camera path shared by all frames |
| `out/renders/%06d.png`, `out/render.mp4` | gsplat renders |

Re-running resumes: finished frames are skipped. Delete `out/frames` to re-extract.

### Options worth knowing

- `--fps` (default 5). See the disk note below.
- `--focal-35mm`. Video frames carry no EXIF, so SHARP assumes 30mm. Set your camera's 35mm-equivalent
  focal length if you know it (phone main cameras are typically 24 to 28). It changes the scale
  and perspective of the splats.
- `--camera static|swipe|shake|rotate|rotate_forward`. `static` re-renders the original viewpoint,
  which makes a good sanity check against the source video. The others loop through
  `--cycle-frames` virtual camera positions, one per video frame.
- `--no-save-ply` renders without keeping splats. `--no-render` writes only PLYs (rendering also turns itself off without CUDA).
- `-c sharp_2572gikvuh.pt`: use a checkpoint you downloaded yourself instead of letting SHARP download one
  (<https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt>). This path loads the file with
  `weights_only=True`, which refuses to run code hidden in the file.

### Disk and time

Each frame's PLY is about 65 MB. 2 minutes at 5 fps is 600 frames, or about 40 GB. At 30 fps it would be
about 240 GB. Pick `--fps` accordingly, or use `--no-save-ply` if you only need the renders.
The SHARP paper reports under a second per image for prediction on a standard GPU. I haven't timed
this script end to end, and gsplat rendering adds to that. The script logs seconds per frame and an ETA as it runs,
so a short `--duration` trial gives a real estimate.

## Camera consistency

SHARP derives its camera trajectory from each scene's depth statistics. Recomputing it for every frame
would make the virtual camera drift as depth estimates fluctuate. The script computes the path once, from the first
frame, and reuses it, so it assumes a roughly static camera. If the camera moves a lot during the clip,
split the video and run each part separately.

## Tests

```bash
cd VIDEO_SPLAT && uv run pytest -q
```

These cover frame extraction only and need ffmpeg but not torch. The SHARP and gsplat stages need
a GPU and the model weights; they were only exercised with a stubbed predictor and renderer.
