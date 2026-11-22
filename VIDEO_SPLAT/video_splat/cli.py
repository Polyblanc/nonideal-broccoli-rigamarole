"""Turn a single-camera video into a sequence of 3D Gaussian splats ("4D" playback).

Pipeline:
    1. ffmpeg      video  -> frames/%06d.png
    2. SHARP       frame  -> splats/%06d.ply   (feed-forward 3DGS, one pass per frame)
    3. gsplat      splat  -> renders/%06d.png  (one fixed camera path shared by all frames)
    4. ffmpeg      renders -> render.mp4

SHARP (https://github.com/apple/ml-sharp) predicts 3D Gaussians directly from one image, so
there is no intermediate point-cloud step. Each frame is predicted independently, so the
result is a per-frame splat sequence, not a temporally consistent dynamic model.

Every stage is resumable: re-running skips frames whose outputs already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

LOGGER = logging.getLogger("video_splat")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CAMERA_CHOICES = ["static", "swipe", "shake", "rotate", "rotate_forward"]


def find_video(path: Path) -> Path:
    """Return the video file at `path`, or the only video inside the folder `path`."""
    if path.is_file():
        return path
    videos = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(videos) != 1:
        found = ", ".join(str(v) for v in videos) or "none"
        raise SystemExit(f"Expected exactly one video under {path}, found: {found}")
    return videos[0]


def extract_frames(
    video: Path,
    frames_dir: Path,
    fps: float,
    max_side: int,
    start: float | None,
    duration: float | None,
) -> list[Path]:
    """Extract frames with ffmpeg (rotation metadata is applied by ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH.")

    existing = sorted(frames_dir.glob("*.png"))
    if existing:
        LOGGER.info("Reusing %d frames in %s (delete the folder to re-extract).", len(existing), frames_dir)
        return existing

    frames_dir.mkdir(parents=True, exist_ok=True)
    # Shrink only (min with the input size); SHARP resizes to 1536x1536 internally anyway.
    scale = (
        f"scale=w='min({max_side},iw)':h='min({max_side},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps},{scale}", "-start_number", "0", str(frames_dir / "%06d.png")]
    LOGGER.info("Extracting frames: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise SystemExit("ffmpeg produced no frames.")
    LOGGER.info("Extracted %d frames.", len(frames))
    return frames


def encode_video(renders_dir: Path, output: Path, fps: float) -> None:
    """Assemble rendered PNGs into an mp4."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", str(renders_dir / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    LOGGER.info("Wrote %s", output)


def run_splat_stage(args: argparse.Namespace, frames: list[Path], out_dir: Path) -> None:
    """Predict a splat per frame with SHARP and optionally render each with gsplat."""
    # Heavy imports live here so that `--frames-only` works without the ML stack.
    try:
        import imageio.v2 as iio
        import torch
        from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
        from sharp.models import PredictorParams, create_predictor
        from sharp.utils import camera, io
        from sharp.utils import gsplat as sharp_gsplat
        from sharp.utils.gaussians import load_ply, save_ply
    except ImportError as err:
        raise SystemExit(
            f"SHARP is not installed ({err}).\n"
            'Install it with:  uv tool install --force --editable "./VIDEO_SPLAT[sharp]"\n'
            "or use --frames-only to just extract frames."
        )

    if args.device == "default":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    LOGGER.info("SHARP device: %s", device)
    # SHARP warns about missing EXIF once per frame; video frames never have it.
    logging.getLogger("sharp.utils.io").setLevel(logging.ERROR)
    if args.focal_35mm is None:
        LOGGER.warning("No --focal-35mm given: SHARP will assume a 30mm-equivalent lens for every frame.")

    render = args.render
    if render and not torch.cuda.is_available():
        LOGGER.warning("gsplat rendering needs CUDA; skipping the render stage (PLYs are still written).")
        render = False
    if not args.save_ply and not render:
        raise SystemExit("Nothing to do: --no-save-ply with rendering disabled.")

    splats_dir = out_dir / "splats"
    renders_dir = out_dir / "renders"
    camera_path = out_dir / "camera.json"
    if args.save_ply:
        splats_dir.mkdir(exist_ok=True)
    if render:
        renders_dir.mkdir(exist_ok=True)

    def is_done(stem: str) -> bool:
        return (not args.save_ply or (splats_dir / f"{stem}.ply").exists()) and (
            not render or (renders_dir / f"{stem}.png").exists()
        )

    pending = [f for f in frames if not is_done(f.stem)]
    if not pending:
        LOGGER.info("All %d frames already processed.", len(frames))
    else:
        if args.checkpoint is None:
            LOGGER.info("Downloading/loading default SHARP checkpoint from %s", DEFAULT_MODEL_URL)
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        else:
            state_dict = torch.load(args.checkpoint, weights_only=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(device)

        # Fresh predictions are linearRGB. PLYs are exported as sRGB, but load_ply converts back to linearRGB.
        renderer = sharp_gsplat.GSplatRenderer(color_space="linearRGB")
        cameras: dict | None = json.loads(camera_path.read_text()) if render and camera_path.exists() else None

        t0 = time.time()
        for n, frame_path in enumerate(pending, start=1):
            image, _, f_px = io.load_rgb(frame_path)
            height, width = image.shape[:2]
            if args.focal_35mm is not None:
                f_px = io.convert_focallength(width, height, args.focal_35mm)

            ply_path = splats_dir / f"{frame_path.stem}.ply"
            if args.save_ply and ply_path.exists():
                # Predicted on an earlier run (e.g. with --no-render); reload instead of re-running.
                gaussians, _ = load_ply(ply_path)
                gaussians = gaussians.to(device)
            else:
                gaussians = predict_image(predictor, image, f_px, device)
                if args.save_ply:
                    save_ply(gaussians, f_px, (height, width), ply_path)

            if render:
                if cameras is None:
                    # One camera path for the whole clip, derived from the first frame only.
                    # SHARP derives its trajectory from each scene's depth statistics, so
                    # recomputing it per frame would make the virtual camera jitter.
                    if frame_path != frames[0]:
                        raise SystemExit(f"{camera_path} is missing but frame 0 is already rendered; delete renders/.")
                    cameras = build_cameras(args, gaussians, f_px, (width, height), camera, device)
                    camera_path.write_text(json.dumps(cameras))
                index = int(frame_path.stem) % len(cameras["extrinsics"])
                with torch.no_grad():
                    out = renderer(
                        gaussians.to(device),
                        extrinsics=torch.tensor(cameras["extrinsics"][index], dtype=torch.float32, device=device)[None],
                        intrinsics=torch.tensor(cameras["intrinsics"], dtype=torch.float32, device=device)[None],
                        image_width=cameras["width"],
                        image_height=cameras["height"],
                    )
                color = (out.color[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
                iio.imwrite(renders_dir / f"{frame_path.stem}.png", color.cpu().numpy())

            del gaussians
            elapsed = time.time() - t0
            LOGGER.info("[%d/%d] %s  (%.1fs/frame, ETA %.0f min)", n, len(pending), frame_path.name,
                        elapsed / n, elapsed / n * (len(pending) - n) / 60)  # fmt: skip

    if render and not args.no_video:
        encode_video(renders_dir, out_dir / "render.mp4", args.fps)


def build_cameras(args, gaussians, f_px, size, camera, device) -> dict:
    """Precompute extrinsics for every step of the chosen camera path (OpenCV convention)."""
    import torch

    width, height = size
    intrinsics = torch.tensor(
        [[f_px, 0, (width - 1) / 2.0, 0], [0, f_px, (height - 1) / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )  # fmt: skip
    model = camera.create_camera_model(gaussians.to("cpu"), intrinsics, resolution_px=(width, height))
    if args.camera == "static":
        eye_positions = [torch.zeros(3)]
    else:
        params = camera.TrajectoryParams(type=args.camera, num_steps=args.cycle_frames, num_repeats=1)
        eye_positions = camera.create_eye_trajectory(
            gaussians.to("cpu"), params, resolution_px=(width, height), f_px=f_px
        )
    infos = [model.compute(eye) for eye in eye_positions]
    return {
        "camera": args.camera,
        "width": infos[0].width,
        "height": infos[0].height,
        "intrinsics": infos[0].intrinsics[:3, :3].tolist(),
        "extrinsics": [info.extrinsics.tolist() for info in infos],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="video-splat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="Video file, or a folder containing exactly one video.")
    p.add_argument("-o", "--output", type=Path, default=Path("splat_out"), help="Output folder.")
    g = p.add_argument_group("frames")
    g.add_argument("--fps", type=float, default=5.0, help="Frames/second to extract (default 5; see README on disk use).")
    g.add_argument("--max-side", type=int, default=1536, help="Downscale frames so the long side is at most this.")
    g.add_argument("--start", type=float, help="Start offset in seconds (handy for short test clips).")
    g.add_argument("--duration", type=float, help="Only process this many seconds.")
    g.add_argument("--frames-only", action="store_true", help="Stop after frame extraction.")
    g = p.add_argument_group("SHARP")
    g.add_argument("-c", "--checkpoint", type=Path, help="Local sharp_*.pt (otherwise downloaded on first run).")
    g.add_argument("--device", default="default", help="cpu | mps | cuda (default: best available).")
    g.add_argument("--focal-35mm", type=float, help="35mm-equivalent focal length of the camera. "
                   "Video frames have no EXIF, so SHARP otherwise assumes 30mm.")  # fmt: skip
    g.add_argument("--save-ply", action=argparse.BooleanOptionalAction, default=True,
                   help="Write one ~65 MB .ply per frame (default on).")  # fmt: skip
    g = p.add_argument_group("gsplat rendering (CUDA only)")
    g.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--camera", choices=CAMERA_CHOICES, default="rotate_forward",
                   help="Virtual camera path; 'static' re-renders the original viewpoint.")  # fmt: skip
    g.add_argument("--cycle-frames", type=int, default=60, help="Frames per camera-path loop.")
    g.add_argument("--no-video", action="store_true", help="Keep the PNG renders, skip the mp4.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")

    video = find_video(args.input)
    LOGGER.info("Video: %s", video)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(video, args.output / "frames", args.fps, args.max_side, args.start, args.duration)
    if args.frames_only:
        return
    run_splat_stage(args, frames, args.output)
