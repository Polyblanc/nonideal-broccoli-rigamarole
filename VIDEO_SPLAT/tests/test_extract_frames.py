import shutil
import subprocess
from pathlib import Path

import pytest

from video_splat import cli as v2s

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    folder = tmp_path / "upload"
    folder.mkdir()
    path = folder / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
         "-t", "4", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_find_video_in_folder(clip: Path):
    assert v2s.find_video(clip.parent) == clip


def test_find_video_rejects_ambiguous_folder(clip: Path):
    (clip.parent / "other.mov").write_bytes(b"")
    with pytest.raises(SystemExit):
        v2s.find_video(clip.parent)


def test_extract_frames_count_and_downscale(clip: Path, tmp_path: Path):
    frames = v2s.extract_frames(clip, tmp_path / "frames", fps=5, max_side=1536, start=None, duration=None)
    assert len(frames) == 20
    assert frames[0].name == "000000.png"
    size = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(frames[0])],
        text=True,
    ).strip()
    assert size == "1536,864"


def test_extract_frames_is_resumable(clip: Path, tmp_path: Path):
    first = v2s.extract_frames(clip, tmp_path / "frames", fps=5, max_side=1536, start=None, duration=None)
    mtime = first[0].stat().st_mtime_ns
    second = v2s.extract_frames(clip, tmp_path / "frames", fps=5, max_side=1536, start=None, duration=None)
    assert second == first and second[0].stat().st_mtime_ns == mtime
