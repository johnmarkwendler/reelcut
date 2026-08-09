"""ffmpeg/video utilities.

No system ffmpeg is assumed: resolve the binary from PATH first, else fall
back to the static binary bundled with imageio-ffmpeg. Probing uses cv2
(bundled with inference) — imageio-ffmpeg ships no ffprobe.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

_STDERR_TAIL_CHARS = 4000


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary (PATH lookup, else imageio_ffmpeg)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # deferred: only needed when PATH has no ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(video: Path) -> VideoMeta:
    """Read fps/frames/size via cv2.VideoCapture. Raises FileNotFoundError /
    ValueError on unreadable input."""
    if not video.exists():
        raise FileNotFoundError(f"no such video: {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"cv2 cannot open video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"unreadable video {video}: fps={fps} frames={frame_count} "
            f"size={width}x{height}"
        )
    return VideoMeta(
        path=video, fps=fps, frame_count=frame_count, width=width, height=height
    )


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with ``args`` (no leading binary), raising CalledProcessError
    with captured stderr tail on failure. Always pass -y and -loglevel error."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL_CHARS:]
        exc = subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=tail
        )
        exc.add_note(f"ffmpeg stderr (tail):\n{tail.strip()}")
        raise exc
