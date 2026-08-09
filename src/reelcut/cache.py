"""Stage-output caching.

Each pipeline stage persists its output under
``<out>/cache/<video_key>/stageN_<name>.json.gz`` where ``video_key`` is a
sha256 over (first 8 MiB of the video, file size, sample fps, workflow ref).
Downstream stages re-run without re-inference; ``--force-stage N`` recomputes
stage N and everything after it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .types import from_jsonable, to_jsonable

_HEAD_BYTES = 8 * 1024 * 1024


def video_cache_key(video: Path, sample_fps: float, workflow_ref: str) -> str:
    """Hex digest identifying (video content, sampling, workflow version).

    Reads only the first 8 MiB of the file plus its size, so hashing a 90-min
    4K video stays instant. Must be deterministic across runs.
    """
    h = hashlib.sha256()
    if video.is_file():
        with video.open("rb") as f:
            h.update(f.read(_HEAD_BYTES))
        size = video.stat().st_size
    else:
        # Stub-without-video mode: no file to hash. Empty head + size 0 keeps
        # the key deterministic (the workflow ref still carries the stub seed).
        size = 0
    meta = f"|size={size}|fps={sample_fps!r}|workflow={workflow_ref}"
    h.update(meta.encode("utf-8"))
    return h.hexdigest()


class StageCache:
    """Load/save JSON-able payloads per (video_key, stage name)."""

    def __init__(self, cache_dir: Path, video_key: str) -> None:
        self.dir = cache_dir / video_key
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage_index: int, name: str) -> Path:
        return self.dir / f"stage{stage_index}_{name}.json.gz"

    def has(self, stage_index: int, name: str) -> bool:
        return self.path(stage_index, name).is_file()

    def save(self, stage_index: int, name: str, payload: Any) -> Path:
        """``payload`` is any structure accepted by ``types.to_jsonable``."""
        path = self.path(stage_index, name)
        data = json.dumps(to_jsonable(payload)).encode("utf-8")
        path.write_bytes(gzip.compress(data, mtime=0))
        return path

    def load(self, stage_index: int, name: str) -> Any:
        """Inverse of :meth:`save` (via ``types.from_jsonable``)."""
        raw = gzip.decompress(self.path(stage_index, name).read_bytes())
        return from_jsonable(json.loads(raw.decode("utf-8")))

    def invalidate_from(self, stage_index: int) -> None:
        """Delete cached outputs for ``stage_index`` and every later stage."""
        for i in range(stage_index, 10):
            for p in self.dir.glob(f"stage{i}_*"):
                p.unlink(missing_ok=True)
