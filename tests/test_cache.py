"""Tests for reelcut.cache: video keying and gzip-JSON stage persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import ball, make_frame, player, walk_frames
from reelcut.cache import StageCache, video_cache_key
from reelcut.types import BallObs, BBox, FrameObservation, OcrRead, PlayerObs


@pytest.fixture
def video(tmp_path: Path) -> Path:
    p = tmp_path / "game.mp4"
    p.write_bytes(b"\x00\x01\x02fake-mp4-bytes" * 1000)
    return p


@pytest.fixture
def cache(tmp_path: Path) -> StageCache:
    return StageCache(tmp_path / "cache", "deadbeef")


def _frames() -> list[FrameObservation]:
    frames = walk_frames(7, 4, 100.0, 200.0, 12.0, 0.0, speed=60.0,
                         hsv=(0.1, 0.2, 0.7))
    frames.append(
        make_frame(
            4,
            players=[player(7, 148.0, 200.0), player(9, 400.0, 300.0, cls="referee")],
            ball=ball(160.0, 240.0),
            ocr=[OcrRead(track_id=7, text="10", confidence=0.8)],
        )
    )
    return frames


# --------------------------------------------------------------------------- #
# video_cache_key
# --------------------------------------------------------------------------- #

def test_key_stable_across_calls(video: Path) -> None:
    k1 = video_cache_key(video, 5.0, "ws/reelcut-tracking@3")
    k2 = video_cache_key(video, 5.0, "ws/reelcut-tracking@3")
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


def test_key_changes_with_fps(video: Path) -> None:
    assert video_cache_key(video, 5.0, "wf@1") != video_cache_key(video, 2.5, "wf@1")


def test_key_changes_with_workflow_ref(video: Path) -> None:
    assert video_cache_key(video, 5.0, "wf@1") != video_cache_key(video, 5.0, "wf@2")


def test_key_changes_with_content(video: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different bytes" + video.read_bytes())
    assert video_cache_key(video, 5.0, "wf@1") != video_cache_key(other, 5.0, "wf@1")


# --------------------------------------------------------------------------- #
# StageCache
# --------------------------------------------------------------------------- #

def test_roundtrip_frame_observations(cache: StageCache) -> None:
    frames = _frames()
    path = cache.save(1, "frames", frames)
    assert path.is_file()
    assert path.name == "stage1_frames.json.gz"

    loaded = cache.load(1, "frames")
    assert loaded == frames

    # dataclass types (not plain dicts/lists) survive the roundtrip
    assert isinstance(loaded, list)
    for fr in loaded:
        assert isinstance(fr, FrameObservation)
        assert isinstance(fr.players, tuple)
        for p in fr.players:
            assert isinstance(p, PlayerObs)
            assert isinstance(p.bbox, BBox)
    last = loaded[-1]
    assert isinstance(last.ball, BallObs)
    assert isinstance(last.ocr, tuple)
    assert isinstance(last.ocr[0], OcrRead)
    assert last.players[0].torso_hsv is None
    assert loaded[0].players[0].torso_hsv == (0.1, 0.2, 0.7)


def test_has(cache: StageCache) -> None:
    assert not cache.has(1, "frames")
    cache.save(1, "frames", [])
    assert cache.has(1, "frames")
    assert not cache.has(2, "frames")
    assert not cache.has(1, "other")


def test_invalidate_from_keeps_earlier_stages(cache: StageCache) -> None:
    cache.save(1, "frames", [])
    cache.save(2, "tracklets", [])
    cache.save(2, "identity", [])
    cache.save(3, "scores", [])
    cache.save(4, "clips", [])

    cache.invalidate_from(2)

    assert cache.has(1, "frames")
    assert not cache.has(2, "tracklets")
    assert not cache.has(2, "identity")
    assert not cache.has(3, "scores")
    assert not cache.has(4, "clips")


def test_invalidate_from_is_idempotent(cache: StageCache) -> None:
    cache.save(1, "frames", [])
    cache.invalidate_from(2)
    cache.invalidate_from(2)  # nothing to delete: must not raise
    assert cache.has(1, "frames")
