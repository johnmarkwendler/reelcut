"""End-to-end pipeline test: StubWorkflowClient, no video file on disk.

Runs every stage (INFER -> IDENTITY -> SCORE -> CUT) against the synthetic
stub game, then re-runs to prove the stage caches short-circuit inference.
The seed box is probed programmatically from the stub itself so the test
never hardcodes synthetic geometry.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from reelcut import pipeline
from reelcut.config import Paths, ReelcutConfig
from reelcut.types import Clip, TargetSpec
from reelcut.workflow_client import StubWorkflowClient

_STUB_TARGET_TRACK_ID = 7   # StubWorkflowClient._TARGET_SLOT 6 -> initial id 7
_JERSEY = "10"
_EPS = 1e-6


def _probe_target_spec(cfg: ReelcutConfig, missing_video: Path) -> TargetSpec:
    """Run the stub for a few frames and build a seed box on track 7."""
    client = StubWorkflowClient(cfg.seed, _JERSEY)
    for frame in itertools.islice(client.run(missing_video, cfg), 5):
        for p in frame.players:
            if p.track_id == _STUB_TARGET_TRACK_ID:
                return TargetSpec(
                    jersey=_JERSEY,
                    team_color="blue",
                    seed_frame_index=frame.frame_index,
                    seed_box=p.bbox,
                )
    raise AssertionError(
        f"stub never produced track {_STUB_TARGET_TRACK_ID} in its first frames"
    )


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory):
    """First full pipeline run into a fresh out dir; shared by the asserts."""
    out_dir = tmp_path_factory.mktemp("e2e_stub_out")
    missing_video = out_dir / "no_such_game.mp4"
    assert not missing_video.exists()

    cfg = ReelcutConfig()
    spec = _probe_target_spec(cfg, missing_video)
    paths = Paths(out_dir=out_dir)
    client = StubWorkflowClient(cfg.seed, _JERSEY)
    result = pipeline.run_pipeline(missing_video, spec, cfg, paths, client)
    return cfg, spec, paths, missing_video, result


def test_clips_non_empty(e2e) -> None:
    _, _, _, _, result = e2e
    assert result.clips
    assert all(isinstance(c, Clip) for c in result.clips)


def test_highlights_json_written_and_schema_valid(e2e) -> None:
    _, _, paths, missing_video, result = e2e
    assert paths.highlights_json.is_file()
    payload = json.loads(paths.highlights_json.read_text(encoding="utf-8"))

    assert payload["generated_by"] == "reelcut"
    assert payload["source"] == str(missing_video)
    assert isinstance(payload["clips"], list)
    assert len(payload["clips"]) == len(result.clips)
    for entry in payload["clips"]:
        assert set(entry) == {"start_s", "end_s", "score", "reasons", "confidence"}
        assert isinstance(entry["start_s"], (int, float))
        assert isinstance(entry["end_s"], (int, float))
        assert entry["end_s"] > entry["start_s"]
        assert isinstance(entry["score"], (int, float))
        assert 0.0 <= entry["confidence"] <= 1.0
        assert isinstance(entry["reasons"], list) and entry["reasons"]
        assert all(isinstance(r, str) for r in entry["reasons"])


def test_no_highlights_mp4_without_source_video(e2e) -> None:
    _, _, paths, _, _ = e2e
    assert not paths.highlights_mp4.exists()


def test_clip_lengths_within_bounds(e2e) -> None:
    cfg, _, _, _, result = e2e
    for c in result.clips:
        length = c.end_s - c.start_s
        assert length >= cfg.clip_min_s - _EPS, (c, length)
        assert length <= cfg.clip_max_s + _EPS, (c, length)
        assert math.isfinite(c.score)
        assert 0.0 <= c.confidence <= 1.0


def test_clips_sorted_non_overlapping(e2e) -> None:
    _, _, _, _, result = e2e
    clips = result.clips
    assert clips == sorted(clips, key=lambda c: c.start_s)
    for prev, nxt in itertools.pairwise(clips):
        assert prev.end_s <= nxt.start_s + _EPS, (prev, nxt)


def test_stage_caches_present(e2e) -> None:
    _, _, paths, _, _ = e2e
    key_dirs = [p for p in paths.cache_dir.iterdir() if p.is_dir()]
    assert len(key_dirs) == 1
    for stage in (1, 2, 3, 4):
        matches = list(key_dirs[0].glob(f"stage{stage}_*.json.gz"))
        assert matches, f"no cache file for stage {stage}"


def test_second_run_loads_entirely_from_cache(e2e) -> None:
    cfg, spec, paths, missing_video, first = e2e

    client = StubWorkflowClient(cfg.seed, _JERSEY)

    def _boom(video: Path, cfg: ReelcutConfig):
        raise AssertionError("client.run was called: stage 1 cache was ignored")

    client.run = _boom  # type: ignore[method-assign]

    second = pipeline.run_pipeline(missing_video, spec, cfg, paths, client)

    assert second.clips == first.clips
    assert second.events == first.events
    assert second.scores == first.scores
    assert second.identity == first.identity
    assert len(second.frames) == len(first.frames)
