"""Pipeline-level regression tests: target-aware stage caching and runs
that produce zero involvement events.

Stage 1 caching is keyed on the video/workflow only; stages 2-4 additionally
depend on WHO the target is (TargetSpec) and the identity/scoring/cut config.
Re-running the same video for a different kid must recompute stages 2+ instead
of silently delivering the previous kid's reel.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import pytest

from reelcut import pipeline
from reelcut.config import Paths, ReelcutConfig
from reelcut.types import BBox, TargetSpec
from reelcut.workflow_client import StubWorkflowClient

JERSEY = "10"


def _spec_for_track(
    cfg: ReelcutConfig, video: Path, track_id: int, jersey: str = JERSEY
) -> TargetSpec:
    """Probe the stub for a seed box on ``track_id`` (no hardcoded geometry)."""
    client = StubWorkflowClient(cfg.seed, jersey)
    for frame in itertools.islice(client.run(video, cfg), 5):
        for p in frame.players:
            if p.track_id == track_id:
                return TargetSpec(
                    jersey=jersey,
                    team_color="blue",
                    seed_frame_index=frame.frame_index,
                    seed_box=p.bbox,
                )
    raise AssertionError(f"stub never produced track {track_id}")


def _seeded_track_id(result: pipeline.PipelineResult) -> int:
    return next(
        lt.tracklet.track_id
        for lt in result.labeled
        if lt.evidence.get("seed") == 1.0
    )


def _boom(*args, **kwargs):
    raise AssertionError("stage cache was ignored")


# --------------------------------------------------------------------------- #
# target-aware cache keying
# --------------------------------------------------------------------------- #

def test_target_fingerprint_sensitive_to_spec_and_config() -> None:
    cfg = ReelcutConfig()
    s1 = TargetSpec("10", "blue", 0, BBox(0, 0, 10, 20))
    s2 = TargetSpec("23", "blue", 0, BBox(0, 0, 10, 20))
    s3 = TargetSpec("10", "red", 0, BBox(0, 0, 10, 20))
    s4 = TargetSpec("10", "blue", 5, BBox(1, 0, 10, 20))
    fingerprints = {pipeline._target_fingerprint(s, cfg) for s in (s1, s2, s3, s4)}
    assert len(fingerprints) == 4
    # deterministic across equal inputs, sensitive to config tuning
    assert pipeline._target_fingerprint(s1, cfg) == pipeline._target_fingerprint(
        s1, ReelcutConfig()
    )
    assert pipeline._target_fingerprint(s1, cfg) != pipeline._target_fingerprint(
        s1, replace(cfg, event_threshold=0.5)
    )


def test_stub_workflow_ref_includes_target_jersey() -> None:
    # Stub stage-1 output (OCR texts, confuser pool) depends on the jersey,
    # so the inference cache key must too.
    cfg = ReelcutConfig()
    assert pipeline._workflow_ref(cfg, StubWorkflowClient(1, "10")) != (
        pipeline._workflow_ref(cfg, StubWorkflowClient(1, "23"))
    )


def test_rerun_with_different_target_recomputes_stages_2_plus(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    video = out / "no_such_game.mp4"
    cfg = ReelcutConfig()
    paths = Paths(out_dir=out)

    spec_a = _spec_for_track(cfg, video, 7)
    first = pipeline.run_pipeline(
        video, spec_a, cfg, paths, StubWorkflowClient(cfg.seed, JERSEY)
    )
    assert _seeded_track_id(first) == 7

    # Same video, same jersey (same cache key), DIFFERENT kid (track 3).
    spec_b = _spec_for_track(cfg, video, 3)
    client_b = StubWorkflowClient(cfg.seed, JERSEY)
    client_b.run = _boom  # type: ignore[method-assign]  # stage 1 from cache
    second = pipeline.run_pipeline(video, spec_b, cfg, paths, client_b)

    # Stages 2+ must be recomputed for the new target, not reused from run A.
    assert _seeded_track_id(second) == 3
    assert second.identity != first.identity


def test_rerun_with_same_target_reuses_stages_2_plus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    video = out / "no_such_game.mp4"
    cfg = ReelcutConfig()
    paths = Paths(out_dir=out)

    spec = _spec_for_track(cfg, video, 7)
    first = pipeline.run_pipeline(
        video, spec, cfg, paths, StubWorkflowClient(cfg.seed, JERSEY)
    )

    # Identical spec/config: every stage must load from cache.
    monkeypatch.setattr(pipeline.stitching, "build_tracklets", _boom)
    monkeypatch.setattr(pipeline.scoring, "score_timeline", _boom)
    monkeypatch.setattr(pipeline.clipcutter, "plan_clips", _boom)
    client = StubWorkflowClient(cfg.seed, JERSEY)
    client.run = _boom  # type: ignore[method-assign]
    second = pipeline.run_pipeline(video, spec, cfg, paths, client)

    assert second.clips == first.clips
    assert second.identity == first.identity


# --------------------------------------------------------------------------- #
# zero involvement events with a real source video
# --------------------------------------------------------------------------- #

def test_no_events_completes_without_reel(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    out = tmp_path / "out"
    out.mkdir()
    video = out / "game.mp4"
    vw = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    if not vw.isOpened():
        pytest.skip("cv2.VideoWriter cannot encode mp4 here")
    for i in range(20):
        vw.write(np.full((64, 64, 3), (i * 10) % 255, dtype=np.uint8))
    vw.release()
    if video.stat().st_size == 0:
        pytest.skip("cv2.VideoWriter produced an empty mp4")

    # No smoothed score can clear an impossible threshold (scores are <= 1.0):
    # a kid benched for the recorded stretch is normal, not a crash.
    cfg = ReelcutConfig(event_threshold=1.5)
    spec = _spec_for_track(cfg, video, 7)
    paths = Paths(out_dir=out)
    result = pipeline.run_pipeline(
        video, spec, cfg, paths, StubWorkflowClient(cfg.seed, JERSEY)
    )

    assert result.events == []
    assert result.clips == []
    assert paths.highlights_json.is_file()
    assert not paths.highlights_mp4.exists()
