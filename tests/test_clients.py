"""Tests for ffmpeg utilities and the Stage-1 workflow clients.

No network, no Roboflow account: the real InferencePipeline path is covered
only via its pure conversion helper (_observation_from_prediction).
"""
from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from reelcut.config import ReelcutConfig
from reelcut.ffmpeg import ffmpeg_exe, probe, run_ffmpeg
from reelcut.types import to_jsonable
from reelcut.workflow_client import (
    HIST_H_BINS,
    HIST_S_BINS,
    StubWorkflowClient,
    _canonical_class,
    _observation_from_prediction,
    torso_histogram,
)

CFG = ReelcutConfig()
MISSING = Path("definitely-not-a-real-video-file.mp4")


def stub_frames(seed: int = 1, n: int | None = None):
    it = StubWorkflowClient(seed=seed).run(MISSING, CFG)
    return list(it if n is None else itertools.islice(it, n))


# --------------------------------------------------------------------------- #
# torso_histogram
# --------------------------------------------------------------------------- #

def test_torso_histogram_solid_blue_lands_in_expected_bin() -> None:
    # BGR pure blue -> OpenCV HSV (120, 255, 255): H bin 120//15 = 8, S bin 3.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    hist = torso_histogram(frame, (50.0, 50.0, 45.0, 100.0))
    assert len(hist) == HIST_H_BINS * HIST_S_BINS == 48
    assert hist[8 * HIST_S_BINS + 3] == pytest.approx(1.0)
    assert sum(hist) == pytest.approx(1.0)


def test_torso_histogram_l1_normalized_on_noise() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    hist = torso_histogram(frame, (10.0, 10.0, 40.0, 90.0))
    assert len(hist) == 48
    assert all(v >= 0.0 for v in hist)
    assert sum(hist) == pytest.approx(1.0)


def test_torso_histogram_degenerate_region_is_zeros() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out_of_frame = torso_histogram(frame, (500.0, 500.0, 40.0, 80.0))
    zero_size = torso_histogram(frame, (10.0, 10.0, 0.0, 0.0))
    assert out_of_frame == tuple(0.0 for _ in range(48))
    assert zero_size == tuple(0.0 for _ in range(48))


# --------------------------------------------------------------------------- #
# StubWorkflowClient
# --------------------------------------------------------------------------- #

def test_stub_same_seed_is_deterministic() -> None:
    a = stub_frames(seed=42, n=20)
    b = stub_frames(seed=42, n=20)
    assert a == b


def test_stub_different_seed_differs() -> None:
    a = stub_frames(seed=42, n=20)
    b = stub_frames(seed=43, n=20)
    assert a != b


def test_stub_uses_cfg_seed_when_client_seed_is_none() -> None:
    a = list(itertools.islice(StubWorkflowClient().run(MISSING, CFG), 10))
    b = list(itertools.islice(StubWorkflowClient(seed=CFG.seed).run(MISSING, CFG), 10))
    assert a == b


def test_stub_shape_and_timeline() -> None:
    frames = stub_frames()
    # 120 s synthetic game sampled at cfg.sample_fps
    assert len(frames) == int(round(120.0 * CFG.sample_fps))
    for i, obs in enumerate(frames[:50]):
        assert obs.timestamp_s == pytest.approx(i / CFG.sample_fps)
        assert obs.frame_index == round(obs.timestamp_s * 30.0)  # 30 fps source
        assert (obs.frame_w, obs.frame_h) == (1280, 720)
        assert len(obs.players) == 22
        for p in obs.players:
            assert p.class_name == "player"
            assert 70.0 <= p.bbox.h <= 90.0
            assert -1.0 <= p.bbox.x and p.bbox.x2 <= 1281.0
            assert -1.0 <= p.bbox.y and p.bbox.y2 <= 721.0
            assert p.speed is not None and p.speed >= 0.0
            assert p.torso_hsv is not None and len(p.torso_hsv) == 48
            assert sum(p.torso_hsv) == pytest.approx(1.0)


def test_stub_speed_matches_actual_walk() -> None:
    frames = stub_frames(n=30)
    dt = 1.0 / CFG.sample_fps
    for prev, cur in zip(frames, frames[1:]):
        prev_by_id = {p.track_id: p for p in prev.players}
        for p in cur.players:
            q = prev_by_id.get(p.track_id)
            if q is None:
                continue
            moved = ((p.bbox.cx - q.bbox.cx) ** 2 + (p.bbox.cy - q.bbox.cy) ** 2) ** 0.5
            assert p.speed == pytest.approx(moved / dt, abs=1e-6)


def test_stub_emits_target_and_confuser_ocr() -> None:
    frames = stub_frames()
    target_reads = [r for f in frames for r in f.ocr if r.text == "10"]
    other_reads = [r for f in frames for r in f.ocr if r.text != "10"]
    assert 0.15 <= len(target_reads) / len(frames) <= 0.45   # ~30% of frames
    assert other_reads                                       # confusers exist
    for r in target_reads + other_reads:
        assert 0.5 <= r.confidence <= 0.95
    # target reads follow one player slot; confusers are different tracks
    frames_with_target = [
        f for f in frames if any(r.text == "10" for r in f.ocr)
    ]
    for f in frames_with_target[:20]:
        read = next(r for r in f.ocr if r.text == "10")
        assert read.track_id in {p.track_id for p in f.players}


def test_stub_ball_has_gaps() -> None:
    frames = stub_frames()
    missing = sum(1 for f in frames if f.ball is None)
    present = len(frames) - missing
    assert present > 0
    assert 0.10 <= missing / len(frames) <= 0.40             # ~25% gaps
    for f in frames:
        if f.ball is not None:
            assert 0.0 < f.ball.confidence <= 1.0


def test_stub_tracks_fragment_over_time() -> None:
    frames = stub_frames()
    first_ids = {p.track_id for p in frames[0].players}
    assert first_ids == set(range(1, 23))
    # ids are stable within the first ~20 s window ...
    early_ids = {p.track_id for f in frames[:99] for p in f.players}
    assert early_ids == first_ids
    # ... then ~30% get reassigned to fresh, previously unused ids
    later_ids = {p.track_id for p in frames[150].players}
    assert later_ids != first_ids
    assert max(later_ids) > 22
    assert len(later_ids) == 22
    all_ids = {p.track_id for f in frames for p in f.players}
    assert len(all_ids) > 22


def test_stub_team_colors_split_blue_red() -> None:
    frames = stub_frames(n=1)
    blue_bins = range(5 * HIST_S_BINS, 9 * HIST_S_BINS)      # H 75..135
    for p in frames[0].players:
        blue_mass = sum(p.torso_hsv[i] for i in blue_bins)
        if p.track_id <= 11:                                 # target's team
            assert blue_mass > 0.5
        else:
            assert blue_mass < 0.2


def test_stub_output_is_json_serializable() -> None:
    obs = stub_frames(n=1)[0]
    json.dumps(to_jsonable(obs))                             # must not raise


def test_stub_probes_existing_video(tiny_video: Path) -> None:
    frames = list(StubWorkflowClient(seed=1).run(tiny_video, CFG))
    # 12 frames @ 10 fps = 1.2 s -> 6 samples at 5 fps, sized like the video
    assert len(frames) == int(round(1.2 * CFG.sample_fps))
    assert (frames[0].frame_w, frames[0].frame_h) == (64, 48)


# --------------------------------------------------------------------------- #
# ffmpeg utilities
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()
    return path


def test_ffmpeg_exe_is_executable() -> None:
    import os

    exe = ffmpeg_exe()
    assert os.path.isfile(exe)
    assert os.access(exe, os.X_OK)


def test_probe_tiny_video(tiny_video: Path) -> None:
    meta = probe(tiny_video)
    assert meta.path == tiny_video
    assert (meta.width, meta.height) == (64, 48)
    assert meta.frame_count == 12
    assert meta.fps == pytest.approx(10.0, abs=0.5)
    assert meta.duration_s == pytest.approx(1.2, rel=0.1)


def test_probe_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        probe(MISSING)


def test_probe_unreadable_file_raises(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video at all")
    with pytest.raises(ValueError):
        probe(junk)


def test_run_ffmpeg_failure_captures_stderr(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_ffmpeg(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")])
    assert exc_info.value.returncode != 0
    assert exc_info.value.stderr                             # tail captured


# --------------------------------------------------------------------------- #
# RoboflowWorkflowClient pure helpers (no pipeline, no network)
# --------------------------------------------------------------------------- #

class FakeDetections:
    """Just enough of the sv.Detections surface for the converter."""

    def __init__(self, xyxy, tracker_id=None, confidence=None, data=None):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.tracker_id = tracker_id
        self.confidence = confidence
        self.data = data or {}

    def __len__(self) -> int:
        return len(self.xyxy)


def test_canonical_class_mapping() -> None:
    assert _canonical_class("player", CFG) == "player"
    assert _canonical_class("goalkeeper", CFG) == "goalkeeper"
    assert _canonical_class("goalie", CFG) == "goalkeeper"
    assert _canonical_class("referee", CFG) == "referee"
    assert _canonical_class("ref", CFG) == "referee"
    assert _canonical_class("something-else", CFG) == "player"
    assert _canonical_class(None, CFG) == "player"


def test_observation_from_prediction_full() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 255                                     # solid blue
    predictions = {
        "tracked_players": FakeDetections(
            xyxy=[
                [10, 10, 40, 90],       # player, has OCR text
                [50, 10, 80, 90],       # goalie -> goalkeeper
                [90, 10, 120, 90],      # ref -> referee
                [130, 40, 160, 41.5],   # too short (< 2% of frame h): dropped
                [130, 50, 160, 95],     # no tracker id: dropped
            ],
            tracker_id=np.array([5, 8, 9, 12, None], dtype=object),
            confidence=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            data={
                "class_name": np.array(["player", "goalie", "ref", "player", "player"]),
                "velocity": np.array(
                    [[3.0, 4.0], [np.nan, np.nan], [0, 0], [0, 0], [0, 0]]
                ),
                "speed": np.array([5.0, np.nan, 0.0, 0.0, 0.0]),
            },
        ),
        "ball": FakeDetections(
            xyxy=[[5, 5, 15, 15], [20, 20, 30, 30]],
            confidence=np.array([0.3, 0.9]),
        ),
        "jersey_texts": ["10", None],                        # shorter: no-reads
    }
    obs = _observation_from_prediction(
        predictions, frame, frame_index=90, timestamp_s=3.0, cfg=CFG
    )
    assert (obs.frame_index, obs.timestamp_s) == (90, 3.0)
    assert (obs.frame_w, obs.frame_h) == (200, 100)

    assert [p.track_id for p in obs.players] == [5, 8, 9]
    assert [p.class_name for p in obs.players] == ["player", "goalkeeper", "referee"]
    assert [p.confidence for p in obs.players] == pytest.approx([0.9, 0.8, 0.7])
    p0 = obs.players[0]
    assert p0.bbox.x == 10 and p0.bbox.y == 10 and p0.bbox.w == 30 and p0.bbox.h == 80
    assert p0.velocity == (3.0, 4.0) and p0.speed == 5.0
    assert obs.players[1].velocity is None and obs.players[1].speed is None
    for p in obs.players:
        assert p.torso_hsv is not None
        assert sum(p.torso_hsv) == pytest.approx(1.0)        # computed from frame

    assert obs.ocr == tuple([obs.ocr[0]])
    assert obs.ocr[0].track_id == 5 and obs.ocr[0].text == "10"

    assert obs.ball is not None
    assert obs.ball.confidence == pytest.approx(0.9)         # highest-confidence row
    assert (obs.ball.bbox.x, obs.ball.bbox.y) == (20.0, 20.0)


def test_observation_from_prediction_missing_keys() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    obs = _observation_from_prediction(
        {}, frame, frame_index=1, timestamp_s=0.033, cfg=CFG
    )
    assert obs.players == ()
    assert obs.ball is None
    assert obs.ocr == ()

    empty = {
        "tracked_players": FakeDetections(np.empty((0, 4))),
        "ball": None,
        "jersey_texts": None,
    }
    obs = _observation_from_prediction(
        empty, frame, frame_index=2, timestamp_s=0.066, cfg=CFG
    )
    assert obs.players == () and obs.ball is None and obs.ocr == ()


# --------------------------------------------------------------------------- #
# RoboflowWorkflowClient.run against a fake InferencePipeline (no network)
# --------------------------------------------------------------------------- #

class _FakeInferencePipeline:
    """Captures init_with_workflow kwargs and replays 1-based VideoFrames
    through the sink, like the real inference package does."""

    captured_kwargs: dict = {}

    def __init__(self, sink) -> None:
        self._sink = sink

    @classmethod
    def init_with_workflow(cls, **kwargs) -> "_FakeInferencePipeline":
        cls.captured_kwargs = kwargs
        return cls(kwargs["on_prediction"])

    def start(self, use_main_thread: bool = True) -> None:
        pass

    def join(self) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        for frame_id in (1, 2, 3):  # inference's counter is 1-based
            frame = type("VideoFrame", (), {"frame_id": frame_id, "image": image})()
            self._sink({}, frame)

    def terminate(self) -> None:
        pass


def test_roboflow_client_zero_bases_frame_ids_and_passes_model_id(
    tiny_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types as types_mod

    from reelcut import workflow_client as wc

    fake_inference = types_mod.ModuleType("inference")
    fake_inference.InferencePipeline = _FakeInferencePipeline
    monkeypatch.setitem(sys.modules, "inference", fake_inference)
    monkeypatch.setattr(wc, "_shim_torch_mps", lambda: None)

    client = wc.RoboflowWorkflowClient(
        api_key="k", workspace=CFG.workspace, workflow_id=CFG.workflow_id
    )
    frames = list(client.run(tiny_video, CFG))

    # VideoFrame.frame_id is 1-based; reelcut frame indices and timestamps are
    # 0-based, so the first sample must land at frame 0 / t = 0.
    source_fps = probe(tiny_video).fps
    assert [f.frame_index for f in frames] == [0, 1, 2]
    assert [f.timestamp_s for f in frames] == pytest.approx(
        [0.0, 1.0 / source_fps, 2.0 / source_fps]
    )

    # --model-id must actually reach the workflow, not just the cache key.
    kwargs = _FakeInferencePipeline.captured_kwargs
    assert kwargs["workflows_parameters"] == {"model_id": CFG.model_id}
    assert kwargs["max_fps"] == CFG.sample_fps
    assert kwargs["workspace_name"] == CFG.workspace
    assert kwargs["workflow_id"] == CFG.workflow_id
