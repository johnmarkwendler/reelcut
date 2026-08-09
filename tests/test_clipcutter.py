"""Unit tests for clipcutter: plan_clips invariants, JSON schema, cut_reel."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from reelcut import clipcutter
from reelcut.clipcutter import cut_reel, plan_clips, write_highlights_json
from reelcut.config import ReelcutConfig
from reelcut.types import Clip, IdentityPoint, InvolvementEvent, ScorePoint

CFG = ReelcutConfig()  # pad 4, merge gap 3, min 5, max 25
EPS = 1e-6


def ev(start: float, end: float, peak: float = 0.8, mean: float = 0.5,
       tags: tuple[str, ...] = ("near_ball",)) -> InvolvementEvent:
    return InvolvementEvent(start_s=start, end_s=end, peak_score=peak,
                            mean_score=mean, tags=tags)


def flat_scores(t0: float, t1: float, step: float = 0.2,
                val: float = 0.5) -> list[ScorePoint]:
    n = int(round((t1 - t0) / step))
    return [ScorePoint(timestamp_s=t0 + i * step, score=val) for i in range(n + 1)]


def flat_identity(duration: float, step: float = 1.0,
                  conf: float = 1.0) -> list[IdentityPoint]:
    n = int(round(duration / step))
    return [IdentityPoint(timestamp_s=i * step, confidence=conf)
            for i in range(n + 1)]


def check_invariants(clips: list[Clip], duration: float,
                     cfg: ReelcutConfig = CFG) -> None:
    assert clips == sorted(clips, key=lambda c: c.start_s)
    for c in clips:
        assert -EPS <= c.start_s < c.end_s <= duration + EPS
        assert c.end_s - c.start_s <= cfg.clip_max_s + EPS
        assert c.reasons
    for a, b in zip(clips, clips[1:]):
        assert b.start_s >= a.end_s - EPS  # non-overlapping (touching ok)


# --------------------------------------------------------------------------- #
# plan_clips
# --------------------------------------------------------------------------- #

def test_empty_events_returns_empty() -> None:
    assert plan_clips([], flat_scores(0, 100), flat_identity(100), 100.0, CFG) == []


def test_padding_applied() -> None:
    clips = plan_clips([ev(10, 12)], flat_scores(0, 100), flat_identity(100),
                       100.0, CFG)
    assert len(clips) == 1
    assert clips[0].start_s == pytest.approx(10 - CFG.clip_pad_s)
    assert clips[0].end_s == pytest.approx(12 + CFG.clip_pad_s)
    assert clips[0].score == pytest.approx(0.8)
    check_invariants(clips, 100.0)


def test_clamped_at_video_edges() -> None:
    clips = plan_clips([ev(1, 3), ev(27, 29)], flat_scores(0, 30),
                       flat_identity(30), 30.0, CFG)
    assert len(clips) == 2
    assert clips[0].start_s == pytest.approx(0.0)
    assert clips[0].end_s == pytest.approx(7.0)
    assert clips[1].start_s == pytest.approx(23.0)
    assert clips[1].end_s == pytest.approx(30.0)
    check_invariants(clips, 30.0)


def test_merge_below_gap_unions_tags_and_takes_max_score() -> None:
    events = [ev(10, 12, peak=0.6, tags=("near_ball",)),
              ev(21, 23, peak=0.9, tags=("sprint",))]
    # padded: (6,16) and (17,27) -> gap 1 < 3 -> merged
    clips = plan_clips(events, flat_scores(0, 100), flat_identity(100),
                       100.0, CFG)
    assert len(clips) == 1
    assert clips[0].start_s == pytest.approx(6.0)
    assert clips[0].end_s == pytest.approx(27.0)
    assert clips[0].score == pytest.approx(0.9)
    assert clips[0].reasons == ("near_ball", "sprint")
    check_invariants(clips, 100.0)


def test_no_merge_at_or_above_gap() -> None:
    # padded: (6,16) and (19,29) -> gap exactly 3 -> NOT merged (strict <)
    clips = plan_clips([ev(10, 12), ev(23, 25)], flat_scores(0, 100),
                       flat_identity(100), 100.0, CFG)
    assert len(clips) == 2
    assert clips[1].start_s - clips[0].end_s == pytest.approx(3.0)
    check_invariants(clips, 100.0)


def test_min_extend_symmetric() -> None:
    cfg = ReelcutConfig(clip_pad_s=0.5)
    clips = plan_clips([ev(10.0, 10.4)], flat_scores(0, 100),
                       flat_identity(100), 100.0, cfg)
    assert len(clips) == 1
    c = clips[0]
    assert c.end_s - c.start_s == pytest.approx(cfg.clip_min_s)
    # center preserved: padded span was (9.5, 10.9), center 10.2
    assert (c.start_s + c.end_s) / 2 == pytest.approx(10.2)
    check_invariants(clips, 100.0, cfg)


def test_min_extend_clamped_at_start_pushes_right() -> None:
    cfg = ReelcutConfig(clip_pad_s=0.5)
    clips = plan_clips([ev(0.2, 0.4)], flat_scores(0, 100),
                       flat_identity(100), 100.0, cfg)
    assert len(clips) == 1
    assert clips[0].start_s == pytest.approx(0.0)
    assert clips[0].end_s == pytest.approx(cfg.clip_min_s)
    check_invariants(clips, 100.0, cfg)


def test_min_extend_clamped_at_end_pushes_left() -> None:
    cfg = ReelcutConfig(clip_pad_s=0.5)
    clips = plan_clips([ev(29.4, 29.6)], flat_scores(0, 30),
                       flat_identity(30), 30.0, cfg)
    assert len(clips) == 1
    assert clips[0].end_s == pytest.approx(30.0)
    assert clips[0].start_s == pytest.approx(30.0 - cfg.clip_min_s)
    check_invariants(clips, 30.0, cfg)


def test_video_shorter_than_min_yields_whole_video() -> None:
    clips = plan_clips([ev(1, 2)], flat_scores(0, 3), flat_identity(3),
                       3.0, ReelcutConfig(clip_pad_s=0.1))
    assert len(clips) == 1
    assert clips[0].start_s == pytest.approx(0.0)
    assert clips[0].end_s == pytest.approx(3.0)


def test_remerge_after_min_extension_overlap() -> None:
    cfg = ReelcutConfig(clip_pad_s=0.5)
    # padded: (9.5,10.7) and (15.5,16.7): gap 4.8 >= 3, no first merge;
    # min-extended to (7.6,12.6) and (13.6,18.6): gap 1.0 < 3 -> re-merged.
    clips = plan_clips([ev(10.0, 10.2, tags=("touch",)),
                        ev(16.0, 16.2, tags=("sprint",))],
                       flat_scores(0, 100), flat_identity(100), 100.0, cfg)
    assert len(clips) == 1
    assert clips[0].start_s == pytest.approx(7.6)
    assert clips[0].end_s == pytest.approx(18.6)
    assert clips[0].reasons == ("sprint", "touch")
    check_invariants(clips, 100.0, cfg)


def test_max_split_at_lowest_valley() -> None:
    # padded span (6, 44), length 38 > 25; valley at t=25.
    scores = [
        ScorePoint(timestamp_s=p.timestamp_s,
                   score=(0.05 if abs(p.timestamp_s - 25.0) < 0.01 else 0.5))
        for p in flat_scores(0, 50)
    ]
    clips = plan_clips([ev(10, 40)], scores, flat_identity(50), 50.0, CFG)
    assert len(clips) == 2
    assert clips[0].start_s == pytest.approx(6.0)
    assert clips[0].end_s == pytest.approx(25.0)
    assert clips[1].start_s == pytest.approx(25.0)
    assert clips[1].end_s == pytest.approx(44.0)
    check_invariants(clips, 50.0)
    for c in clips:
        assert CFG.clip_min_s - EPS <= c.end_s - c.start_s <= CFG.clip_max_s + EPS


def test_split_point_at_least_min_from_edges() -> None:
    # Global valley at t=8 is < clip_min_s from the span start (6) and must be
    # ignored; the eligible valley is at t=30.
    scores = []
    for p in flat_scores(0, 50):
        s = 0.5
        if abs(p.timestamp_s - 8.0) < 0.01:
            s = 0.01
        elif abs(p.timestamp_s - 30.0) < 0.01:
            s = 0.1
        scores.append(ScorePoint(timestamp_s=p.timestamp_s, score=s))
    clips = plan_clips([ev(10, 40)], scores, flat_identity(50), 50.0, CFG)
    assert len(clips) == 2
    assert clips[0].end_s == pytest.approx(30.0)
    check_invariants(clips, 50.0)


def test_max_split_recursive_pieces_within_bounds() -> None:
    # One long event covering nearly the whole video -> recursive splitting.
    duration = 200.0
    clips = plan_clips([ev(0, 196)], flat_scores(0, duration),
                       flat_identity(duration), duration, CFG)
    assert len(clips) >= 2
    check_invariants(clips, duration)
    for c in clips:
        assert CFG.clip_min_s - EPS <= c.end_s - c.start_s <= CFG.clip_max_s + EPS
    # pieces tile the padded span with no loss
    assert clips[0].start_s == pytest.approx(0.0)
    assert clips[-1].end_s == pytest.approx(duration)
    for a, b in zip(clips, clips[1:]):
        assert b.start_s == pytest.approx(a.end_s)


def test_confidence_is_mean_identity_over_span() -> None:
    identity = [IdentityPoint(timestamp_s=float(t),
                              confidence=(1.0 if t < 10 else 0.5))
                for t in range(101)]
    clips = plan_clips([ev(10, 12)], flat_scores(0, 100), identity, 100.0, CFG)
    assert len(clips) == 1
    # span (6,16): points 6..9 conf 1.0, points 10..16 conf 0.5
    expected = (4 * 1.0 + 7 * 0.5) / 11
    assert clips[0].confidence == pytest.approx(expected)


def test_confidence_defaults_when_no_identity_points() -> None:
    clips = plan_clips([ev(10, 12)], flat_scores(0, 100), [], 100.0, CFG)
    assert clips[0].confidence == pytest.approx(1.0)


def test_reasons_fallback_involvement_when_no_tags() -> None:
    clips = plan_clips([ev(10, 12, tags=())], flat_scores(0, 100),
                       flat_identity(100), 100.0, CFG)
    assert clips[0].reasons == ("involvement",)


def test_many_events_sorted_nonoverlapping() -> None:
    events = [ev(60, 62), ev(10, 12), ev(35, 37), ev(90, 91), ev(11, 14)]
    clips = plan_clips(events, flat_scores(0, 120), flat_identity(120),
                       120.0, CFG)
    assert clips
    check_invariants(clips, 120.0)
    for c in clips:
        assert c.end_s - c.start_s >= CFG.clip_min_s - EPS


# --------------------------------------------------------------------------- #
# write_highlights_json
# --------------------------------------------------------------------------- #

def test_write_highlights_json_roundtrip(tmp_path: Path) -> None:
    clips = [
        Clip(start_s=1.2345, end_s=7.891, score=0.5678,
             reasons=("near_ball", "sprint"), confidence=0.8712),
        Clip(start_s=20.0, end_s=30.0, score=1.0,
             reasons=("involvement",), confidence=1.0),
    ]
    out = tmp_path / "highlights.json"
    src = tmp_path / "game.mp4"
    write_highlights_json(clips, out, src, CFG)

    with out.open() as f:
        data = json.load(f)
    assert set(data) == {"source", "generated_by", "clips"}
    assert data["source"] == str(src)
    assert data["generated_by"] == "reelcut"
    assert len(data["clips"]) == 2
    first = data["clips"][0]
    assert set(first) == {"start_s", "end_s", "score", "reasons", "confidence"}
    assert first["start_s"] == 1.23
    assert first["end_s"] == 7.89
    assert first["score"] == 0.57
    assert first["confidence"] == 0.87
    assert first["reasons"] == ["near_ball", "sprint"]
    assert data["clips"][1]["end_s"] == 30.0


def test_write_highlights_json_empty_clips(tmp_path: Path) -> None:
    out = tmp_path / "highlights.json"
    write_highlights_json([], out, tmp_path / "game.mp4", CFG)
    with out.open() as f:
        data = json.load(f)
    assert data["clips"] == []


# --------------------------------------------------------------------------- #
# cut_reel — needs a working cv2.VideoWriter and an ffmpeg binary
# --------------------------------------------------------------------------- #

def _cv2_writer_works() -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "probe.mp4")
        try:
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
            if not vw.isOpened():
                return False
            vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
            vw.release()
            return os.path.getsize(p) > 0
        except Exception:
            return False


def _resolve_ffmpeg() -> tuple[str | None, bool]:
    """(ffmpeg path or None, whether ffmpeg_exe needs monkeypatching)."""
    try:
        return clipcutter.ffmpeg_exe(), False
    except NotImplementedError:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe, True
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe(), True
    except Exception:
        return None, True


def _write_synthetic_video(path: Path, seconds: float = 10.0,
                           fps: float = 30.0) -> None:
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (64, 64))
    assert vw.isOpened()
    for i in range(int(seconds * fps)):
        frame = np.full((64, 64, 3), (i * 7) % 256, dtype=np.uint8)
        vw.write(frame)
    vw.release()


@pytest.mark.skipif(not _cv2_writer_works(),
                    reason="cv2.VideoWriter cannot encode mp4 here")
def test_cut_reel_two_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    exe, needs_patch = _resolve_ffmpeg()
    if exe is None:
        pytest.skip("no ffmpeg binary available")
    if needs_patch:
        monkeypatch.setattr(clipcutter, "ffmpeg_exe", lambda: exe)

    src = tmp_path / "game.mp4"
    _write_synthetic_video(src, seconds=10.0)

    clips = [
        Clip(start_s=1.0, end_s=3.0, score=0.9, reasons=("near_ball",)),
        Clip(start_s=5.0, end_s=8.0, score=0.8, reasons=("sprint",)),
    ]
    out = tmp_path / "highlights.mp4"
    work = tmp_path / "work"
    cut_reel(clips, src, out, work)

    assert out.is_file() and out.stat().st_size > 0
    assert (work / "seg_000.mp4").is_file()
    assert (work / "seg_001.mp4").is_file()
    assert (work / "concat.txt").is_file()

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    assert fps > 0
    duration = frames / fps
    assert abs(duration - 5.0) <= 1.5


def _video_duration_s(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    assert fps > 0
    return frames / fps


@pytest.mark.skipif(not _cv2_writer_works(),
                    reason="cv2.VideoWriter cannot encode mp4 here")
def test_cut_reel_no_gop_preroll_on_long_gop_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: '-ss -c copy' segment cuts kept packets back to the previous
    # keyframe (hidden by an mp4 edit list the copy concat then discarded),
    # splicing up to a GOP of unrequested footage per clip into the reel.
    import subprocess

    exe, needs_patch = _resolve_ffmpeg()
    if exe is None:
        pytest.skip("no ffmpeg binary available")
    if needs_patch:
        monkeypatch.setattr(clipcutter, "ffmpeg_exe", lambda: exe)

    raw = tmp_path / "raw.mp4"
    _write_synthetic_video(raw, seconds=35.0)
    src = tmp_path / "game.mp4"
    # One keyframe every 20 s: a keyframe-seek stream copy would drag in many
    # seconds of pre-roll per clip.
    subprocess.run(
        [exe, "-y", "-loglevel", "error", "-i", str(raw),
         "-c:v", "libx264", "-g", "600", "-keyint_min", "600",
         "-sc_threshold", "0", str(src)],
        check=True,
    )

    clips = [
        Clip(start_s=5.0, end_s=12.0, score=0.9, reasons=("near_ball",)),
        Clip(start_s=25.0, end_s=32.0, score=0.8, reasons=("sprint",)),
    ]
    out = tmp_path / "highlights.mp4"
    work = tmp_path / "work"
    cut_reel(clips, src, out, work)

    # Each segment and the reel must match the plan, not plan + GOP pre-roll.
    for i, clip in enumerate(clips):
        seg_dur = _video_duration_s(work / f"seg_{i:03d}.mp4")
        assert abs(seg_dur - (clip.end_s - clip.start_s)) <= 0.25, (i, seg_dur)
    reel_dur = _video_duration_s(out)
    assert abs(reel_dur - 14.0) <= 0.5, reel_dur


def test_cut_reel_empty_clips_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cut_reel([], tmp_path / "game.mp4", tmp_path / "out.mp4",
                 tmp_path / "work")
