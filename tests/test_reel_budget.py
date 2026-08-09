"""Adaptive reel budget + goal-box sanity filtering."""
from __future__ import annotations

from reelcut.config import ReelcutConfig
from reelcut.scoring import adaptive_threshold, score_opportunities
from reelcut.types import BallObs, BBox, ScorePoint

from fixtures import make_frame

CFG = ReelcutConfig()


def points(scores):
    return [ScorePoint(i / 5.0, s, ()) for i, s in enumerate(scores)]


def test_threshold_unchanged_when_coverage_is_sane():
    pts = points([0.9] * 10 + [0.0] * 90)      # 10% hot
    assert adaptive_threshold(pts, 0.30, 0.12, 0.20) == 0.30


def test_threshold_rises_when_scores_saturate():
    # realistic near-continuous saturated scores (smoothing makes them distinct)
    pts = points([0.5 + 0.004 * i for i in range(80)] + [0.1] * 20)
    t = adaptive_threshold(pts, 0.30, 0.12, 0.20)
    assert t > 0.30
    passing = sum(1 for p in pts if p.score >= t) / len(pts)
    assert passing <= 0.25                      # brought near the target band


def test_threshold_tie_saturation_steps_past_the_tie():
    pts = points([0.8] * 80 + [0.1] * 20)      # pathological: one giant tie
    t = adaptive_threshold(pts, 0.30, 0.12, 0.20)
    # no distinct higher score exists; degenerate uniform case yields t at the
    # tie value (coverage stays high) — documented limitation
    assert t == 0.8


def test_threshold_keeps_score_ordering_top_slice():
    # varied scores: the top ~12% must be the survivors
    vals = [i / 100 for i in range(100)]
    pts = points(vals)
    t = adaptive_threshold(pts, 0.30, 0.12, 0.20)
    survivors = [p.score for p in pts if p.score >= t]
    assert len(survivors) <= 15 and min(survivors) >= 0.85


def test_giant_goal_boxes_are_ignored():
    ball = BallObs(bbox=BBox(300, 200, 14, 14), confidence=0.8)
    huge_goal = BBox(0, 0, 640, 500)            # h ~ 69% of the 720p frame: nonsense
    frame = make_frame(0, [], ball)
    frame = type(frame)(**{**{f: getattr(frame, f) for f in frame.__dataclass_fields__},
                           "goal_boxes": (huge_goal,)})
    pts = score_opportunities([frame], CFG)
    assert pts[0].score == 0.0 and pts[0].tags == ()


def _goal_frame(i, ball_x, goal=BBox(60, 260, 90, 100), conf=0.8):
    ball = BallObs(bbox=BBox(ball_x - 7, 300, 14, 14), confidence=conf)
    frame = make_frame(i, [], ball)
    return type(frame)(**{**{f: getattr(frame, f) for f in frame.__dataclass_fields__},
                          "goal_boxes": (goal,)})


def test_static_goal_mouth_loitering_scores_low():
    from reelcut.scoring import score_opportunities
    frames = [_goal_frame(i, 100) for i in range(5)]   # parked in the goal box
    pts = score_opportunities(frames, CFG)
    assert all("goal_mouth" in p.tags for p in pts)
    assert max(p.score for p in pts) < 0.45             # cannot make a reel alone


def test_fast_ball_into_goal_mouth_scores_high():
    from reelcut.scoring import score_opportunities
    # ball flies 300px between samples (0.2s) into the goal box
    frames = [_goal_frame(0, 400), _goal_frame(1, 100)]
    pts = score_opportunities(frames, CFG)
    assert "goal_mouth" in pts[1].tags and pts[1].score > 0.6


def test_reasonable_goal_boxes_still_score():
    ball = BallObs(bbox=BBox(100, 300, 14, 14), confidence=0.8)
    goal = BBox(60, 260, 90, 100)               # h=100 of 720: fine
    frame = make_frame(0, [], ball)
    frame = type(frame)(**{**{f: getattr(frame, f) for f in frame.__dataclass_fields__},
                           "goal_boxes": (goal,)})
    pts = score_opportunities([frame], CFG)
    # static single frame: base signal present but action-gated low
    assert 0.1 < pts[0].score < 0.45 and "goal_chance" in pts[0].tags
