"""Unit tests for reelcut.scoring — synthetic timelines, no video."""
from __future__ import annotations

from fixtures import FPS, ball, make_frame, player
from reelcut.config import ReelcutConfig
from reelcut.scoring import (
    extract_events,
    interpolate_ball,
    score_timeline,
    smooth_scores,
)
from reelcut.types import BallObs, BBox, IdentityPoint, ScorePoint

CFG = ReelcutConfig()

# player(1, 300, 300) -> BBox(300, 300, 36, 80), center (318, 340), h = 80
PX, PY = 300.0, 300.0
CX, CY = 318.0, 340.0
H = 80.0


def ident(i: int, x: float = PX, y: float = PY, *, conf: float = 0.9,
          speed: float | None = 0.0) -> IdentityPoint:
    return IdentityPoint(timestamp_s=i / FPS, confidence=conf,
                         bbox=BBox(x, y, H * 0.45, H), track_id=1, speed=speed)


# --------------------------------------------------------------------------- #
# interpolate_ball
# --------------------------------------------------------------------------- #

def test_interpolate_ball_fills_short_gap():
    frames = [
        make_frame(i, [], None if i in (3, 4, 5) else ball(100 + 20 * i, 200))
        for i in range(8)
    ]
    out = interpolate_ball(frames, CFG.ball_gap_interp_max_s)

    for i in (3, 4, 5):
        assert out[i].ball is not None
        assert out[i].ball.interpolated
    # linear position: mid-gap ball lands on the straight line
    assert abs(out[4].ball.bbox.cx - (100 + 20 * 4)) < 1e-6
    assert abs(out[4].ball.bbox.cy - 200.0) < 1e-6
    # confidence decays toward the middle of the gap, below bounding confs
    assert out[4].ball.confidence < out[3].ball.confidence < 0.8
    assert out[4].ball.confidence < out[5].ball.confidence < 0.8
    # real detections untouched; input list unmodified
    assert not out[2].ball.interpolated and not out[6].ball.interpolated
    assert frames[4].ball is None


def test_interpolate_ball_leaves_long_and_unbounded_gaps():
    # i 0..1: leading gap (no left bound); i 3..11: 2.0s gap >= max 1.5s
    frames = [
        make_frame(i, [], ball(100 + 10 * i, 200) if i in (2, 12, 13) else None)
        for i in range(14)
    ]
    out = interpolate_ball(frames, CFG.ball_gap_interp_max_s)
    for i in (0, 1, *range(3, 12)):
        assert out[i].ball is None


# --------------------------------------------------------------------------- #
# score_timeline
# --------------------------------------------------------------------------- #

def test_possession_after_min_duration():
    # ball parked 30 px from a stationary target for 1.5 s
    n = 8
    frames = [make_frame(i, [player(1, PX, PY)], ball(CX + 30, CY))
              for i in range(n)]
    identity = [ident(i, conf=0.9, speed=0.0) for i in range(n)]
    pts = score_timeline(identity, frames, CFG)

    assert "possession" not in pts[0].tags  # not sustained yet
    assert "possession" in pts[-1].tags
    first = min(i for i, p in enumerate(pts) if "possession" in p.tags)
    # fires exactly when the run duration reaches possession_min_s
    assert abs(pts[first].timestamp_s - CFG.sport.possession_min_s) < 1e-6
    # proximity + possession, identity-weighted
    w = CFG.sport
    assert abs(pts[-1].score - 0.9 * (w.w_proximity + w.w_possession)) < 1e-9
    assert "near_ball" in pts[-1].tags


def test_touch_on_sharp_bounce():
    # ball rolls in along +x, hits the target, bounces straight back
    xs = [CX - 160 + 40 * i for i in range(5)] + [CX - 40, CX - 80]
    frames = [make_frame(i, [player(1, PX, PY)], ball(x, CY))
              for i, x in enumerate(xs)]
    identity = [ident(i, conf=1.0, speed=0.0) for i in range(len(xs))]
    pts = score_timeline(identity, frames, CFG)

    assert "touch" in pts[4].tags
    assert all("touch" not in p.tags for i, p in enumerate(pts) if i != 4)
    assert pts[4].score >= CFG.sport.w_touch
    assert all("possession" not in p.tags for p in pts)  # too brief


def test_touch_ignored_when_endpoint_interpolated():
    xs = [CX - 160 + 40 * i for i in range(5)] + [CX - 40, CX - 80]
    frames = [make_frame(i, [player(1, PX, PY)], ball(x, CY))
              for i, x in enumerate(xs)]
    # outgoing endpoint of the bounce is an interpolated ball -> no touch
    frames[5] = make_frame(5, [player(1, PX, PY)], BallObs(
        bbox=BBox(xs[5] - 8, CY - 8, 16, 16), confidence=0.5,
        interpolated=True))
    identity = [ident(i, conf=1.0, speed=0.0) for i in range(len(xs))]
    pts = score_timeline(identity, frames, CFG)

    assert all("touch" not in p.tags for p in pts)
    assert "near_ball" in pts[4].tags  # interpolated still counts for proximity


def test_sprint_without_ball():
    n = 6
    frames = [make_frame(i, [player(1, PX + 60 * i, PY)], None)
              for i in range(n)]
    # 300 px/s over an 80 px box = 3.75 heights/s > sprint_speed 2.5
    identity = [ident(i, PX + 60 * i, PY, conf=0.9, speed=300.0)
                for i in range(n)]
    pts = score_timeline(identity, frames, CFG)

    for p in pts:
        assert "sprint" in p.tags
        assert abs(p.score - 0.9 * CFG.sport.w_sprint) < 1e-9
        assert p.score > 0


def test_zero_identity_confidence_scores_zero():
    n = 5
    frames = [make_frame(i, [player(1, PX, PY)], ball(CX, CY))
              for i in range(n)]  # ball dead-center on the target
    identity = [ident(i, conf=0.0, speed=400.0) for i in range(n)]
    pts = score_timeline(identity, frames, CFG)

    assert len(pts) == n
    assert all(p.score == 0.0 for p in pts)


# --------------------------------------------------------------------------- #
# smooth_scores / extract_events
# --------------------------------------------------------------------------- #

def test_smoothing_flattens_single_sample_spike():
    n = 11
    pts = [ScorePoint(i / FPS, 1.0 if i == 5 else 0.0,
                      ("touch",) if i == 5 else ()) for i in range(n)]
    sm = smooth_scores(pts, CFG.smooth_window_s)

    assert len(sm) == n
    assert all(a.timestamp_s == b.timestamp_s for a, b in zip(sm, pts))
    assert sm[5].score < CFG.event_threshold  # spike averaged away
    # tags union over the centered window
    assert "touch" in sm[5].tags and "touch" in sm[3].tags
    assert extract_events(sm, CFG.event_threshold, CFG.event_min_s) == []


def test_extract_events_valleys_and_min_duration():
    scores = [0.5] * 6 + [0.1] * 3 + [0.5] * 2 + [0.1] * 3 + [0.6] * 6
    pts = [ScorePoint(i / FPS, s, ("near_ball",) if s >= 0.3 else ())
           for i, s in enumerate(scores)]
    events = extract_events(pts, 0.3, 1.0)

    # sub-threshold valleys split runs; the 2-sample run fails min duration
    assert len(events) == 2
    e0, e1 = events
    assert abs(e0.start_s - 0.0) < 1e-9 and abs(e0.end_s - 5 / FPS) < 1e-9
    assert abs(e1.start_s - 14 / FPS) < 1e-9 and abs(e1.end_s - 19 / FPS) < 1e-9
    assert e1.peak_score == 0.6
    assert abs(e1.mean_score - 0.6) < 1e-9
    assert e0.tags == ("near_ball",)


def test_extract_events_empty():
    assert extract_events([], 0.3, 1.0) == []


# --------------------------------------------------------------------------- #
# persist_goal_boxes
# --------------------------------------------------------------------------- #

def test_goal_boxes_persist_through_short_dropouts():
    from dataclasses import replace

    from reelcut.scoring import persist_goal_boxes

    goal = BBox(100, 100, 200, 150)
    frames = [make_frame(i) for i in range(20)]        # 4s at 5 fps
    frames[0] = replace(frames[0], goal_boxes=(goal,))
    out = persist_goal_boxes(frames, hold_s=2.5)
    # held for 2.5s (through i=12, t=2.4s), gone at i=13 (t=2.6s)
    assert out[12].goal_boxes == (goal,)
    assert out[13].goal_boxes == ()


def test_fresh_goal_detection_replaces_held_box():
    from dataclasses import replace

    from reelcut.scoring import persist_goal_boxes

    old, new = BBox(100, 100, 200, 150), BBox(400, 100, 200, 150)
    frames = [make_frame(i) for i in range(6)]
    frames[0] = replace(frames[0], goal_boxes=(old,))
    frames[3] = replace(frames[3], goal_boxes=(new,))
    out = persist_goal_boxes(frames, hold_s=2.5)
    assert out[2].goal_boxes == (old,)
    assert out[4].goal_boxes == (new,)


def test_persist_goal_boxes_no_goals_is_noop():
    from reelcut.scoring import persist_goal_boxes

    frames = [make_frame(i) for i in range(4)]
    assert persist_goal_boxes(frames, hold_s=2.5) == frames


def test_giant_hallucinated_box_does_not_evict_held_goal():
    """Review-confirmed failure: crowded goal mouths make the detector emit
    oversized boxes that score_opportunities discards; such a frame must
    count as a dropout (inherit the held real goal), not as fresh evidence."""
    from dataclasses import replace

    from fixtures import H as FRAME_H
    from reelcut.scoring import persist_goal_boxes

    real = BBox(100, 100, 200, 150)
    giant = BBox(0, 0, 600, FRAME_H * 0.8)
    frames = [make_frame(i) for i in range(4)]
    frames[0] = replace(frames[0], goal_boxes=(real,))
    frames[2] = replace(frames[2], goal_boxes=(giant,))
    out = persist_goal_boxes(frames, hold_s=2.5)
    assert out[2].goal_boxes == (real,)
    assert out[3].goal_boxes == (real,)


# --------------------------------------------------------------------------- #
# goal_transient_events
# --------------------------------------------------------------------------- #

def test_goal_transients_cluster_into_events():
    from reelcut.scoring import goal_transient_events

    raw = [
        ScorePoint(12.0, 0.70, ("goal_chance", "goal_mouth")),
        ScorePoint(13.0, 0.65, ("goal_chance", "goal_mouth")),
        ScorePoint(15.5, 0.80, ("goal_mouth",)),       # ball fetched from net
        ScorePoint(26.0, 0.70, ("near_ball",)),        # fast but not at a goal
        ScorePoint(2.0, 0.40, ("goal_mouth",)),        # at the goal but slow
        ScorePoint(40.0, 0.75, ("goal_mouth",)),       # second goal
    ]
    out = goal_transient_events(raw, lead_s=5.0, tail_s=3.0, min_peak=0.6)
    assert len(out) == 2
    assert (out[0].start_s, out[0].end_s) == (7.0, 18.5)   # 12-5 .. 15.5+3
    assert out[0].peak_score == 0.80
    assert (out[1].start_s, out[1].end_s) == (35.0, 43.0)


def test_goal_transients_independent_of_smoothing():
    """A lone 1-sample spike is still a goal event — no smoothed-timeline
    event needs to exist around it (the failure that lost the t=42 goal)."""
    from reelcut.scoring import goal_transient_events

    raw = [ScorePoint(42.0, 0.85, ("goal_mouth", "goal_chance"))]
    out = goal_transient_events(raw, lead_s=5.0, tail_s=3.0, min_peak=0.6)
    assert len(out) == 1
    assert (out[0].start_s, out[0].end_s) == (37.0, 45.0)


def test_goal_transients_none_without_goal_mouth():
    from reelcut.scoring import goal_transient_events

    raw = [ScorePoint(10.0, 0.9, ("near_ball",))]
    assert goal_transient_events(raw, 5.0, 3.0, 0.6) == []
