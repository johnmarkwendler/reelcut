"""Unit tests for identity stitching (src/reelcut/stitching.py)."""
from __future__ import annotations

import pytest
from fixtures import make_frame, player, spec

from reelcut.config import ReelcutConfig
from reelcut.stitching import (
    build_tracklets,
    chain_target_tracklets,
    find_seed_tracklet,
    hist_distance,
    identity_timeline,
    label_tracklets,
    team_color_reference,
)
from reelcut.types import (
    BBox,
    FrameObservation,
    IdentityLabel,
    LabeledTracklet,
    OcrRead,
    Tracklet,
    TrackletPoint,
)
from reelcut.workflow_client import HIST_H_BINS, HIST_S_BINS

CFG = ReelcutConfig()
BLUE = team_color_reference("blue")
RED = team_color_reference("red")


def frames_from(
    players_by_i: dict[int, list],
    ocr_by_i: dict[int, list[OcrRead]] | None = None,
) -> list[FrameObservation]:
    ocr_by_i = ocr_by_i or {}
    return [
        make_frame(i, players_by_i[i], ocr=ocr_by_i.get(i, []))
        for i in sorted(players_by_i)
    ]


# --------------------------------------------------------------------------- #
# build_tracklets
# --------------------------------------------------------------------------- #

def test_build_tracklets_groups_orders_and_votes_class() -> None:
    frames = frames_from(
        {
            0: [player(1, 100, 100), player(2, 300, 100, cls="referee")],
            1: [player(1, 102, 100), player(2, 300, 100, cls="referee")],
            2: [player(1, 104, 100, cls="referee")],  # minority vote
            3: [player(1, 106, 100)],
        },
        {1: [OcrRead(1, "10", 0.8)]},
    )
    tracklets = {t.track_id: t for t in build_tracklets(frames)}
    assert set(tracklets) == {1, 2}
    t1 = tracklets[1]
    assert len(t1.points) == 4
    assert [p.timestamp_s for p in t1.points] == sorted(p.timestamp_s for p in t1.points)
    assert [p.frame_index for p in t1.points] == [0, 6, 12, 18]
    assert t1.class_name == "player"  # majority over 3x player / 1x referee
    assert t1.ocr_reads == [OcrRead(1, "10", 0.8)]
    t2 = tracklets[2]
    assert t2.class_name == "referee"
    assert t2.ocr_reads == []


# --------------------------------------------------------------------------- #
# hist_distance / team_color_reference
# --------------------------------------------------------------------------- #

def test_hist_distance_basics() -> None:
    assert hist_distance(BLUE, BLUE) == pytest.approx(0.0)
    assert hist_distance(BLUE, RED) == pytest.approx(1.0)  # disjoint hue bins


def test_hist_distance_degenerate_inputs() -> None:
    zeros = (0.0,) * len(BLUE)
    assert hist_distance((), BLUE) == 1.0
    assert hist_distance(BLUE, ()) == 1.0
    assert hist_distance(None, BLUE) == 1.0  # type: ignore[arg-type]
    assert hist_distance(zeros, BLUE) == 1.0
    assert hist_distance(BLUE, BLUE[:10]) == 1.0  # length mismatch


def test_team_color_reference_is_indicator_and_shaped() -> None:
    # The reference is an unnormalized indicator over the color's (H, S)
    # support: hist_distance(torso, ref) then measures 1 - in-range mass, so
    # a concentrated real jersey scores near 0 instead of being capped at
    # ~2/K by a uniform normalized reference.
    from reelcut.config import TEAM_COLORS

    for name in TEAM_COLORS:
        ref = team_color_reference(name)
        assert len(ref) == HIST_H_BINS * HIST_S_BINS
        assert all(v in (0.0, 1.0) for v in ref)
        assert any(v == 1.0 for v in ref)


def test_team_color_reference_white_black_low_saturation() -> None:
    for name in ("white", "black"):
        ref = team_color_reference(name)
        for h_bin in range(HIST_H_BINS):
            assert ref[h_bin * HIST_S_BINS] > 0.0        # lowest-S bin
            for s_bin in range(1, HIST_S_BINS):
                assert ref[h_bin * HIST_S_BINS + s_bin] == 0.0


def test_team_color_reference_unknown_color_raises() -> None:
    with pytest.raises(ValueError):
        team_color_reference("chartreuse")


def _solid_torso_hist(hue: int, sat: int = 200, val: int = 180) -> tuple[float, ...]:
    """Actual torso_histogram of a solid-color crop (concentrated, like a
    real jersey), via the same extractor the runner uses."""
    import cv2
    import numpy as np

    from reelcut.workflow_client import torso_histogram

    hsv = np.full((120, 60, 3), (hue, sat, val), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return torso_histogram(bgr, (0.0, 0.0, 60.0, 120.0))


def test_concentrated_real_jersey_passes_own_color_veto() -> None:
    # Royal blue (hue ~116) concentrates in 1-2 hue bins; the veto must not
    # fire on the target's own team (regression: uniform normalized reference
    # capped intersection at ~2/12 and vetoed realistic kits).
    blue_jersey = _solid_torso_hist(hue=116)
    assert hist_distance(blue_jersey, BLUE) < 0.25
    red_jersey = _solid_torso_hist(hue=3)
    assert hist_distance(red_jersey, BLUE) > CFG.color_veto_dist


def test_concentrated_jersey_labeling_keeps_own_team_unvetoed() -> None:
    blue_jersey = _solid_torso_hist(hue=116)
    red_jersey = _solid_torso_hist(hue=3)
    frames = frames_from(
        {i: [player(61, 500, 100, hsv=blue_jersey),
             player(62, 650, 100, hsv=red_jersey)] for i in range(6)}
    )
    labeled = {lt.tracklet.track_id: lt for lt in
               label_tracklets(build_tracklets(frames), spec(color="blue"),
                               frames, CFG)}
    assert labeled[61].label is IdentityLabel.UNKNOWN  # own team: no veto
    assert labeled[62].label is IdentityLabel.NOT_TARGET
    assert labeled[62].evidence.get("color", 0.0) > CFG.color_veto_dist


def test_concentrated_matching_ocr_promotes_despite_realistic_kit() -> None:
    # A same-team tracklet with a concentrated (realistic) torso histogram
    # and two confident matching jersey reads must become TARGET, not be
    # color-vetoed before the OCR rule can run.
    blue_jersey = _solid_torso_hist(hue=116)
    frames = frames_from(
        {i: [player(71, 500, 100, hsv=blue_jersey)] for i in range(6)},
        {0: [OcrRead(71, "10", 0.8)], 1: [OcrRead(71, "10", 0.8)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(color="blue"),
                              frames, CFG)
    assert labeled[0].label is IdentityLabel.TARGET
    assert labeled[0].evidence.get("ocr_pos") == 2.0


# --------------------------------------------------------------------------- #
# find_seed_tracklet / seed labeling
# --------------------------------------------------------------------------- #

def test_seed_click_pins_correct_tracklet_among_several() -> None:
    frames = frames_from(
        {i: [player(1, 100, 100), player(2, 400, 300), player(3, 800, 200)]
         for i in range(6)}
    )
    tracklets = build_tracklets(frames)
    s = spec(frame=0, box=BBox(400, 300, 36, 80))  # clicked on track 2
    assert find_seed_tracklet(tracklets, s, frames) == 2

    labeled = {lt.tracklet.track_id: lt for lt in
               label_tracklets(tracklets, s, frames, CFG)}
    assert labeled[2].label is IdentityLabel.TARGET
    assert labeled[2].confidence == 1.0
    assert labeled[2].evidence.get("seed") == 1.0
    assert labeled[1].label is not IdentityLabel.TARGET
    assert labeled[3].label is not IdentityLabel.TARGET


def test_seed_click_requires_iou_overlap() -> None:
    frames = frames_from({i: [player(1, 100, 100)] for i in range(6)})
    tracklets = build_tracklets(frames)
    assert find_seed_tracklet(tracklets, spec(box=BBox(900, 500, 36, 80)),
                              frames) is None


def test_seed_click_ignores_observations_outside_frame_window() -> None:
    # Track only exists from fixture index 2 on (source frame 12) — more than
    # 5 source frames from the click at frame 0.
    frames = frames_from({i: [player(1, 100, 100)] for i in range(2, 6)})
    tracklets = build_tracklets(frames)
    assert find_seed_tracklet(tracklets, spec(frame=0), frames) is None


def test_seed_window_scales_with_sampling_stride() -> None:
    # 60 fps source sampled at 5 fps -> observations 12 source frames apart.
    # A pixel-perfect click on frame 30 (midway between samples 24 and 36)
    # must still match; a fixed +-5 window would return None.
    frames = [
        FrameObservation(
            frame_index=i * 12,
            timestamp_s=i / 5.0,
            frame_w=1280,
            frame_h=720,
            players=(player(1, 100, 100),),
        )
        for i in range(6)
    ]
    tracklets = build_tracklets(frames)
    s = spec(frame=30, box=BBox(100, 100, 36, 80))
    assert find_seed_tracklet(tracklets, s, frames) == 1


def test_seed_beats_negative_ocr() -> None:
    # First-hit-wins ordering: the seeded tracklet stays TARGET even with
    # confident different-number reads.
    frames = frames_from(
        {i: [player(1, 100, 100)] for i in range(4)},
        {0: [OcrRead(1, "7", 0.9)], 1: [OcrRead(1, "7", 0.9)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.TARGET
    assert labeled[0].confidence == 1.0


# --------------------------------------------------------------------------- #
# label_tracklets: OCR and color rules
# --------------------------------------------------------------------------- #

def test_ocr_negative_votes_flip_to_not_target() -> None:
    frames = frames_from(
        {i: [player(21, 500, 100), player(22, 650, 100)] for i in range(6)},
        {
            0: [OcrRead(21, "#7.", 0.9), OcrRead(22, "7", 0.2)],
            1: [OcrRead(21, " 7", 0.9), OcrRead(22, "7", 0.35)],
            2: [OcrRead(22, "??", 0.9)],  # no digits -> never a vote
        },
    )
    labeled = {lt.tracklet.track_id: lt for lt in
               label_tracklets(build_tracklets(frames), spec(), frames, CFG)}
    # 2 confident "7" reads vs jersey "10" (text normalized to digits).
    assert labeled[21].label is IdentityLabel.NOT_TARGET
    assert labeled[21].evidence.get("ocr_neg") == 2.0
    # Low-confidence and digit-less reads never count.
    assert labeled[22].label is IdentityLabel.UNKNOWN


def test_opponent_color_veto() -> None:
    frames = frames_from(
        {i: [player(31, 500, 100, hsv=RED), player(32, 650, 100, hsv=BLUE)]
         for i in range(6)}
    )
    labeled = {lt.tracklet.track_id: lt for lt in
               label_tracklets(build_tracklets(frames), spec(color="blue"),
                               frames, CFG)}
    assert labeled[31].label is IdentityLabel.NOT_TARGET
    assert labeled[31].evidence.get("color") == pytest.approx(1.0)
    assert labeled[32].label is IdentityLabel.UNKNOWN  # same kit, no OCR


def test_referee_class_is_not_target() -> None:
    frames = frames_from({i: [player(51, 500, 100, cls="referee")]
                          for i in range(4)})
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.NOT_TARGET


def test_matching_ocr_votes_promote_to_target() -> None:
    frames = frames_from(
        {i: [player(41, 500, 100)] for i in range(6)},
        {
            0: [OcrRead(41, "10", 0.6)],
            2: [OcrRead(41, "10.", 0.45)],
            3: [OcrRead(41, "10", 0.3)],  # below ocr_pos_confidence
        },
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.TARGET
    assert labeled[0].confidence == pytest.approx(0.9)
    assert labeled[0].evidence.get("ocr_pos") == 2.0


def test_one_matching_read_alone_does_not_promote() -> None:
    """Target promotion needs the confirmed pair: one read can be a misread,
    and a misread must never steal the target box (numbers.py keeps reading
    until two agree, so genuine binds always reach two)."""
    frames = frames_from(
        {i: [player(41, 500, 100)] for i in range(6)},
        {0: [OcrRead(41, "10", 0.6)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.UNKNOWN


def test_two_agreeing_reads_promote() -> None:
    frames = frames_from(
        {i: [player(41, 500, 100)] for i in range(6)},
        {0: [OcrRead(41, "10", 0.6)], 1: [OcrRead(41, "10", 0.7)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.TARGET


def test_matching_read_with_conflict_stays_unknown() -> None:
    """A track that ALSO read a different number never binds on one sighting."""
    frames = frames_from(
        {i: [player(41, 500, 100)] for i in range(6)},
        {0: [OcrRead(41, "10", 0.6)], 1: [OcrRead(41, "7", 0.6)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert labeled[0].label is IdentityLabel.UNKNOWN
    assert labeled[0].confidence == 0.0


# --------------------------------------------------------------------------- #
# chain_target_tracklets
# --------------------------------------------------------------------------- #

def _gap_scenario(gap_players_by_i: dict[int, list]) -> list[LabeledTracklet]:
    """Seeded target track 1 (t 0..1.8), OCR-confirmed track 3 (t 4.0..5.8),
    2 s gap populated by ``gap_players_by_i`` (fixture indices 10..17)."""
    players_by_i: dict[int, list] = {}
    for i in range(10):
        players_by_i[i] = [player(1, 100, 100)]
    for i, plist in gap_players_by_i.items():
        players_by_i.setdefault(i, []).extend(plist)
    for i in range(20, 30):
        players_by_i.setdefault(i, []).append(player(3, 110, 100))
    ocr = {20: [OcrRead(3, "10", 0.9)], 21: [OcrRead(3, "10", 0.9)]}
    frames = frames_from(players_by_i, ocr)
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    return chain_target_tracklets(labeled, CFG)


def test_chaining_promotes_plausible_bridge_across_2s_gap() -> None:
    chained = {lt.tracklet.track_id: lt for lt in _gap_scenario(
        {i: [player(2, 105, 100)] for i in range(10, 18)}
    )}
    assert chained[1].label is IdentityLabel.TARGET  # anchors untouched
    assert chained[3].label is IdentityLabel.TARGET
    bridge = chained[2]
    assert bridge.label is IdentityLabel.TARGET
    # p = 1 - (5 px / 80 px) / (link_max_dist * max(0.2 s, 0.25)) at the
    # tighter junction; confidence = 0.5 * p, never 1.0.
    expected_p = 1.0 - (5.0 / 80.0) / (CFG.link_max_dist * 0.25)
    assert bridge.confidence == pytest.approx(0.5 * expected_p)
    assert 0.0 < bridge.confidence <= 0.5
    assert bridge.evidence.get("kinematic") == pytest.approx(expected_p)


def test_chaining_rejects_implausible_teleport() -> None:
    # 40 player-heights away (3200 px at h=80): implied speed far above
    # cfg.max_plausible_speed, plausibility clamps to 0.
    chained = {lt.tracklet.track_id: lt for lt in _gap_scenario(
        {i: [player(4, 3300, 100)] for i in range(10, 18)}
    )}
    assert chained[4].label is IdentityLabel.UNKNOWN
    assert chained[4].confidence == 0.0


def test_chaining_promotes_best_and_at_most_one_per_gap() -> None:
    chained = {lt.tracklet.track_id: lt for lt in _gap_scenario(
        {i: [player(2, 105, 100), player(4, 3300, 100)] for i in range(10, 18)}
    )}
    assert chained[2].label is IdentityLabel.TARGET
    assert chained[4].label is IdentityLabel.UNKNOWN


def test_tracklet_overlapping_confirmed_target_never_promoted() -> None:
    # Track 5 sits right next to the target and spans t 1.6..2.8 — it
    # time-overlaps confirmed track 1 (t 0..1.8), so it must stay UNKNOWN
    # no matter how kinematically convenient it is.
    chained = {lt.tracklet.track_id: lt for lt in _gap_scenario(
        {i: [player(5, 105, 100)] for i in range(8, 15)}
    )}
    assert chained[1].label is IdentityLabel.TARGET
    assert chained[5].label is IdentityLabel.UNKNOWN
    assert chained[5].confidence == 0.0


def test_chaining_returns_new_list_and_keeps_input_unmodified() -> None:
    players_by_i: dict[int, list] = {i: [player(1, 100, 100)] for i in range(10)}
    for i in range(10, 18):
        players_by_i[i] = [player(2, 105, 100)]
    for i in range(20, 30):
        players_by_i[i] = [player(3, 110, 100)]
    ocr = {20: [OcrRead(3, "10", 0.9)], 21: [OcrRead(3, "10", 0.9)]}
    frames = frames_from(players_by_i, ocr)
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    chained = chain_target_tracklets(labeled, CFG)
    assert chained is not labeled
    by_id = {lt.tracklet.track_id: lt for lt in labeled}
    assert by_id[2].label is IdentityLabel.UNKNOWN  # input untouched


# --------------------------------------------------------------------------- #
# identity_timeline
# --------------------------------------------------------------------------- #

def test_timeline_zero_where_nobody_claims_coverage() -> None:
    players_by_i: dict[int, list] = {i: [player(1, 100, 100)] for i in range(10)}
    for i in range(10, 15):
        players_by_i[i] = []  # target walked off screen: normal, not an error
    frames = frames_from(players_by_i)
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    timeline = identity_timeline(labeled, frames, CFG)
    assert len(timeline) == len(frames)
    assert [p.timestamp_s for p in timeline] == [f.timestamp_s for f in frames]
    for point in timeline[:10]:
        assert point.confidence == 1.0
        assert point.bbox == BBox(100, 100, 36, 80)
        assert point.track_id == 1
    for point in timeline[10:]:
        assert point.confidence == 0.0
        assert point.bbox is None
        assert point.track_id is None


def test_conflicting_simultaneous_target_claims_zero_confidence() -> None:
    # Two tracklets both OCR-confirm jersey "10" over the same frames: the
    # identity is unknowable there, so confidence must drop to 0.
    frames = frames_from(
        {i: [player(11, 500, 300), player(12, 700, 300)] for i in range(10)},
        {
            0: [OcrRead(11, "10", 0.9), OcrRead(12, "10", 0.9)],
            1: [OcrRead(11, "10", 0.9), OcrRead(12, "10", 0.9)],
        },
    )
    labeled = label_tracklets(build_tracklets(frames), spec(), frames, CFG)
    assert [lt.label for lt in labeled] == [IdentityLabel.TARGET] * 2
    timeline = identity_timeline(labeled, frames, CFG)
    assert len(timeline) == len(frames)
    for point in timeline:
        assert point.confidence == 0.0
        assert point.bbox is None
        assert point.track_id is None


def test_timeline_drops_target_claims_below_identity_floor() -> None:
    weak = LabeledTracklet(
        tracklet=Tracklet(
            track_id=9,
            points=[TrackletPoint(frame_index=0, timestamp_s=0.0,
                                  bbox=BBox(0, 0, 36, 80))],
        ),
        label=IdentityLabel.TARGET,
        confidence=CFG.identity_floor - 0.05,
    )
    timeline = identity_timeline([weak], [make_frame(0)], CFG)
    assert timeline[0].confidence == 0.0
    assert timeline[0].bbox is None


def test_partial_digit_read_is_neutral_not_negative() -> None:
    """A '1' read off a '16' shirt (occluded digit) must not count against."""
    frames = frames_from(
        {i: [player(41, 500, 100)] for i in range(6)},
        {0: [OcrRead(41, "1", 0.6)], 1: [OcrRead(41, "6", 0.6)],
         2: [OcrRead(41, "16", 0.6)], 3: [OcrRead(41, "16", 0.7)]},
    )
    labeled = label_tracklets(build_tracklets(frames), spec(jersey="16"), frames, CFG)
    assert labeled[0].label is IdentityLabel.TARGET  # partials neutral, pair binds


def test_dedupe_player_boxes_drops_overlapping_duplicate():
    from fixtures import make_frame, player
    from reelcut.stitching import dedupe_player_boxes

    a = player(1, 100, 100, h=100)
    dup = player(2, 102, 101, h=100)            # same kid, second track
    far = player(3, 500, 100, h=100)
    f = make_frame(0, [a, dup, far])
    out = dedupe_player_boxes([f], overlap=0.65)
    ids = [p.track_id for p in out[0].players]
    assert ids == [1, 3]                        # higher-confidence kept, order stable


def test_dedupe_player_boxes_keeps_separated_players():
    from fixtures import make_frame, player
    from reelcut.stitching import dedupe_player_boxes

    f = make_frame(0, [player(1, 100, 100), player(2, 300, 100)])
    out = dedupe_player_boxes([f], overlap=0.65)
    assert len(out[0].players) == 2
