"""Identity stitching — label tracklets target / not_target / unknown.

Pure functions, no I/O. Evidence sources, in trust order:
  1. seed click (pins one tracklet as TARGET)
  2. jersey OCR (matching number = strong positive; confident different
     number = strong negative)
  3. team-color histogram (cheap NOT_TARGET veto for opponents/referees)
  4. kinematic continuity (plausible hand-off between tracklets in time)

Ambiguity stays UNKNOWN: a reel of the wrong kid is worse than a missed play.
Never flip a tracklet to TARGET on appearance similarity alone.
"""
from __future__ import annotations

import math
from collections import Counter

from .config import TEAM_COLORS, ReelcutConfig
from .types import (
    BBox,
    FrameObservation,
    IdentityLabel,
    IdentityPoint,
    LabeledTracklet,
    TargetSpec,
    Tracklet,
    TrackletPoint,
)
from .workflow_client import HIST_H_BINS, HIST_S_BINS

# Seed matching
_SEED_FRAME_WINDOW = 5        # floor: observations within +-5 source frames
_SEED_MIN_IOU = 0.2

# Labeling
_OCR_TARGET_CONFIDENCE = 0.9  # confidence granted by matching-number votes

# Kinematic chaining
_CHAIN_MIN_PLAUSIBILITY = 0.3
_MIN_JUNCTION_GAP_S = 0.25    # floor on the gap used to scale distance/speed


def dedupe_player_boxes(
    frames: list[FrameObservation], overlap: float
) -> list[FrameObservation]:
    """Per frame, drop the lower-confidence of two player boxes whose
    CONTAINMENT (intersection over the smaller box's area) exceeds
    ``overlap``.

    Pile-ups make the detector/tracker put two boxes on one kid — often a
    small partial box nested inside the full one (measured: a '#8' and a
    '#2' box on the same player). Containment catches nesting that plain
    IoU misses when the box sizes differ. Returns a new list; input
    unmodified.
    """
    from dataclasses import replace as _replace

    def containment(a, b) -> float:
        ix = max(0.0, min(a.x2, b.x2) - max(a.x, b.x))
        iy = max(0.0, min(a.y2, b.y2) - max(a.y, b.y))
        smaller = min(a.area, b.area)
        return (ix * iy) / smaller if smaller > 0 else 0.0

    out: list[FrameObservation] = []
    for f in frames:
        if len(f.players) < 2:
            out.append(f)
            continue
        keep = []
        for p in sorted(f.players, key=lambda p: -p.confidence):
            if all(containment(p.bbox, k.bbox) <= overlap for k in keep):
                keep.append(p)
        if len(keep) == len(f.players):
            out.append(f)
        else:
            order = {p.track_id: i for i, p in enumerate(f.players)}
            keep.sort(key=lambda p: order[p.track_id])
            out.append(_replace(f, players=tuple(keep)))
    return out


def build_tracklets(frames: list[FrameObservation]) -> list[Tracklet]:
    """Group per-frame PlayerObs into Tracklets by track_id.

    A gap of any length within one track_id stays one tracklet (the workflow
    tracker already decided identity); points are time-ordered. OCR reads are
    attached to their tracklet. Referee-class tracks are still returned (the
    labeling step will mark them NOT_TARGET).
    """
    by_id: dict[int, Tracklet] = {}
    class_votes: dict[int, Counter[str]] = {}
    for frame in frames:
        for obs in frame.players:
            tracklet = by_id.get(obs.track_id)
            if tracklet is None:
                tracklet = Tracklet(track_id=obs.track_id)
                by_id[obs.track_id] = tracklet
                class_votes[obs.track_id] = Counter()
            tracklet.points.append(
                TrackletPoint(
                    frame_index=frame.frame_index,
                    timestamp_s=frame.timestamp_s,
                    bbox=obs.bbox,
                    speed=obs.speed,
                    torso_hsv=obs.torso_hsv,
                )
            )
            class_votes[obs.track_id][obs.class_name] += 1
    for frame in frames:
        for read in frame.ocr:
            tracklet = by_id.get(read.track_id)
            if tracklet is not None:  # reads for unseen ids carry no geometry
                tracklet.ocr_reads.append(read)
    for track_id, tracklet in by_id.items():
        # Majority class; Counter.most_common breaks ties by first appearance.
        tracklet.class_name = class_votes[track_id].most_common(1)[0][0]
    return list(by_id.values())


def find_seed_tracklet(
    tracklets: list[Tracklet], spec: TargetSpec, frames: list[FrameObservation]
) -> int | None:
    """Return track_id of the tracklet best matching the user's seed click.

    Match = highest IoU with ``spec.seed_box`` among observations at (or
    nearest to, within +-5 frames of) ``spec.seed_frame_index``. Require
    IoU > 0.2; else None (caller errors out with a helpful message).
    """
    # Sampled observations are source_fps / sample_fps frames apart, so the
    # +-5-frame window must grow with the sampling stride or a pixel-perfect
    # click midway between samples on a high-fps source would match nothing.
    # Half the (median) stride reaches the nearest sample from any click.
    deltas = sorted(
        b.frame_index - a.frame_index
        for a, b in zip(frames, frames[1:])
        if b.frame_index > a.frame_index
    )
    stride = deltas[len(deltas) // 2] if deltas else 1
    window = max(_SEED_FRAME_WINDOW, math.ceil(stride / 2))
    best_key: tuple[float, int, int] | None = None
    best_id: int | None = None
    for tracklet in tracklets:
        for point in tracklet.points:
            delta = abs(point.frame_index - spec.seed_frame_index)
            if delta > window:
                continue
            iou = spec.seed_box.iou(point.bbox)
            if iou <= _SEED_MIN_IOU:
                continue
            # Highest IoU wins; ties fall to the nearer frame, then lowest id.
            key = (-iou, delta, tracklet.track_id)
            if best_key is None or key < best_key:
                best_key = key
                best_id = tracklet.track_id
    return best_id


def hist_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Symmetric 0..1 distance between two normalized histograms
    (1 - intersection)."""
    if not a or not b or len(a) != len(b):
        return 1.0
    if sum(a) <= 0.0 or sum(b) <= 0.0:
        return 1.0
    intersection = sum(min(x, y) for x, y in zip(a, b))
    return min(max(1.0 - intersection, 0.0), 1.0)


def team_color_reference(color_name: str) -> tuple[float, ...]:
    """Synthesize the reference torso histogram for a named kit color using
    config.TEAM_COLORS bins. Same binning as the runner's torso extractor
    (workflow_client.torso_histogram)."""
    if color_name not in TEAM_COLORS:
        raise ValueError(
            f"unknown team color {color_name!r}; have {sorted(TEAM_COLORS)}"
        )
    # Indicator histogram: 1.0 in every (H, S) cell the config range touches,
    # 0 elsewhere, deliberately NOT L1-normalized. A real solid-color jersey
    # concentrates its torso mass in 1-2 hue bins, so intersecting against a
    # normalized uniform reference (1/K per cell) would cap the intersection
    # at ~2/K and veto the target's own team. Against this indicator,
    # hist_distance(torso, ref) = 1 - (torso mass inside the color's range):
    # concentrated same-color kits score near 0, opponents near 1, and
    # cfg.color_veto_dist = 0.75 means "veto below 25% in-range mass".
    # H bins span 0..179 (width 15), S bins 0..255 (width 64); flat index =
    # h_bin * HIST_S_BINS + s_bin. V is not binned (dropped for lighting
    # robustness), so v_min never moves mass. White (low S, high V) and black
    # (low V) are achromatic — hue carries no signal — so their support covers
    # all H bins at the lowest-S bin.
    h_width = 180.0 / HIST_H_BINS
    s_width = 256.0 / HIST_S_BINS
    hist = [0.0] * (HIST_H_BINS * HIST_S_BINS)
    if color_name in ("white", "black"):
        for h_bin in range(HIST_H_BINS):
            hist[h_bin * HIST_S_BINS] = 1.0
    else:
        for h_lo, h_hi, s_min, _v_min in TEAM_COLORS[color_name]:
            h_bins = [
                i for i in range(HIST_H_BINS)
                if i * h_width <= h_hi and (i + 1) * h_width > h_lo
            ]
            s_bins = [
                j for j in range(HIST_S_BINS)
                if (j + 0.5) * s_width >= s_min  # bin center at/above s_min
            ]
            for i in h_bins:
                for j in s_bins:
                    hist[i * HIST_S_BINS + j] = 1.0
    return tuple(hist)


def _normalized_digits(text: str) -> str:
    """OCR text normalization: keep digits only ("#7." -> "7")."""
    return "".join(ch for ch in text if ch.isdigit())


def _mean_color_distance(
    tracklet: Tracklet, reference: tuple[float, ...]
) -> float | None:
    """Mean hist_distance of the tracklet's torso histograms to reference.

    None when no point carries a histogram (rule not applicable). Degenerate
    (all-zero) histograms count as distance 1.0 via hist_distance.
    """
    hists = [p.torso_hsv for p in tracklet.points if p.torso_hsv is not None]
    if not hists:
        return None
    return sum(hist_distance(h, reference) for h in hists) / len(hists)


def label_tracklets(
    tracklets: list[Tracklet],
    spec: TargetSpec,
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[LabeledTracklet]:
    """Assign labels and confidences per the evidence rules.

    Rules (applied in order, first hit wins for hard labels):
      * seed tracklet -> TARGET, confidence 1.0, evidence {"seed": 1}.
      * >= cfg.ocr_neg_votes confident reads of a DIFFERENT number, dominating
        matching reads 5:1 -> NOT_TARGET (evidence "ocr_neg").
      * referee class -> NOT_TARGET.
      * mean torso-hist distance to team reference > cfg.color_veto_dist
        -> NOT_TARGET (evidence "color").
      * >= cfg.ocr_pos_votes confident reads of the matching number,
        dominating different-number reads 5:1 -> TARGET, confidence ~0.9
        (evidence "ocr_pos").
      * else UNKNOWN with a soft prior in evidence for the chaining step.

    Vote counts are DOMINANCE ratios, not absolutes: always-on continuous
    reading yields hundreds of reads per long track, so "2 stray misreads"
    must not outvote 300 correct ones (measured ~1% confident-misread rate
    would otherwise flip every long target track NOT_TARGET).
    """
    seed_id = find_seed_tracklet(tracklets, spec, frames)
    reference = team_color_reference(spec.team_color)
    jersey = _normalized_digits(spec.jersey)
    out: list[LabeledTracklet] = []
    for tracklet in tracklets:
        pos = neg = 0
        for read in tracklet.ocr_reads:
            if read.confidence < cfg.ocr_pos_confidence:
                continue
            digits = _normalized_digits(read.text)
            if not digits or not jersey:
                continue
            if digits == jersey:
                pos += 1
            elif digits in jersey or jersey in digits:
                pass  # substring either way is neutral: "1" off a 16 shirt is
                      # an occluded digit, and "71" containing the 7 is usually
                      # a neighbor's digit fused onto the right number — neither
                      # is evidence of a different kid
            else:
                neg += 1
        color_dist = _mean_color_distance(tracklet, reference)

        if tracklet.track_id == seed_id:
            labeled = LabeledTracklet(
                tracklet, IdentityLabel.TARGET, 1.0, {"seed": 1.0}
            )
        elif neg >= cfg.ocr_neg_votes and neg >= 5 * pos:
            labeled = LabeledTracklet(
                tracklet, IdentityLabel.NOT_TARGET, 0.0, {"ocr_neg": float(neg)}
            )
        elif tracklet.class_name in cfg.referee_classes:
            labeled = LabeledTracklet(tracklet, IdentityLabel.NOT_TARGET, 0.0, {})
        elif color_dist is not None and color_dist > cfg.color_veto_dist:
            labeled = LabeledTracklet(
                tracklet, IdentityLabel.NOT_TARGET, 0.0, {"color": color_dist}
            )
        elif pos >= cfg.ocr_pos_votes and pos >= 5 * neg:
            # Dominance guard (was neg == 0): a track that reads a different
            # number nearly as often as the matching one never binds on jersey
            # evidence, but a rare stray misread among hundreds of correct
            # always-on reads cannot block (or steal) a bind either.
            labeled = LabeledTracklet(
                tracklet,
                IdentityLabel.TARGET,
                _OCR_TARGET_CONFIDENCE,
                {"ocr_pos": float(pos)},
            )
        else:
            # Soft prior for the chaining step: how the weak evidence leaned.
            evidence: dict[str, float] = {
                "ocr_pos": float(pos), "ocr_neg": float(neg),
            }
            if color_dist is not None:
                evidence["color"] = color_dist
            labeled = LabeledTracklet(tracklet, IdentityLabel.UNKNOWN, 0.0, evidence)
        out.append(labeled)
    return out


def _junction_plausibility(
    a: TrackletPoint, b: TrackletPoint, cfg: ReelcutConfig
) -> float | None:
    """Plausibility of the target hopping from point ``a`` to point ``b``.

    p = 1 - dist_ph / (cfg.link_max_dist * max(gap_s, 0.25)), clamped to
    [0, 1], where dist_ph is the box-center distance in player-heights.
    None = hard reject (backwards in time, gap too long, implied speed above
    cfg.max_plausible_speed, or degenerate boxes).
    """
    gap_s = b.timestamp_s - a.timestamp_s
    if gap_s < 0.0 or gap_s > cfg.link_max_gap_s:
        return None
    height = (a.bbox.h + b.bbox.h) / 2.0
    if height <= 0.0:
        return None
    dist_ph = a.bbox.center_dist(b.bbox) / height
    effective_gap = max(gap_s, _MIN_JUNCTION_GAP_S)
    if dist_ph / effective_gap > cfg.max_plausible_speed:
        return None
    p = 1.0 - dist_ph / (cfg.link_max_dist * effective_gap)
    return min(max(p, 0.0), 1.0)


def chain_target_tracklets(
    labeled: list[LabeledTracklet], cfg: ReelcutConfig
) -> list[LabeledTracklet]:
    """Greedy kinematic chaining of UNKNOWN tracklets onto the TARGET chain.

    Sort TARGET tracklets by time. For gaps between consecutive confirmed
    TARGET tracklets, consider UNKNOWN tracklets that fit inside the gap and
    whose endpoints are kinematically plausible (distance between boxes <=
    cfg.link_max_dist player-heights scaled by the time gap; implied speed <=
    cfg.max_plausible_speed). Promote at most one non-overlapping chain per
    gap to TARGET with confidence = 0.5 * plausibility (never 1.0). Tracklets
    overlapping in time with a confirmed TARGET tracklet are never promoted.
    Returns a new list.
    """
    out = list(labeled)
    confirmed = sorted(
        (i for i, lt in enumerate(labeled)
         if lt.label is IdentityLabel.TARGET and lt.tracklet.points),
        key=lambda i: labeled[i].tracklet.start_s,
    )
    if len(confirmed) < 2:
        return out
    target_spans = [
        (labeled[i].tracklet.start_s, labeled[i].tracklet.end_s) for i in confirmed
    ]
    unknown_idx = [
        i for i, lt in enumerate(labeled)
        if lt.label is IdentityLabel.UNKNOWN and lt.tracklet.points
    ]
    promoted: set[int] = set()
    for prev_i, next_i in zip(confirmed, confirmed[1:]):
        prev_t = labeled[prev_i].tracklet
        next_t = labeled[next_i].tracklet
        if next_t.start_s <= prev_t.end_s:
            continue
        best_p = 0.0
        best_i: int | None = None
        for cand_i in unknown_idx:
            if cand_i in promoted:
                continue
            cand = labeled[cand_i].tracklet
            if not (cand.start_s >= prev_t.end_s and cand.end_s <= next_t.start_s):
                continue  # must fit inside the gap
            if any(cand.start_s < e and cand.end_s > s for s, e in target_spans):
                continue  # never promote over a confirmed TARGET
            p_in = _junction_plausibility(prev_t.points[-1], cand.points[0], cfg)
            p_out = _junction_plausibility(cand.points[-1], next_t.points[0], cfg)
            if p_in is None or p_out is None:
                continue
            p = min(p_in, p_out)
            if p <= _CHAIN_MIN_PLAUSIBILITY:
                continue
            if p > best_p:  # ties keep the earlier candidate: deterministic
                best_p = p
                best_i = cand_i
        if best_i is not None:
            old = labeled[best_i]
            out[best_i] = LabeledTracklet(
                tracklet=old.tracklet,
                label=IdentityLabel.TARGET,
                confidence=0.5 * best_p,
                evidence={**old.evidence, "kinematic": best_p},
            )
            promoted.add(best_i)
    return out


def identity_timeline(
    labeled: list[LabeledTracklet],
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[IdentityPoint]:
    """One IdentityPoint per frame timestamp.

    If a TARGET tracklet covers the timestamp: confidence = tracklet
    confidence, bbox/speed from its point at that frame. If two TARGET
    tracklets claim the same timestamp (shouldn't happen post-chaining),
    confidence 0 — conflicting identity is unknown identity. Otherwise
    confidence 0, bbox None. Target off-screen is NORMAL: zero, not an error.
    """
    claims: dict[int, list[tuple[float, TrackletPoint, int]]] = {}
    for lt in labeled:
        if lt.label is not IdentityLabel.TARGET:
            continue
        if lt.confidence < cfg.identity_floor:
            continue  # too weak to claim the reel
        for point in lt.tracklet.points:
            claims.setdefault(point.frame_index, []).append(
                (lt.confidence, point, lt.tracklet.track_id)
            )
    timeline: list[IdentityPoint] = []
    for frame in frames:
        frame_claims = claims.get(frame.frame_index, [])
        if len(frame_claims) == 1:
            confidence, point, track_id = frame_claims[0]
            timeline.append(
                IdentityPoint(
                    timestamp_s=frame.timestamp_s,
                    confidence=confidence,
                    bbox=point.bbox,
                    track_id=track_id,
                    speed=point.speed,
                )
            )
        else:
            # 0 claims = target off-screen; 2+ = conflict -> unknown identity.
            timeline.append(
                IdentityPoint(timestamp_s=frame.timestamp_s, confidence=0.0)
            )
    return timeline
