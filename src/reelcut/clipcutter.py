"""Clip planning (pure) + reel writing (ffmpeg wrapper).

Events -> padded/merged/split clips -> highlights.mp4 + highlights.json.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import ReelcutConfig
from .ffmpeg import ffmpeg_exe
from .types import Clip, IdentityPoint, InvolvementEvent, ScorePoint

_EPS = 1e-9

# Mutable working span while planning: [start_s, end_s, score, tags-set].
_Span = list  # [float, float, float, set[str]]


def _merge_spans(spans: list[_Span], merge_gap_s: float) -> list[_Span]:
    """Merge spans whose gap is < merge_gap_s (overlaps have negative gap)."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s[0], s[1]))
    merged: list[_Span] = [spans[0]]
    for span in spans[1:]:
        cur = merged[-1]
        if span[0] - cur[1] < merge_gap_s:
            cur[1] = max(cur[1], span[1])
            cur[2] = max(cur[2], span[2])
            cur[3] |= span[3]
        else:
            merged.append(span)
    return merged


def _extend_to_min(span: _Span, min_s: float, duration_s: float) -> None:
    """Grow span symmetrically to min_s; deficits at a video edge push to the
    other side. A video shorter than min_s yields the whole video."""
    need = min_s - (span[1] - span[0])
    if need <= 0:
        return
    span[0] -= need / 2.0
    span[1] += need / 2.0
    if span[0] < 0.0:
        span[1] -= span[0]  # push the clamped-off part right
        span[0] = 0.0
    if span[1] > duration_s:
        span[0] -= span[1] - duration_s  # push the clamped-off part left
        span[1] = duration_s
    span[0] = max(0.0, span[0])


def _split_to_max(
    span: _Span, scores: list[ScorePoint], min_s: float, max_s: float
) -> list[_Span]:
    """Recursively split an over-max span at its lowest smoothed-score valley.

    The split point is the min-score timestamp at least min_s from both edges
    (earliest wins ties). With no eligible score point, fall back to the
    midpoint; a span too short to yield two >= min_s pieces stays whole.
    """
    start, end, score, tags = span
    if end - start <= max_s + _EPS:
        return [span]
    lo = start + min_s
    hi = end - min_s
    candidates = [p for p in scores if lo <= p.timestamp_s <= hi]
    if candidates:
        t = min(candidates, key=lambda p: (p.score, p.timestamp_s)).timestamp_s
    elif end - start >= 2.0 * min_s:
        t = (start + end) / 2.0
    else:
        return [span]
    if t - start < _EPS or end - t < _EPS:  # degenerate piece (min_s == 0)
        t = (start + end) / 2.0
    left: _Span = [start, t, score, set(tags)]
    right: _Span = [t, end, score, set(tags)]
    return (
        _split_to_max(left, scores, min_s, max_s)
        + _split_to_max(right, scores, min_s, max_s)
    )


def _mean_identity(identity: list[IdentityPoint], start: float, end: float) -> float:
    confs = [p.confidence for p in identity if start - _EPS <= p.timestamp_s <= end + _EPS]
    return sum(confs) / len(confs) if confs else 1.0


def plan_clips(
    events: list[InvolvementEvent],
    scores: list[ScorePoint],
    identity: list[IdentityPoint],
    video_duration_s: float,
    cfg: ReelcutConfig,
) -> list[Clip]:
    """Pure planning core.

    * pad each event by cfg.clip_pad_s both sides, clamp to [0, duration]
    * merge clips whose gap < cfg.clip_merge_gap_s (reasons/tags unioned,
      score = max)
    * enforce min length cfg.clip_min_s (extend symmetrically, clamped)
    * enforce max length cfg.clip_max_s by splitting at the lowest smoothed
      score valley inside the clip (each piece re-checked against max)
    * clip.confidence = mean identity confidence over the clip's span
    * output sorted, non-overlapping
    """
    duration = float(video_duration_s)
    spans: list[_Span] = []
    for ev in events:
        start = max(0.0, ev.start_s - cfg.clip_pad_s)
        end = min(duration, ev.end_s + cfg.clip_pad_s)
        if end - start <= _EPS:
            continue
        spans.append([start, end, ev.peak_score, set(ev.tags)])
    if not spans:
        return []

    spans = _merge_spans(spans, cfg.clip_merge_gap_s)
    for span in spans:
        _extend_to_min(span, cfg.clip_min_s, duration)
    spans = _merge_spans(spans, cfg.clip_merge_gap_s)  # extension may re-overlap

    pieces: list[_Span] = []
    for span in spans:
        pieces.extend(_split_to_max(span, scores, cfg.clip_min_s, cfg.clip_max_s))

    return [
        Clip(
            start_s=start,
            end_s=end,
            score=score,
            reasons=tuple(sorted(tags)) if tags else ("involvement",),
            confidence=_mean_identity(identity, start, end),
        )
        for start, end, score, tags in pieces
    ]


def write_highlights_json(
    clips: list[Clip], out_path: Path, source_video: Path, cfg: ReelcutConfig
) -> None:
    """Schema:
    {"source": str, "generated_by": "reelcut", "clips": [
        {"start_s": float, "end_s": float, "score": float,
         "reasons": [...], "confidence": float}]}
    """
    payload = {
        "source": str(source_video),
        "generated_by": "reelcut",
        "clips": [
            {
                "start_s": round(c.start_s, 2),
                "end_s": round(c.end_s, 2),
                "score": round(c.score, 2),
                "reasons": list(c.reasons),
                "confidence": round(c.confidence, 2),
            }
            for c in clips
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(exe: str, args: list[str], check: bool) -> bool:
    """Run ffmpeg with -y/-loglevel error prepended; True on success. When
    ``check``, raise CalledProcessError carrying the stderr tail instead."""
    proc = subprocess.run(
        [exe, "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and check:
        tail = (proc.stderr or "").strip()[-2000:]
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=proc.stdout, stderr=tail
        )
    return proc.returncode == 0


def _has_frames(path: Path) -> bool:
    """True when the file exists and cv2 can decode at least one frame."""
    import cv2  # bundled with inference; keep plan_clips importable without it

    if not path.is_file() or path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


def cut_reel(
    clips: list[Clip], source_video: Path, out_path: Path, work_dir: Path
) -> None:
    """Cut each clip from the source at original quality and concat.

    Use ffmpeg stream copy (-c copy) with the concat demuxer where possible;
    fall back to re-encode (libx264, crf 18) if stream copy fails (e.g.
    cut points not on keyframes producing broken output is acceptable for
    v1 ONLY via re-encode fallback — prefer accurate cuts over speed:
    use -ss before -i with re-encode for frame-accurate cuts).
    Segments go under work_dir; concat list file too.
    """
    if not clips:
        raise ValueError("cut_reel needs at least one clip")
    exe = ffmpeg_exe()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for i, clip in enumerate(clips):
        seg = work_dir / f"seg_{i:03d}.mp4"
        dur = clip.end_s - clip.start_s
        # Frame-accurate cut: -ss before -i seeks, decoding restarts at the
        # requested time, and re-encoding (libx264, crf 18) makes each segment
        # self-contained. A `-c copy` cut would keep hidden GOP pre-roll
        # packets (masked only by an mp4 edit list the concat demuxer then
        # ignores), splicing up to a GOP of unrequested footage per clip into
        # the reel and corrupting its timing.
        _run(
            exe,
            [
                "-ss", f"{clip.start_s:.3f}", "-i", str(source_video),
                "-t", f"{dur:.3f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-movflags", "+faststart",
                "-c:a", "copy",
                str(seg),
            ],
            check=True,
        )
        if not _has_frames(seg):
            raise RuntimeError(f"ffmpeg produced an unreadable segment: {seg}")
        segments.append(seg)

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{seg.name}'\n" for seg in segments), encoding="utf-8"
    )
    concat_args = ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
    # Segments are uniformly encoded and pre-roll-free, so demuxer concat can
    # stream copy; fall back to re-encode if copy still fails.
    ok = _run(exe, [*concat_args, "-c", "copy", str(out_path)], check=False)
    if not ok or not _has_frames(out_path):
        _run(
            exe,
            [
                *concat_args,
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-movflags", "+faststart",
                "-c:a", "copy",
                str(out_path),
            ],
            check=True,
        )
        if not _has_frames(out_path):
            raise RuntimeError(f"ffmpeg produced an unreadable reel: {out_path}")
