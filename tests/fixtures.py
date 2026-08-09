"""Synthetic builders for unit tests — no video files needed.

Helpers other test modules import; keep them pure and deterministic.
"""
from __future__ import annotations

from reelcut.types import (
    BallObs,
    BBox,
    FrameObservation,
    IdentityPoint,
    OcrRead,
    PlayerObs,
    ScorePoint,
    TargetSpec,
)

FPS = 5.0
W, H = 1280, 720


def make_frame(
    i: int,
    players: list[PlayerObs] = (),
    ball: BallObs | None = None,
    ocr: list[OcrRead] = (),
) -> FrameObservation:
    return FrameObservation(
        frame_index=int(i * 30 / FPS),
        timestamp_s=i / FPS,
        frame_w=W,
        frame_h=H,
        players=tuple(players),
        ball=ball,
        ocr=tuple(ocr),
    )


def player(track_id: int, x: float, y: float, *, h: float = 80.0,
           cls: str = "player", speed: float | None = None,
           hsv: tuple[float, ...] | None = None) -> PlayerObs:
    return PlayerObs(track_id=track_id, bbox=BBox(x, y, h * 0.45, h),
                     class_name=cls, confidence=0.9, speed=speed,
                     torso_hsv=hsv)


def ball(x: float, y: float, *, conf: float = 0.8) -> BallObs:
    return BallObs(bbox=BBox(x - 8, y - 8, 16, 16), confidence=conf)


def walk_frames(
    track_id: int, n: int, x0: float, y0: float, dx: float, dy: float,
    start: int = 0, **player_kw
) -> list[FrameObservation]:
    """n frames of a single player moving linearly from (x0, y0)."""
    return [
        make_frame(start + i, [player(track_id, x0 + dx * i, y0 + dy * i, **player_kw)])
        for i in range(n)
    ]


def spec(jersey: str = "10", color: str = "blue", frame: int = 0,
         box: BBox | None = None) -> TargetSpec:
    return TargetSpec(jersey=jersey, team_color=color, seed_frame_index=frame,
                      seed_box=box or BBox(100, 100, 36, 80))
