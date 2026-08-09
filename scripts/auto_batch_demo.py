"""Hands-off finisher for a submitted Batch Processing job.

Waits for the job, exports results, auto-picks a seed target (most central
tracked player in an early frame, kit color inferred from the video), then
runs pipeline stages 2-4 locally and prints the reel summary. This is the
same sequence a web backend's webhook handler would run.

Usage:
  uv run python scripts/auto_batch_demo.py --job-id reelcut-goals2-j \
      --batch-id reelcut-goals2 --video data/fixtures/game_goals.mp4 \
      --out output/goals_batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv


def pick_seed(client, video: Path, cfg):
    """Most central tracked player within the first ~10 s of results."""
    import cv2

    from reelcut.config import TEAM_COLORS
    from reelcut.stitching import hist_distance, team_color_reference
    from reelcut.workflow_client import torso_histogram

    best = None  # (dist_to_center, frame_index, bbox)
    for obs in client.run(video, cfg):
        if obs.timestamp_s > 10.0:
            break
        for p in obs.players:
            if p.class_name != "player":
                continue
            d = (p.bbox.cx - obs.frame_w / 2) ** 2 + (p.bbox.cy - obs.frame_h / 2) ** 2
            if best is None or d < best[0]:
                best = (d, obs.frame_index, p.bbox)
    if best is None:
        raise SystemExit("no tracked players in the first 10s of batch results")
    _, frame_index, bbox = best

    color = "red"
    if video.exists():
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, img = cap.read()
        cap.release()
        if ok:
            hist = torso_histogram(img, (bbox.x, bbox.y, bbox.w, bbox.h))
            color = min(TEAM_COLORS, key=lambda c: hist_distance(hist, team_color_reference(c)))
    return frame_index, bbox, color


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jersey", default="10", help="placeholder when unknown")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    from reelcut import batch
    from reelcut.batch import BatchResultsClient
    from reelcut.config import Paths, ReelcutConfig
    from reelcut.pipeline import run_pipeline
    from reelcut.types import TargetSpec

    print(f"[demo] waiting for job {args.job_id} ...")
    batch.wait_for_job(args.job_id, poll_s=30.0)
    results_batch = batch.results_batch_of(args.job_id)
    print(f"[demo] job finished; exporting results batch {results_batch}")
    results_dir = args.out / "batch_results"
    batch.export_results(results_batch, results_dir)

    cfg = ReelcutConfig()
    client = BatchResultsClient(results_dir)
    frame_index, bbox, color = pick_seed(client, args.video, cfg)
    print(f"[demo] seed: frame {frame_index}, box "
          f"{bbox.x:.0f},{bbox.y:.0f},{bbox.w:.0f},{bbox.h:.0f}, kit={color}")

    spec = TargetSpec(
        jersey=args.jersey,
        team_color=color,
        seed_frame_index=frame_index,
        seed_box=bbox,
    )
    paths = Paths(out_dir=args.out)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        args.video, spec, cfg, paths, BatchResultsClient(results_dir),
        debug_video=True,
    )
    print(f"[demo] reel: {paths.highlights_mp4}")
    print(f"[demo] metadata: {paths.highlights_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
