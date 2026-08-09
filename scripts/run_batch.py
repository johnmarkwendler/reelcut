"""GPU-offloaded reelcut run via Roboflow Batch Processing.

Stages the video, submits a GPU job running the saved reelcut-tracking
workflow, waits, exports JSONL results, then runs the local pipeline
(identity -> scoring -> cutting) against them. Costs Roboflow credits per
job — nothing is submitted with --dry-run.

Usage:
  uv run python scripts/run_batch.py --video game.mp4 --jersey 10 \
      --team-color blue --target-frame 1500 --target-box x,y,w,h \
      [--out output/batch_run] [--fps 5] [--machine-type gpu] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--jersey", required=True)
    ap.add_argument("--team-color", required=True)
    ap.add_argument("--target-frame", type=int, required=True)
    ap.add_argument("--target-box", required=True, metavar="X,Y,W,H")
    ap.add_argument("--out", type=Path, default=ROOT / "output/batch_run")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--machine-type", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--debug-video", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rf-cloud steps without submitting")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    from reelcut import batch
    from reelcut.config import ReelcutConfig

    cfg = ReelcutConfig()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    batch_id = args.batch_id or f"reelcut-{args.video.stem[:24]}-{stamp}".lower()
    results_dir = args.out / "batch_results"

    if args.dry_run:
        print("would run:")
        print(f"  1. stage_video({args.video}, batch_id={batch_id!r})")
        print(f"  2. submit_job(batch_id={batch_id!r}, workflow_id={cfg.workflow_id!r}, "
              f"fps={args.fps}, machine_type={args.machine_type!r})")
        print(f"  3. wait_for_job(...) ; export_results -> {results_dir}")
        print(f"  4. python -m reelcut --video {args.video} --batch-results {results_dir} ...")
        return 0

    print(f"[batch] staging {args.video} as {batch_id}")
    batch.stage_video(args.video, batch_id)
    print("[batch] submitting job")
    job_id = batch.submit_job(
        batch_id, cfg.workflow_id, args.fps, machine_type=args.machine_type
    )
    print(f"[batch] job {job_id} running — polling every 30s")
    batch.wait_for_job(job_id)
    results_batch = batch.results_batch_of(job_id)
    print(f"[batch] exporting results batch {results_batch}")
    batch.export_results(results_batch, results_dir)

    from reelcut.__main__ import main as reelcut_main
    cli = [
        "--video", str(args.video),
        "--jersey", args.jersey,
        "--team-color", args.team_color,
        "--target-frame", str(args.target_frame),
        "--target-box", args.target_box,
        "--out", str(args.out),
        "--batch-results", str(results_dir),
        "--fps", str(args.fps),
    ]
    if args.debug_video:
        cli.append("--debug-video")
    return reelcut_main(cli)


if __name__ == "__main__":
    raise SystemExit(main())
