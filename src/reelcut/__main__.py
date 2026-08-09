"""CLI: python -m reelcut --video game.mp4 --jersey 10 --team-color blue \
--target-frame 1500 --target-box x,y,w,h --out ./output/

Flags:
  --video PATH (required)      --jersey STR (required)
  --team-color STR (required)  --target-frame INT (required)
  --target-box X,Y,W,H (required, pixels)
  --out DIR (default ./output)
  --debug-video                render annotated debug.mp4
  --force-stage N              recompute stage N and later from cache
  --stub                       use StubWorkflowClient (no Roboflow needed)
  --workflow ID                workspace workflow slug (default from config)
  --model-id ID                override detection model (workflow parameter)
  --api-url URL                inference server URL (default in-process)
  --fps FLOAT                  sampling fps (default 5.0)
  --seed INT                   determinism seed
  --sport NAME                 sport preset (default soccer)

Reads ROBOFLOW_API_KEY from env / .env (python-dotenv). Exits 2 on bad args,
1 with a clear message when the seed click matches no tracklet.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import TEAM_COLORS, Paths, ReelcutConfig
from .types import BBox, TargetSpec


def parse_box(s: str):
    """'x,y,w,h' -> BBox; raises argparse-friendly ValueError."""
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"expected X,Y,W,H (4 comma-separated numbers), got {s!r}"
        )
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected X,Y,W,H (4 comma-separated numbers), got {s!r}"
        ) from None
    return BBox(x, y, w, h)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelcut",
        description="Cut a one-kid highlight reel from a full game video.",
    )
    parser.add_argument("--video", type=Path, required=True, help="source game video")
    parser.add_argument("--jersey", required=True, help="target jersey number, e.g. 10")
    parser.add_argument(
        "--team-color",
        required=True,
        choices=sorted(TEAM_COLORS),
        help="target kit color",
    )
    parser.add_argument(
        "--target-frame", type=int, required=True,
        help="source frame index of the seed click",
    )
    parser.add_argument(
        "--target-box", type=parse_box, required=True, metavar="X,Y,W,H",
        help="clicked box around the kid, pixels",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./output"), help="output directory"
    )
    parser.add_argument(
        "--debug-video", action="store_true", help="render annotated debug.mp4"
    )
    parser.add_argument(
        "--force-stage", type=int, choices=(1, 2, 3, 4), default=None, metavar="N",
        help="recompute stage N and later from cache",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="use StubWorkflowClient (no Roboflow needed)",
    )
    parser.add_argument(
        "--batch-results", type=Path, default=None, metavar="DIR",
        help="use exported Roboflow Batch Processing results (GPU offload) "
             "instead of running inference locally; see scripts/run_batch.py",
    )
    parser.add_argument(
        "--workflow", default=None, metavar="ID", help="workspace workflow slug"
    )
    parser.add_argument(
        "--model-id", default=None, metavar="ID",
        help="override detection model (workflow parameter)",
    )
    parser.add_argument(
        "--api-url", default=None, metavar="URL",
        help="inference server URL (default in-process)",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="sampling fps (default 5.0)"
    )
    parser.add_argument("--seed", type=int, default=None, help="determinism seed")
    parser.add_argument(
        "--sport", default="soccer", metavar="NAME", help="sport preset"
    )
    parser.add_argument(
        "--max-goal-clips", type=int, default=None, metavar="N",
        help="keep only the N strongest goal moments (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fps is not None and not (math.isfinite(args.fps) and args.fps > 0):
        parser.error(f"--fps must be a positive number, got {args.fps}")
    if args.api_url is not None:
        print(
            "reelcut: --api-url is accepted but remote execution is not "
            "implemented yet; inference runs in-process on this machine.",
            file=sys.stderr,
        )

    load_dotenv()

    cfg = ReelcutConfig()
    overrides: dict[str, Any] = {}
    if args.fps is not None:
        overrides["sample_fps"] = args.fps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.workflow is not None:
        overrides["workflow_id"] = args.workflow
    if args.model_id is not None:
        overrides["model_id"] = args.model_id
    if args.api_url is not None:
        overrides["api_url"] = args.api_url
    if args.max_goal_clips is not None:
        overrides["max_goal_clips"] = args.max_goal_clips
    if overrides:
        cfg = replace(cfg, **overrides)
    try:
        cfg = cfg.for_sport(args.sport)
    except ValueError as e:
        parser.error(str(e))

    # Deferred: pipeline pulls in the full stage stack.
    from .pipeline import run_pipeline
    from .workflow_client import RoboflowWorkflowClient, StubWorkflowClient

    if args.stub and args.batch_results:
        parser.error("--stub and --batch-results are mutually exclusive")
    if args.stub:
        client: Any = StubWorkflowClient(cfg.seed, args.jersey)
    elif args.batch_results:
        from .batch import BatchResultsClient

        client = BatchResultsClient(args.batch_results)
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            print(
                "reelcut: ROBOFLOW_API_KEY is not set (env or .env). "
                "Set it, or pass --stub to run without Roboflow.",
                file=sys.stderr,
            )
            return 2
        client = RoboflowWorkflowClient(
            api_key=api_key,
            workspace=cfg.workspace,
            workflow_id=cfg.workflow_id,
            api_url=cfg.api_url,
        )

    spec = TargetSpec(
        jersey=args.jersey,
        team_color=args.team_color,
        seed_frame_index=args.target_frame,
        seed_box=args.target_box,
    )
    paths = Paths(out_dir=args.out)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        args.video,
        spec,
        cfg,
        paths,
        client,
        force_stage=args.force_stage,
        debug_video=args.debug_video,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
