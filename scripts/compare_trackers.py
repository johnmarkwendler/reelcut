"""Phase 1 tracker bake-off: BoT-SORT (CMC on/off) vs ByteTrack vs OC-SORT.

Runs the reelcut tracking workflow on a slice of the fixture with the tracker
step swapped per variant (OCR branch stripped — it is tracker-independent and
dominates CPU time), then reports ground-truth-free stability proxies:

  * unique_ids        — fewer is better once coverage is equal
  * mean tracks/frame — coverage; should match across variants
  * id_switches       — track dies, a NEW id is born within 1.5 s at IoU>0.3
                        of the dead track's last box (fragmentation proxy)
  * mean track dur    — longer is better

Usage:
  uv run python scripts/compare_trackers.py [--duration 45] [--fps 5]
      [--video data/fixtures/game_2min.mp4] [--model ia-foot-8ecu7/1]

Writes output/tracker_comparison.json and prints a table.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from reelcut.ffmpeg import ffmpeg_exe
from reelcut.workflow_client import _shim_torch_mps

# Shared knobs verified to exist on the named manifests (OC-SORT has no
# track_activation_threshold, so it is set per-variant).
_SHARED = {"lost_track_buffer": 30, "minimum_consecutive_frames": 2}

TRACKER_VARIANTS: dict[str, dict] = {
    "botsort_cmc": {
        "type": "roboflow_core/trackers_botsort@v1",
        "enable_cmc": True,
        "cmc_method": "sparseOptFlow",
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
        **_SHARED,
    },
    "botsort_no_cmc": {
        "type": "roboflow_core/trackers_botsort@v1",
        "enable_cmc": False,
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
        **_SHARED,
    },
    "bytetrack": {
        "type": "roboflow_core/trackers_bytetrack@v1",
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
        **_SHARED,
    },
    "ocsort": {
        "type": "roboflow_core/trackers_ocsort@v1",
        "high_conf_det_threshold": 0.5,
        **_SHARED,
    },
}


def build_variant(base_spec: dict, tracker_overrides: dict) -> dict:
    spec = copy.deepcopy(base_spec)
    drop_steps = {"jersey_crops", "ocr", "stitch_ocr"}
    spec["steps"] = [s for s in spec["steps"] if s["name"] not in drop_steps]
    spec["outputs"] = [
        o for o in spec["outputs"] if o["name"] in ("tracked_players", "ball")
    ]
    for step in spec["steps"]:
        if step["name"] == "tracker":
            keep = {"name": step["name"], "image": step["image"], "detections": step["detections"]}
            step.clear()
            step.update(keep)
            step.update(tracker_overrides)
    return spec


def run_variant(spec: dict, video: Path, fps: float, api_key: str) -> list[dict]:
    from inference import InferencePipeline

    rows: list[dict] = []

    def sink(predictions, video_frame) -> None:
        tp = predictions.get("tracked_players")
        ids = [] if tp is None or tp.tracker_id is None else [int(i) for i in tp.tracker_id]
        boxes = [] if tp is None else [[float(v) for v in b] for b in tp.xyxy]
        rows.append({"frame_id": int(video_frame.frame_id), "ids": ids, "boxes": boxes})

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=str(video),
        workflow_specification=spec,
        api_key=api_key,
        on_prediction=sink,
        max_fps=fps,
    )
    pipeline.start()
    pipeline.join()
    return rows


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def metrics(rows: list[dict], source_fps: float) -> dict:
    """frame_id is the SOURCE decode counter, so windows/durations use source_fps."""
    if not rows:
        return {"frames": 0}
    last_seen: dict[int, tuple[int, list[float]]] = {}
    span: dict[int, tuple[int, int]] = {}          # tid -> (first_fid, last_fid)
    switches = 0
    switch_window = int(round(1.5 * source_fps))   # source frames
    warmup = rows[0]["frame_id"] + int(round(2.0 * source_fps))
    births_after_warmup = 0
    for row in rows:
        fid = row["frame_id"]
        present = set(row["ids"])
        for tid, box in zip(row["ids"], row["boxes"]):
            if tid not in span:
                span[tid] = (fid, fid)
                if fid > warmup:
                    births_after_warmup += 1
                    # born where a recently-dead track vanished? -> id switch
                    for dead_id, (dead_fid, dead_box) in last_seen.items():
                        if (
                            dead_id not in present
                            and 0 < fid - dead_fid <= switch_window
                            and _iou(box, dead_box) > 0.3
                        ):
                            switches += 1
                            break
            else:
                span[tid] = (span[tid][0], fid)
            last_seen[tid] = (fid, box)
        # forget tracks dead longer than the window
        last_seen = {
            t: (f, b) for t, (f, b) in last_seen.items()
            if fid - f <= switch_window or t in present
        }
    durs = [(hi - lo) / source_fps for lo, hi in span.values()]
    return {
        "frames": len(rows),
        "unique_ids": len(span),
        "mean_tracks_per_frame": round(sum(len(r["ids"]) for r in rows) / len(rows), 2),
        "id_switches": switches,
        "births_after_warmup": births_after_warmup,
        "mean_track_duration_s": round(sum(durs) / len(durs), 2) if durs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "data/fixtures/game_2min.mp4")
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--model", default=None, help="override model_id parameter default")
    ap.add_argument("--out", type=Path, default=ROOT / "output/tracker_comparison.json")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    import os

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY missing", file=sys.stderr)
        return 2

    _shim_torch_mps()

    base_spec = json.loads((ROOT / "workflows/reelcut_tracking.json").read_text())
    if args.model:
        for inp in base_spec["inputs"]:
            if inp["name"] == "model_id":
                inp["default_value"] = args.model

    slice_path = ROOT / "output" / f"_tracker_slice_{int(args.duration)}s.mp4"
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    if not slice_path.exists():
        subprocess.run(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(args.video),
             "-t", str(args.duration), "-c", "copy", str(slice_path)],
            check=True,
        )

    from reelcut.ffmpeg import probe

    source_fps = probe(slice_path).fps

    report: dict[str, dict] = {}
    for name, overrides in TRACKER_VARIANTS.items():
        spec = build_variant(base_spec, overrides)
        t0 = time.monotonic()
        rows = run_variant(spec, slice_path, args.fps, api_key)
        wall = time.monotonic() - t0
        report[name] = metrics(rows, source_fps) | {"wall_s": round(wall, 1)}
        print(f"{name:16s} {report[name]}")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
