"""Export jersey crops from a cached run for digit-model fine-tuning.

The digit model's systematic confusions (measured: 9 read as 8) are cured by
fine-tuning on corrected crops from the user's own footage. This exports the
enrollment-window crops, pre-sorted into read_<digits>/ folders so correcting
labels in Roboflow Annotate is a 20-minute skim, not a labeling project.

Usage:
  uv run python scripts/export_digit_crops.py <cache_dir> <video> <out_dir>

Then: create/reuse a Roboflow object-detection project, drag the folders in,
fix the wrong ones, retrain rfdetr-nano, and update digit_model_id in config.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reelcut.types import from_jsonable  # noqa: E402

_MAX_PER_TRACK = 6      # spread; a fine-tune wants variety, not 300 near-dupes
_MARGIN = 0.10


def main() -> None:
    cache_dir, video, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with gzip.open(cache_dir / "stage1_observations.json.gz", "rt") as f:
        frames = from_jsonable(json.load(f))
    with gzip.open(cache_dir / "stage2_identity.json.gz", "rt") as f:
        payload = from_jsonable(json.load(f))

    # (frame_index, track_id) -> read text; bbox looked up from observations
    reads = defaultdict(list)
    for fi, r in payload.get("number_reads") or []:
        reads[int(fi)].append(r)
    boxes = {
        (f.frame_index, p.track_id): p.bbox
        for f in frames for p in f.players
    }

    per_track: dict[int, int] = defaultdict(int)
    wanted: dict[int, list] = defaultdict(list)
    for fi in sorted(reads):
        for r in reads[fi]:
            if per_track[r.track_id] >= _MAX_PER_TRACK:
                continue
            bbox = boxes.get((fi, r.track_id))
            if bbox is None:
                continue
            per_track[r.track_id] += 1
            digits = "".join(c for c in r.text if c.isdigit()) or "none"
            wanted[fi].append((r.track_id, bbox, digits))

    cap = cv2.VideoCapture(str(video))
    n_out = 0
    try:
        pending = sorted(wanted)
        pi = 0
        src_idx = 0
        while pi < len(pending):
            ok, img = cap.read()
            if not ok or img is None:
                break
            if src_idx == pending[pi]:
                fh, fw = img.shape[:2]
                for tid, bbox, digits in wanted[src_idx]:
                    mx, my = bbox.w * _MARGIN, bbox.h * _MARGIN
                    x0 = max(0, int(bbox.x - mx)); y0 = max(0, int(bbox.y - my))
                    x1 = min(fw, int(bbox.x2 + mx)); y1 = min(fh, int(bbox.y2 + my))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    d = out_dir / f"read_{digits}"
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(d / f"f{src_idx:05d}_t{tid}.jpg"),
                        img[y0:y1, x0:x1],
                    )
                    n_out += 1
                pi += 1
            src_idx += 1
    finally:
        cap.release()
    print(f"exported {n_out} crops into {out_dir}/read_<digits>/ "
          f"({len(per_track)} tracks)")


if __name__ == "__main__":
    main()
