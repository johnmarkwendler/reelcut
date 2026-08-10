# Water Polo Reelcut Agent Guide

## Goal

Adapt the upstream Reelcut soccer prototype into an automatic water-polo
highlight pipeline while preserving its tested architecture:

1. Roboflow Workflow detection and BoT-SORT tracking.
2. Local cap-number enrollment and audit reads.
3. Conservative target-player tracklet stitching.
4. Goal-candidate ranking and ffmpeg clip export.

## Fixed Roboflow configuration

- Workspace: `water-polo-tracking`
- Workflow: `reelcut-tracking` (forked from upstream)
- Detector: `water-polo-tracking/waterpolo-player-detection-5-rfdetr-medium-t1`
- Detector classes: `player`, `referee`, `ball`, `goalie`, `goal`
- Digit detector: `jersey-number-detection-8a55j-ob8fb/1` (upstream model)

Do not replace these with soccer models or rename classes without explicit user
approval. The detector identifier must be validated on the first live run; do
not invent a `/N` version suffix if Roboflow rejects it.

## Water-polo semantics

- Treat `goalie` as the goalkeeper/player class.
- Read digits only from the cap/head region, not the torso.
- Use cap color for team vetoes, not jersey/kit color.
- Expect small player boxes, splashing, ball occlusion, crowded goal mouths,
  camera pans, and intermittent goal detection.
- Prefer `UNKNOWN` over attaching the wrong player tracklet.
- Keep raw-frame goal transients separate from smoothed involvement scores.

## Safe execution

- Never commit `.env` or print `ROBOFLOW_API_KEY`.
- Run offline tests before live inference.
- Do not submit paid Batch Processing jobs without explicit approval.
- Cache stage-1 detections so scoring changes do not rerun inference.
- Preserve the annotated debug video and exported cap crops for error analysis.

## Verification order

1. `UV_CACHE_DIR=/tmp/reelcut-uv-cache uv sync`
2. `UV_CACHE_DIR=/tmp/reelcut-uv-cache uv run pytest`
3. Validate the detector ID and Workflow outputs on one frame or a short clip.
4. Inspect clickable player boxes and cap crops.
5. Run a short end-to-end clip before processing a full game.

