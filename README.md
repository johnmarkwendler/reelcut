# reelcut

**Every game, their highlights.** Parents film entire kids' soccer games and the
footage dies on their phone. reelcut fixes that: prop the phone up, upload the
game, click your kid once — it tracks the ball and every player, reads jersey
numbers, finds the goals, and cuts a highlight reel.

Built **Roboflow-first**: detection, tracking, class-locking, box smoothing and
velocity run inside a [Roboflow Workflow](workflows/reelcut_tracking.json) on
models trained in the Roboflow UI. The app orchestrates, binds jersey numbers,
stitches identity, scores the action, and cuts clips.

```
video ─▶ Roboflow Workflow (rf-detr + BoT-SORT + class lock + stabilize + velocity)
      ─▶ jersey-number enrollment (3s majority vote per player)
      ─▶ identity stitching (seed click + number dominance + kit color + kinematics)
      ─▶ goal-mouth scoring (ball-in-net transients, action-gated)
      ─▶ clip cutting ─▶ highlights.mp4 + per-goal clips + annotated video
```

## Quickstart

```bash
uv sync                                  # Python 3.12 env (installs `inference`)
echo 'ROBOFLOW_API_KEY=...' > .env       # gitignored
```

No system ffmpeg needed — the binary bundled with `imageio-ffmpeg` is used.

### Web app (the intended experience)

```bash
uv run uvicorn reelcut.webapp:app --port 8008    # → http://localhost:8008
```

1. **Drop the game in** — MP4/MOV straight off the phone.
2. **Scrub to a moment** where your kid is clearly visible and hit
   *find players* — the detector draws clickable boxes on the real frame.
3. **Click your kid**, enter their jersey number and kit color, choose how
   many goal moments you want, and cut the reel.
4. Live stage progress streams in; results land as a cinematic page — the
   reel, per-goal clip cards (clean by default, per-clip *tracking overlay*
   toggle), and downloads.

Jobs run one at a time and everything lands in `output/webapp/jobs/<id>/`.

### CLI

```bash
uv run python -m reelcut \
  --video game.mp4 --jersey 10 --team-color blue \
  --target-frame 1500 --target-box 640,320,40,90 \
  --out ./output/ --debug-video --fps 30 --max-goal-clips 2
```

Useful flags:

| flag | effect |
|---|---|
| `--max-goal-clips N` | keep only the N strongest goal moments (0 = all) |
| `--debug-video` | render the annotated video + annotated clips |
| `--fps` | analysis sampling rate (default 5) |
| `--force-stage N` | recompute stage N and everything after |
| `--stub` | deterministic synthetic game — all stages run offline |
| `--batch-results DIR` | use exported Roboflow Batch GPU results as stage 1 |

Outputs per run: `highlights.mp4`, `highlights.json`, `clips/` (named per-goal
files, clean + `_annotated`), `annotated_full.mp4` (h264, browser-safe),
`debug.mp4`, and a stage cache under `cache/`.

## How it works

| stage | what happens |
|---|---|
| **1 — infer** | The Roboflow Workflow runs per sampled frame via InferencePipeline (in-process): rf-detr detection → per-class confidence filter (players 0.4, ball 0.4, goals 0.15) → BoT-SORT tracking → track class lock → box stabilization → velocity. |
| **1.5 — numbers** | Per-player **3-second enrollment**: every new track is read by the digit detector at every sampled frame from its *first successful read*; the majority of that window becomes the track's number for life and reads stop — the tracker carries it. A player re-entering the frame is a new track and re-enrolls. Labels can never flicker mid-track by construction. |
| **2 — identity** | Which tracks are the chosen kid: the seed click, jersey-read **dominance votes** (5:1 ratios, not absolute counts — a stray misread among hundreds of correct reads can't flip a track), team-kit color veto, and kinematic chaining. Conservative: UNKNOWN beats a wrong claim. |
| **3 — score** | Target involvement (proximity/possession/touch/sprint) blended with a player-agnostic **goal-mouth signal**: ball inside a detected goal box *at speed* (action-gated so parked balls and 2D depth illusions don't count). Goal boxes persist ~2.5s across detection dropouts — players occlude the goal exactly when shots happen. An adaptive threshold keeps dense footage from saturating the reel, and goal transients are judged on the **raw** (pre-smoothing) score so a shot's spike can't be diluted away. |
| **4 — cut** | Goal-only event selection (trimmed to buildup + celebration around each transient, optional top-N dial), padded/merged clip planning, frame-accurate ffmpeg cutting, and the annotated render (every source frame, tweened boxes, jersey numbers as labels). |

### The workflow

[`workflows/reelcut_tracking.json`](workflows/reelcut_tracking.json) is the
source of truth, mirrored in the `aarnavs-space` workspace as
`reelcut-tracking` (update path: edit JSON → validate → `workflows_update`).

### Models (trained in the Roboflow UI)

| model | id | role |
|---|---|---|
| rf-detr-small | `ia-foot-8ecu7/1` | players / ball / goals (mAP50 90.1) |
| rf-detr-nano | `jersey-number-detection-8a55j-ob8fb/1` | digit detection, classes 0–9 (mAP50 95.7) |

To fix systematic digit confusions (e.g. 9↔8) on your own footage:

```bash
uv run python scripts/export_digit_crops.py <run>/cache/<key> game.mp4 digit_crops/
```

exports crops pre-sorted by what the model read (`read_7/`, `read_8/`, …) —
skim, correct, upload to a Roboflow project, retrain, and update
`digit_model_id` in [config.py](src/reelcut/config.py).

## GPU offload (Roboflow Batch Processing)

Local in-process inference handles short clips comfortably; full games belong
on a cloud GPU:

```bash
uv run python scripts/run_batch.py --video game.mp4 --jersey 10 \
    --team-color blue --target-frame 1500 --target-box x,y,w,h --dry-run
```

Drop `--dry-run` to actually submit (**costs Roboflow credits per job**). The
script stages the video, runs `reelcut-tracking` on GPU, exports JSONL
results, and finishes identity/scoring/cutting locally in seconds.
`scripts/auto_batch_demo.py` runs the whole finisher hands-off — the same
sequence a hosted backend's webhook handler would run.

## Design decisions that were measured, not guessed

- **Numbers lock per track, not per frame.** Per-frame reads flicker; per-track
  aggregates are near-ground-truth (measured: 302 reads of "7" on one track).
  The enrollment vote is substring-aware and frequency-ranked, so a fused
  "71" (two kids in one crop) can't hijack a track of clean "7"s.
- **Appearance re-ID is implemented and disabled** (`reid.py`,
  `scripts/calibrate_reid.py`). Measured on real footage, full-crop CLIP cannot
  separate same-team kids — every player lands in a 0.88–0.95 similarity band.
  The machinery stays for a torso-crop or dedicated person-reID embedder.
- **BoT-SORT camera-motion compensation is off by default** (bake-off on real
  footage: 5 vs 21 ID switches; `scripts/compare_trackers.py`).
- **Shots vs goals are indistinguishable in 2D** (a saved rocket and a goal
  look identical without seeing the net bulge), so "just the goals" is a user
  dial (`--max-goal-clips`), not a fragile heuristic.
- **Ball confidence floor is 0.4** — at the old 0.15, cones and stray red
  objects registered as balls; the real ball scores 0.6+.

## Development

```bash
uv run pytest            # ~200 tests, no network or GPU needed
```

The stub client (`--stub`) synthesizes a full deterministic game so every
stage — identity, scoring, cutting, budgets — is testable offline. Stage
outputs cache under `out/cache/<video-key>/`; identity/scoring caches
invalidate automatically when the target spec or config changes.

Test fixtures under `data/fixtures/` are gitignored; sources and licenses are
documented in [SOURCE.md](data/fixtures/SOURCE.md) and
[SOURCE_GOALS.md](data/fixtures/SOURCE_GOALS.md).

## Known platform notes

- **Apple Silicon + torch 2.13**: `inference-models` crashes with
  `torch.mps has no attribute 'current_device'`;
  `workflow_client._prepare_inference_env()` patches it before importing
  `inference` (reported to Roboflow).
- **Serverless can't run this workflow**: tracking blocks are stateful video
  blocks — run via InferencePipeline locally (default) or Batch Processing,
  not per-request serverless.

## Known limits & roadmap

- **Tracker handovers**: BoT-SORT can slide a track between kids during heavy
  occlusion; the number-enrollment freeze keeps labels stable but identity can
  briefly ride the wrong kid. Fix direction: appearance drift detection along
  tracks (needs a better embedder than full-crop CLIP).
- **Goal semantics**: no net-crossing detection yet — goal-mouth transients +
  the top-N dial stand in. Pitch homography (real meters, kickoff-restart
  detection) is the next lever.
- **Hosted version**: the web app is local-first but shaped for hosting — the
  job runner is the same sequence the Batch GPU finisher runs.
