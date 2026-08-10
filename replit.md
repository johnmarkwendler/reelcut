# reelcut

**Every game, their highlights.** Auto-generates soccer highlight reels using Roboflow computer vision — tracks players, reads jersey numbers, finds goals, and cuts clips.

## Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **ML/Vision**: Roboflow Workflows (rf-detr + BoT-SORT), OpenCV
- **Video**: imageio-ffmpeg (bundled binary, no system ffmpeg needed)
- **Package manager**: `uv`

## Running on Replit

The `Start application` workflow runs the web app:

```
cd /home/runner/workspace && uv run uvicorn reelcut.webapp:app --host 0.0.0.0 --port 5000
```

Visit the preview pane to use the app.

## Required secrets

| Secret | Where to get it |
|---|---|
| `ROBOFLOW_API_KEY` | https://app.roboflow.com → Settings → API |

## How to use the web app

1. **Drop a game video in** — MP4/MOV straight off the phone.
2. **Scrub to a moment** where your kid is clearly visible and hit *find players*.
3. **Click your kid**, enter their jersey number and kit color, choose how many goal clips you want.
4. Live progress streams in; results include the highlight reel, per-goal clips, and downloads.

## CLI usage

```bash
uv run python -m reelcut \
  --video game.mp4 --jersey 10 --team-color blue \
  --target-frame 1500 --target-box 640,320,40,90 \
  --out ./output/ --debug-video --fps 30 --max-goal-clips 2
```

## Development

```bash
uv sync          # install dependencies
uv run pytest    # ~200 tests, no network or GPU needed
```

Stage outputs cache under `out/cache/<video-key>/`; identity/scoring caches invalidate automatically when the target spec or config changes.

## Notes

- Jobs run one at a time; results land in `output/webapp/jobs/<id>/`
- System dependency `libGL` is required for OpenCV (installed via Nix)
- Appearance re-ID (reid.py) is implemented but disabled — full-crop CLIP can't separate same-team players reliably
- BoT-SORT camera-motion compensation is off by default (better on real footage)

## User preferences

_None recorded yet._
