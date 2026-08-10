"""reelcut web app — upload a game, click your kid, get their highlight reel.

Local-first v1 (the same flow a hosted version would use): FastAPI serves a
single-page UI; uploads land in output/webapp/uploads/; the seed picker runs
the detector on the chosen frame so the parent clicks a real box; jobs run
the normal CLI pipeline in a subprocess (one at a time) and results are
served straight from the job's output folder.

Run:  uv run uvicorn reelcut.webapp:app --port 8008
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import ReelcutConfig

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = ROOT / "output" / "webapp"
UPLOADS = WEB_DIR / "uploads"
JOBS_DIR = WEB_DIR / "jobs"
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="reelcut")
app.mount("/files", StaticFiles(directory=WEB_DIR), name="files")

_cfg = ReelcutConfig()
_detector = None
_detector_lock = threading.Lock()


def _get_detector():
    global _detector
    with _detector_lock:
        if _detector is None:
            from .workflow_client import _prepare_inference_env

            _prepare_inference_env()
            from inference import get_model

            _detector = get_model(
                _cfg.model_id, api_key=os.environ["ROBOFLOW_API_KEY"]
            )
        return _detector


@dataclass
class Job:
    id: str
    state: str = "queued"           # queued | running | done | failed
    log: list[str] = field(default_factory=list)
    out_dir: Path | None = None


_jobs: dict[str, Job] = {}
_job_lock = threading.Lock()
_active_job: str | None = None


def _video_path(video_id: str) -> Path:
    p = UPLOADS / f"{video_id}.mp4"
    if not p.exists():
        raise HTTPException(404, "unknown video")
    return p


def _probe(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0, 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, frame = cap.read()
        if not ok or frame is None:
            return 0.0, 0.0
        return (n / fps if fps else 0.0), fps
    finally:
        cap.release()


def _read_frame(path: Path, t: float):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, img = cap.read()
        if not ok or img is None:
            raise HTTPException(422, f"cannot read frame at t={t}")
        return img
    finally:
        cap.release()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)) -> dict:
    import shutil

    video_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{video_id}.mp4"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    duration_s, fps = _probe(dest)
    if duration_s <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "not a readable video")
    return {
        "video_id": video_id,
        "duration_s": duration_s,
        "fps": fps,
        "url": f"/files/uploads/{video_id}.mp4",
    }


@app.get("/api/videos/{video_id}/frame.jpg")
def frame_jpg(video_id: str, t: float = 0.0) -> Response:
    img = _read_frame(_video_path(video_id), t)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/videos/{video_id}/detections")
def detections(video_id: str, t: float = 0.0) -> dict:
    """Player boxes on the frame at t — what the picker UI makes clickable."""
    img = _read_frame(_video_path(video_id), t)
    model = _get_detector()
    r = model.infer(img, confidence=0.4)
    preds = r[0].predictions if isinstance(r, list) else r.predictions
    player_classes = {c.lower() for c in _cfg.player_classes}
    players = [
        {
            "x": float(p.x - p.width / 2), "y": float(p.y - p.height / 2),
            "w": float(p.width), "h": float(p.height),
            "conf": float(p.confidence),
        }
        for p in preds
        if str(p.class_name).lower() in player_classes
    ]
    h, w = img.shape[:2]
    return {"frame_w": w, "frame_h": h, "players": players}


def _run_job(job: Job, cmd: list[str]) -> None:
    global _active_job
    job.state = "running"
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job.log.append(line)
        job.state = "done" if proc.wait() == 0 else "failed"
    except Exception as e:                                    # noqa: BLE001
        job.log.append(f"error: {e}")
        job.state = "failed"
    finally:
        with _job_lock:
            _active_job = None


@app.post("/api/jobs")
def start_job(
    video_id: str = Form(...),
    jersey: str = Form(...),
    team_color: str = Form(...),
    seed_t: float = Form(...),
    seed_x: float = Form(...),
    seed_y: float = Form(...),
    seed_w: float = Form(...),
    seed_h: float = Form(...),
    fps: float = Form(10.0),
    max_goal_clips: int = Form(0),
) -> dict:
    global _active_job
    video = _video_path(video_id)
    _, src_fps = _probe(video)
    with _job_lock:
        if _active_job is not None:
            raise HTTPException(409, "a job is already running")
        job = Job(id=uuid.uuid4().hex[:12])
        _active_job = job.id
        _jobs[job.id] = job
    job.out_dir = JOBS_DIR / job.id
    cmd = [
        sys.executable, "-u", "-m", "reelcut",   # -u: unbuffered, so stage
                                                 # lines stream to the UI live
        "--video", str(video),
        "--jersey", jersey,
        "--team-color", team_color,
        "--target-frame", str(round(seed_t * src_fps)),
        "--target-box", f"{seed_x:.0f},{seed_y:.0f},{seed_w:.0f},{seed_h:.0f}",
        "--out", str(job.out_dir),
        "--debug-video",
        "--fps", str(fps),
    ]
    if max_goal_clips > 0:
        cmd += ["--max-goal-clips", str(max_goal_clips)]
    threading.Thread(target=_run_job, args=(job, cmd), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    out: dict = {
        "state": job.state,
        "log": [ln for ln in job.log if ln.startswith("[stage")][-8:],
    }
    if job.state == "failed":
        out["log"] = job.log[-15:]
    if job.state == "done" and job.out_dir is not None:
        rel = f"/files/jobs/{job.id}"
        clips_dir = job.out_dir / "clips"
        out["results"] = {
            "highlights": f"{rel}/highlights.mp4",
            "annotated_full": f"{rel}/annotated_full.mp4",
            "clips": sorted(
                f"{rel}/clips/{p.name}" for p in clips_dir.glob("*.mp4")
            ) if clips_dir.exists() else [],
        }
    return out


@app.get("/api/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None or job.out_dir is None:
        raise HTTPException(404, "unknown job")
    p = (job.out_dir / name).resolve()
    if not p.is_file() or job.out_dir.resolve() not in p.parents:
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=p.name)
