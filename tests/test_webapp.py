"""Web app smoke tests — no model, no video processing."""
from __future__ import annotations

from fastapi.testclient import TestClient

from reelcut.webapp import app

client = TestClient(app)


def test_index_serves_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "reel" in r.text and "Upload" in r.text


def test_unknown_video_404s():
    assert client.get("/api/videos/nope/frame.jpg?t=1").status_code == 404


def test_unknown_job_404s():
    assert client.get("/api/jobs/nope").status_code == 404


def test_bad_upload_rejected(tmp_path):
    r = client.post(
        "/api/videos",
        files={"file": ("junk.mp4", b"not a video at all", "video/mp4")},
    )
    assert r.status_code == 422
