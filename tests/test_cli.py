"""CLI-surface tests: parse_box, required-flag rejection, flag validation,
``--help``.

Mostly touches no other stage module — the CLI parses and exits before any
pipeline work; the --api-url warning test stubs out run_pipeline instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from reelcut.__main__ import main, parse_box
from reelcut.types import BBox

REQUIRED = [
    "--video", "game.mp4",
    "--jersey", "10",
    "--team-color", "blue",
    "--target-frame", "1500",
    "--target-box", "10,20,30,40",
]


# --------------------------------------------------------------------------- #
# parse_box
# --------------------------------------------------------------------------- #

def test_parse_box_valid_ints() -> None:
    assert parse_box("10,20,30,40") == BBox(10.0, 20.0, 30.0, 40.0)


def test_parse_box_valid_floats_and_spaces() -> None:
    assert parse_box("1.5, 2.25, 3.0, 4.75") == BBox(1.5, 2.25, 3.0, 4.75)


@pytest.mark.parametrize(
    "bad",
    ["", "1,2,3", "1,2,3,4,5", "a,b,c,d", "1;2;3;4", "1,2,3,x", "10 20 30 40"],
)
def test_parse_box_malformed(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_box(bad)


# --------------------------------------------------------------------------- #
# argparse: required flags
# --------------------------------------------------------------------------- #

def test_no_args_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_partial_required_flags_exit_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--video", "game.mp4", "--jersey", "10"])
    assert exc.value.code == 2


def test_malformed_target_box_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main([
            "--video", "game.mp4",
            "--jersey", "10",
            "--team-color", "blue",
            "--target-frame", "1500",
            "--target-box", "not-a-box",
        ])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# flag validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_fps", ["0", "-2", "nan", "inf"])
def test_non_positive_fps_exits_2(bad_fps: str) -> None:
    # Regression: --fps 0 used to reach the stub/scoring and die with a raw
    # ZeroDivisionError instead of a clean argparse error.
    with pytest.raises(SystemExit) as exc:
        main([*REQUIRED, "--stub", "--fps", bad_fps])
    assert exc.value.code == 2


def test_api_url_warns_inference_runs_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # --api-url is accepted but remote execution is not implemented; the CLI
    # must say so instead of silently running inference in-process.
    from reelcut import pipeline

    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **k: None)
    rc = main([
        *REQUIRED, "--stub", "--out", str(tmp_path / "out"),
        "--api-url", "http://gpu-box:9001",
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "--api-url" in err
    assert "in-process" in err


# --------------------------------------------------------------------------- #
# python -m reelcut --help
# --------------------------------------------------------------------------- #

def test_module_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "reelcut", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()
    assert "--target-box" in proc.stdout
