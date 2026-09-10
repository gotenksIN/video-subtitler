"""CLI parsing and offline validation through the executable boundary."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "gemini_lyrics.py"


def run_cli(*arguments, cwd):
    environment = dict(os.environ)
    environment["GEMINI_API_KEY"] = ""
    environment["GEMINI_API_BASE"] = "http://127.0.0.1:9/unreachable"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_missing_audio_is_reported_without_traceback(tmp_path):
    process = run_cli("missing.flac", "--api-key", "key", cwd=tmp_path)
    assert process.returncode == 1
    assert "Audio file not found" in process.stderr
    assert "Traceback" not in process.stderr


def test_invalid_target_language_is_rejected(tmp_path):
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"audio")
    process = run_cli(
        str(audio),
        "--api-key",
        "key",
        "--target-language",
        "not a tag",
        cwd=tmp_path,
    )
    assert process.returncode == 1
    assert "BCP 47" in process.stderr


def test_existing_reference_sidecar_is_never_replaced(tmp_path):
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"audio")
    reference = tmp_path / "song.lrc"
    reference.write_text("reference\n", encoding="utf-8")
    process = run_cli(str(audio), "--api-key", "key", cwd=tmp_path)
    assert process.returncode == 1
    assert "Output already exists" in process.stderr
    assert reference.read_text(encoding="utf-8") == "reference\n"
