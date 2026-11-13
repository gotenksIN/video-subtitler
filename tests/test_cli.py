"""CLI contracts through isolated subprocess invocations."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from modules import io

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:9/unreachable"

BOOTSTRAP = """
import json, runpy, sys
from pathlib import Path
from modules import gemini
from tests.support.gemini_fakes import (
    ScriptedGeminiClient, research_call, chunk_call, audio_call, text_call,
)
refine = "--refine-only" in sys.argv
client = ScriptedGeminiClient(
    [research_call(), text_call()] if refine else
    [research_call(), chunk_call(), chunk_call(), audio_call(), text_call()]
)
gemini.genai.Client = client.create_client
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as error:
    if error.code:
        raise
assert not client.pending
Path("requests.json").write_text(json.dumps({
    "credentials": [[request.headers["x-goog-api-key"], request.http_options.base_url]
                    for request in client.requests],
    "requests": [[request.kind, request.model] for request in client.requests],
}))
"""


def run_cli(tmp_path, *arguments, env=None, fake=False, dotenv=False):
    script = tmp_path / "gemini_subs.py"
    shutil.copyfile(ROOT / "gemini_subs.py", script)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GEMINI_", "GOOGLE_"))
        and key != "PYTHON_DOTENV_DISABLED"
    }
    environment.update(
        {
            "GEMINI_API_BASE": BASE_URL,
            "PYTHONPATH": str(ROOT),
        }
    )
    if not dotenv:
        environment["GEMINI_API_KEY"] = ""
    environment.update(env or {})
    command = [sys.executable]
    if fake:
        command += ["-c", BOOTSTRAP]
    result = subprocess.run(
        [*command, str(script), *map(str, arguments)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert "Traceback" not in result.stderr
    return result


def test_help_reports_usage_without_credentials(tmp_path):
    result = run_cli(tmp_path, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--refine-only" in result.stdout


def test_usage_error_equivalence_classes(tmp_path):
    for arguments, message in (
        ([], "required"),
        (["source.mp4", "--unknown"], "unrecognized arguments"),
        (["source.mp4", "--thinking-level", "minimal"], "invalid choice"),
        (["source.mp4", "--chunk-dur", "soon"], "invalid int value"),
    ):
        result = run_cli(tmp_path, *arguments)
        assert result.returncode == 2
        assert message in result.stderr


def test_generation_validation_diagnostics_precede_media_and_api(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"invalid media")
    for name, options, message in (
        ("missing.mp4", [], "Video file not found"),
        ("source.mp4", [], "API key not configured"),
        ("source.mp4", ["--chunk-dur", "0"], "chunk-dur must be greater than 0"),
        ("source.mp4", ["--workers", "0"], "workers must be greater than 0"),
        ("source.mp4", ["--api-key", "test"], "Failed to inspect media"),
    ):
        result = run_cli(tmp_path, name, *options)
        assert result.returncode == 1
        assert message in result.stderr
        assert not (tmp_path / "temp_video_chunks").exists()


def test_output_parent_checks_precede_media_and_api_in_both_modes(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"invalid media")
    parent_file = tmp_path / "file"
    parent_file.write_text("not a directory", encoding="ascii")
    for mode in ([], ["--refine-only"]):
        for parent in ("missing", "file"):
            result = run_cli(
                tmp_path,
                source,
                *mode,
                "--output",
                f"{parent}/out.vtt",
            )
            assert result.returncode == 1
            assert "Output parent is not a directory" in result.stderr
            assert result.stdout == ""


def test_invalid_context_url_precedes_missing_input_checks_in_both_modes(tmp_path):
    for mode in ([], ["--refine-only"]):
        result = run_cli(tmp_path, "missing.mp4", *mode, "--context-url", "ftp://bad")
        assert result.returncode == 1
        assert "Invalid --context-url" in result.stderr
        assert "not found" not in result.stderr


def test_output_cannot_resolve_to_source_through_a_symlink(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(source)
    result = run_cli(tmp_path, source, "--output", alias, "--api-key", "test")
    assert result.returncode == 1
    assert "must not resolve to the source video" in result.stderr
    assert source.read_bytes() == b"source"


def test_refine_only_reports_missing_key_input_and_invalid_vtt(tmp_path):
    source = tmp_path / "source.vtt"
    source.write_text("invalid VTT", encoding="ascii")
    for name, options, message in (
        ("missing.vtt", ["--api-key", "test"], "Input VTT file not found"),
        ("source.vtt", [], "API key not configured"),
        ("source.vtt", ["--api-key", "test"], "Invalid format"),
    ):
        result = run_cli(tmp_path, name, "--refine-only", *options)
        assert result.returncode == 1
        assert message in result.stderr


def test_dotenv_environment_and_cli_precedence_include_empty_model_fallbacks(
    tmp_path, media_files
):
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-key\nGEMINI_MODEL=dotenv-generation\n"
        "GEMINI_REFINE_MODEL=dotenv-text\nGEMINI_AUDIO_REFINE_MODEL=dotenv-audio\n",
        encoding="ascii",
    )
    for env, options, key, models in (
        (
            {},
            [],
            "dotenv-key",
            {
                "chunk": "dotenv-generation",
                "audio": "dotenv-audio",
                "text": "dotenv-text",
            },
        ),
        (
            {
                "GEMINI_API_KEY": "env-key",
                "GEMINI_MODEL": "env-generation",
                "GEMINI_REFINE_MODEL": "",
                "GEMINI_AUDIO_REFINE_MODEL": "",
            },
            [],
            "env-key",
            {
                "chunk": "env-generation",
                "audio": "gemini-4-hydrogen",
                "text": "gemini-4-hydrogen",
            },
        ),
        (
            {"GEMINI_API_KEY": "env-key", "GEMINI_MODEL": "env-generation"},
            [
                "--api-key",
                "cli-key",
                "--model",
                "cli-generation",
                "--refine-model",
                "cli-text",
                "--audio-refine-model",
                "cli-audio",
            ],
            "cli-key",
            {"chunk": "cli-generation", "audio": "cli-audio", "text": "cli-text"},
        ),
    ):
        result = run_cli(
            tmp_path,
            media_files["h264"],
            "--chunk-dur",
            "1",
            *options,
            env=env,
            fake=True,
            dotenv=True,
        )
        assert result.returncode == 0, result.stderr
        recorded = json.loads((tmp_path / "requests.json").read_text())
        assert recorded["credentials"]
        assert all(value == [key, BASE_URL] for value in recorded["credentials"])
        for kind, model in recorded["requests"]:
            assert model == models["chunk" if kind == "research" else kind]
        assert len(io.read_plain_vtt(tmp_path / "output_subtitles.vtt")) == 2


def test_refine_only_success_uses_grounded_context_without_media_work(tmp_path):
    source = tmp_path / "source.vtt"
    source.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHOST: A &amp; B\n",
        encoding="ascii",
    )
    for env, options, model in (
        ({}, [], "gemini-4-hydrogen"),
        ({"GEMINI_MODEL": "ignored-generation"}, [], "gemini-4-hydrogen"),
        ({}, ["--model", "cli-fallback"], "cli-fallback"),
        (
            {"GEMINI_REFINE_MODEL": ""},
            ["--model", "cli-fallback"],
            "cli-fallback",
        ),
        (
            {"GEMINI_REFINE_MODEL": "env-text"},
            ["--model", "cli-fallback"],
            "env-text",
        ),
        (
            {"GEMINI_REFINE_MODEL": "env-text"},
            ["--model", "cli-fallback", "--refine-model", "cli-text"],
            "cli-text",
        ),
    ):
        result = run_cli(
            tmp_path,
            source,
            "--refine-only",
            "--api-key",
            "test",
            *options,
            env=env,
            fake=True,
        )
        assert result.returncode == 0, result.stderr
        assert (
            io.read_plain_vtt(tmp_path / "output_subtitles.vtt")[0].raw_text
            == "Host: A & B"
        )
        assert not (tmp_path / "temp_video_chunks").exists()
        recorded = json.loads((tmp_path / "requests.json").read_text())["requests"]
        assert recorded == [["research", model], ["text", model]]
