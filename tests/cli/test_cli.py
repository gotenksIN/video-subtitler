"""CLI behavior through real subprocess invocations.

Only help, parsing, validation, mode routing, and exit diagnostics are
covered. No test reaches a live API request.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "gemini_subs.py"
SCRUBBED_ENV = (
    "GEMINI_API_KEY",
    "GEMINI_API_BASE",
    "GEMINI_MODEL",
    "GEMINI_REFINE_MODEL",
)


def run_cli(*arguments, cwd, extra_env=None):
    # Pin empty credentials so the repository .env cannot leak a real API
    # key into subprocesses. The unreachable base URL guarantees that any
    # accidental API attempt fails fast instead of reaching a real endpoint.
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env["GEMINI_API_KEY"] = ""
    env["GEMINI_API_BASE"] = "http://127.0.0.1:9/unreachable"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def output(process):
    return f"{process.stdout}\n{process.stderr}"


def make_video(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not a real video")
    return video


def test_help_exits_zero_and_documents_every_cli_option(tmp_path):
    process = run_cli("--help", cwd=tmp_path)

    assert process.returncode == 0
    text = output(process)
    for option in (
        "--output",
        "--api-key",
        "--base-url",
        "--model",
        "--refine-model",
        "--disable-text-refine",
        "--refine-only",
        "--chunk-dur",
        "--overlap",
        "--workers",
        "--thinking-level",
        "--context-url",
    ):
        assert option in text
    assert "usage:" in text


def test_missing_positional_argument_is_a_usage_error(tmp_path):
    process = run_cli(cwd=tmp_path)

    assert process.returncode == 2
    assert "the following arguments are required" in output(process)


def test_unknown_option_is_a_usage_error(tmp_path):
    process = run_cli("--frobnicate", "video.mp4", cwd=tmp_path)

    assert process.returncode == 2
    assert "unrecognized arguments" in output(process)


def test_invalid_thinking_level_choice_is_a_usage_error(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(str(video), "--thinking-level", "extreme", cwd=tmp_path)

    assert process.returncode == 2
    assert "invalid choice" in output(process)


def test_non_integer_chunk_duration_is_a_usage_error(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(str(video), "--chunk-dur", "soon", cwd=tmp_path)

    assert process.returncode == 2
    assert "invalid int value" in output(process)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--chunk-dur", "0"), "chunk-dur must be greater than 0"),
        (("--workers", "0"), "workers must be greater than 0"),
        (("--overlap", "-1"), "overlap must be greater than or equal to 0"),
        (
            ("--chunk-dur", "5", "--overlap", "5"),
            "overlap must be smaller than --chunk-dur",
        ),
        (
            ("--model", "pro", "--thinking-level", "minimal"),
            "only supported by Flash",
        ),
    ],
)
def test_generation_validation_failures_exit_one_with_stable_diagnostics(
    tmp_path, arguments, message
):
    video = make_video(tmp_path)
    process = run_cli(str(video), "--api-key", "key", *arguments, cwd=tmp_path)

    assert process.returncode == 1
    assert message in output(process)
    assert not (tmp_path / "temp_video_chunks").exists()


def test_generation_reports_missing_video_file(tmp_path):
    process = run_cli("missing.mp4", "--api-key", "key", cwd=tmp_path)

    assert process.returncode == 1
    assert "Video file not found" in output(process)
    assert not (tmp_path / "temp_video_chunks").exists()


def test_generation_reports_missing_api_key(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(str(video), cwd=tmp_path)

    assert process.returncode == 1
    assert "API key not configured" in output(process)


def test_api_key_environment_variable_is_accepted(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(str(video), cwd=tmp_path, extra_env={"GEMINI_API_KEY": "key"})

    assert process.returncode == 1
    text = output(process)
    assert "Failed to probe video format" in text
    assert "API key not configured" not in text
    assert not (tmp_path / "temp_video_chunks").exists()


def test_environment_model_allows_minimal_thinking_for_flash(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(
        str(video),
        "--api-key",
        "key",
        "--thinking-level",
        "minimal",
        cwd=tmp_path,
        extra_env={"GEMINI_MODEL": "gemini-flash"},
    )

    assert process.returncode == 1
    text = output(process)
    assert "Failed to probe video format" in text
    assert "only supported by Flash" not in text


def test_cli_model_overrides_environment_model(tmp_path):
    video = make_video(tmp_path)
    process = run_cli(
        str(video),
        "--api-key",
        "key",
        "--model",
        "cli-flash",
        "--thinking-level",
        "minimal",
        cwd=tmp_path,
        extra_env={"GEMINI_MODEL": "environment-pro"},
    )

    assert process.returncode == 1
    text = output(process)
    assert "Failed to probe video format" in text
    assert "only supported by Flash" not in text


def test_generation_rejects_output_resolving_to_the_source(tmp_path):
    video = make_video(tmp_path)
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(video)
    process = run_cli(
        str(video), "--api-key", "key", "--output", str(alias), cwd=tmp_path
    )

    assert process.returncode == 1
    assert "must not resolve to the source video" in output(process)


def test_malformed_context_url_fails_before_any_other_validation(tmp_path):
    process = run_cli("missing.mp4", "--context-url", "not-a-url", cwd=tmp_path)

    assert process.returncode == 1
    text = output(process)
    assert "Invalid --context-url" in text
    assert "Video file not found" not in text
    assert not (tmp_path / "temp_video_chunks").exists()


def test_refine_only_reports_missing_input_vtt(tmp_path):
    process = run_cli("missing.vtt", "--refine-only", "--api-key", "key", cwd=tmp_path)

    assert process.returncode == 1
    assert "Input VTT file not found" in output(process)


def test_refine_only_reports_missing_api_key(tmp_path):
    source = tmp_path / "source.vtt"
    source.write_text("WEBVTT\n", encoding="utf-8")
    process = run_cli(str(source), "--refine-only", cwd=tmp_path)

    assert process.returncode == 1
    assert "API key not configured" in output(process)


def test_refine_only_rejects_malformed_context_url_before_input_checks(tmp_path):
    process = run_cli(
        "missing.vtt",
        "--refine-only",
        "--api-key",
        "key",
        "--context-url",
        "ftp://example.com/x",
        cwd=tmp_path,
    )

    assert process.returncode == 1
    text = output(process)
    assert "Invalid --context-url" in text
    assert "Input VTT file not found" not in text


def test_refine_only_with_unparsable_vtt_fails_before_any_api_request(tmp_path):
    source = tmp_path / "source.vtt"
    source.write_text("this is not a webvtt file\n", encoding="utf-8")
    process = run_cli(str(source), "--refine-only", "--api-key", "key", cwd=tmp_path)

    assert process.returncode == 1
    text = output(process)
    assert "Error: Invalid format" in text
    assert "Researching speaker identities" not in text
    assert "Traceback" not in text
