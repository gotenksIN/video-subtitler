"""Observable cache and resume outcomes for persistent work state."""

from dataclasses import replace
from pathlib import Path

import pytest

from modules import gemini, media, pipeline
from tests.support.workdir import FakeMediaTools


def make_config(tmp_path, monkeypatch, **overrides):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        media,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    values = {
        "video_path": video,
        "output_path": tmp_path / "output.vtt",
        "model": "model",
        "api_key": "key",
        "chunk_dur": 60,
        "overlap": 0,
        "workers": 7,
        "thinking_level": "high",
    }
    values.update(overrides)
    return pipeline.GenerationConfig(**values)


def leave_resumable_failure(config, monkeypatch):
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    monkeypatch.setattr(gemini, "process_chunk", lambda *_args: False)

    with pytest.raises(RuntimeError, match="Failed to process"):
        pipeline.run_generation(config)

    return {path.name for path in (Path(pipeline.CHUNK_ROOT)).iterdir()}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("chunk_dur", 30),
        ("overlap", 1),
        ("thinking_level", "medium"),
    ],
)
def test_chunk_generation_changes_create_separate_resume_state(
    tmp_path, monkeypatch, field, value
):
    config = make_config(tmp_path, monkeypatch)
    first = leave_resumable_failure(config, monkeypatch)

    second = leave_resumable_failure(replace(config, **{field: value}), monkeypatch)

    assert len(first) == 1
    assert len(second) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_path", Path("other.vtt")),
        ("workers", 2),
        ("refine_model", "other-refiner"),
        ("refine_text", False),
        ("api_key", "other-key"),
        ("base_url", "https://proxy.example"),
        ("context_urls", ("https://example.com/notes",)),
    ],
)
def test_non_chunk_options_reuse_existing_resume_state(
    tmp_path, monkeypatch, field, value
):
    config = make_config(tmp_path, monkeypatch)
    first = leave_resumable_failure(config, monkeypatch)
    if field == "output_path":
        value = tmp_path / value

    second = leave_resumable_failure(replace(config, **{field: value}), monkeypatch)

    assert len(first) == 1
    assert second == first


def test_changing_the_source_file_creates_separate_resume_state(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch)
    first = leave_resumable_failure(config, monkeypatch)
    config.video_path.write_bytes(b"changed content")

    second = leave_resumable_failure(config, monkeypatch)

    assert len(first) == 1
    assert len(second) == 2
