"""Observable cache and resume outcomes for persistent work state."""

from dataclasses import replace
from pathlib import Path

import pytest
import webvtt

from modules import core, gemini, media, pipeline
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


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
        "workers": 7,
        "thinking_level": "high",
    }
    values.update(overrides)
    return pipeline.GenerationConfig(**values)


def leave_resumable_failure(config, monkeypatch):
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    monkeypatch.setattr(gemini, "process_chunk", lambda *_args: False)
    monkeypatch.setattr(
        gemini,
        "run_preflight_context",
        lambda *_args, **_kwargs: core.PreflightContext(),
    )

    with pytest.raises(RuntimeError, match="Failed to process"):
        pipeline.run_generation(config)

    return {path.name for path in (Path(pipeline.CHUNK_ROOT)).iterdir()}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("chunk_dur", 30),
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
        ("audio_refine_model", "other-audio-refiner"),
        ("audio_refine", False),
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


def test_retry_after_audio_stage_failure_reuses_split_chunks_and_audio(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path, monkeypatch, audio_refine=True, refine_text=False)
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    monkeypatch.setattr(
        gemini,
        "run_preflight_context",
        lambda *_args, **_kwargs: core.PreflightContext(),
    )

    def process(_key, _base, chunk, chunk_dir, *_args):
        write_chunk_subtitles(
            chunk_dir,
            chunk["idx"],
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {chunk['idx']}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)

    def failing_refine(*_args, **_kwargs):
        raise RuntimeError("audio refinement failed")

    monkeypatch.setattr(gemini, "boundary_audio_refine_subtitles", failing_refine)
    with pytest.raises(RuntimeError, match="audio refinement failed"):
        pipeline.run_generation(config)

    def audio_refine(**kwargs):
        value = webvtt.read(kwargs["stitched_vtt"])
        for caption in value.captions:
            caption.text = f"Audio: {caption.text}"
        value.save(str(kwargs["output_vtt"]))

    monkeypatch.setattr(gemini, "boundary_audio_refine_subtitles", audio_refine)
    pipeline.run_generation(config)

    assert len(tools.split_calls()) == 1
    assert len(tools.audio_extraction_calls()) == 1
    result = webvtt.read(config.output_path)
    assert [caption.text for caption in result] == ["Audio: Cue 0", "Audio: Cue 1"]


def test_retry_reuses_cached_preflight_context_without_new_research(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path, monkeypatch, audio_refine=False, refine_text=False)
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    research_runs = []

    def preflight(*_args, **_kwargs):
        research_runs.append(1)
        return core.PreflightContext(identity_context="Jane Doe: Host.")

    monkeypatch.setattr(gemini, "run_preflight_context", preflight)
    monkeypatch.setattr(gemini, "process_chunk", lambda *_args: False)

    with pytest.raises(RuntimeError, match="Failed to process"):
        pipeline.run_generation(config)

    work_dir = next(iter(Path(pipeline.CHUNK_ROOT).iterdir()))
    assert (work_dir / gemini.PREFLIGHT_CONTEXT_FILENAME).exists()

    def process(_key, _base, chunk, chunk_dir, *_args):
        write_chunk_subtitles(
            chunk_dir,
            chunk["idx"],
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {chunk['idx']}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)
    pipeline.run_generation(config)

    assert research_runs == [1]
    assert [caption.text for caption in webvtt.read(config.output_path)] == [
        "Cue 0",
        "Cue 1",
    ]
