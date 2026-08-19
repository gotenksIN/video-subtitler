"""Complete generation lifecycle and recovery outcomes."""

import pytest
import webvtt

from modules import gemini, media, pipeline
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


def prepare_generation(
    tmp_path, monkeypatch, refine_text=False, overlap=0.0, output_path=None
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(
        media,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    config = pipeline.GenerationConfig(
        video_path=video,
        output_path=output_path or tmp_path / "output.vtt",
        model="model",
        api_key="key",
        chunk_dur=60,
        overlap=overlap,
        workers=2,
        thinking_level="high",
        refine_text=refine_text,
    )
    return config, tools


def write_two_chunk_subtitles(monkeypatch):
    def process(_key, _base, chunk, chunk_dir, *_args):
        index = chunk["idx"]
        write_chunk_subtitles(
            chunk_dir,
            index,
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {index}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)


def active_work_directory(tmp_path):
    return next(iter((tmp_path / "work_root").iterdir()))


def test_successful_generation_publishes_and_cleans_work(tmp_path, monkeypatch):
    config, _tools = prepare_generation(tmp_path, monkeypatch)
    write_two_chunk_subtitles(monkeypatch)

    pipeline.run_generation(config)

    result = webvtt.read(config.output_path)
    assert [(c.start, c.end, c.text) for c in result] == [
        ("00:00:00.500", "00:00:01.500", "Cue 0"),
        ("00:00:02.500", "00:00:03.500", "Cue 1"),
    ]
    work = active_work_directory(tmp_path)
    assert sorted(path.name for path in work.iterdir()) == [pipeline.LOCK_NAME]
    lock = pipeline.acquire_lock(work)
    pipeline.release_lock(lock)


def test_failed_chunk_processing_keeps_resume_state_and_releases_lock(
    tmp_path, monkeypatch
):
    config, _tools = prepare_generation(tmp_path, monkeypatch)

    def process(_key, _base, chunk, chunk_dir, *_args):
        if chunk["idx"] == 0:
            write_chunk_subtitles(
                chunk_dir,
                0,
                [{"id": 0, "start": "0.5", "end": "1.5", "text": "Cue 0"}],
            )
            return True
        return False

    monkeypatch.setattr(gemini, "process_chunk", process)

    with pytest.raises(RuntimeError, match="Failed to process 1 chunk"):
        pipeline.run_generation(config)

    work = active_work_directory(tmp_path)
    names = sorted(path.name for path in work.iterdir())
    assert "subtitle_chunk_000.json" in names
    assert media.SPLIT_COMPLETE_MARKER in names
    assert not config.output_path.exists()
    lock = pipeline.acquire_lock(work)
    pipeline.release_lock(lock)


def test_retry_resumes_failed_work_without_resplitting_and_publishes(
    tmp_path, monkeypatch
):
    config, tools = prepare_generation(tmp_path, monkeypatch)
    failed_once = False

    def process(_key, _base, chunk, chunk_dir, *_args):
        nonlocal failed_once
        index = chunk["idx"]
        if index == 1 and not failed_once:
            failed_once = True
            return False
        write_chunk_subtitles(
            chunk_dir,
            index,
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {index}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)

    with pytest.raises(RuntimeError, match="Failed to process 1 chunk"):
        pipeline.run_generation(config)
    pipeline.run_generation(config)

    assert len(tools.split_calls()) == 1
    result = webvtt.read(config.output_path)
    assert [c.text for c in result] == ["Cue 0", "Cue 1"]
    work = active_work_directory(tmp_path)
    assert sorted(path.name for path in work.iterdir()) == [pipeline.LOCK_NAME]


def test_refinement_failure_preserves_output_and_resume_state(tmp_path, monkeypatch):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    config, _tools = prepare_generation(
        tmp_path, monkeypatch, refine_text=True, output_path=output
    )
    write_two_chunk_subtitles(monkeypatch)

    def refine(_input_path, *_args, **_kwargs):
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(gemini, "global_refine_subtitles", refine)

    with pytest.raises(RuntimeError, match="refinement failed"):
        pipeline.run_generation(config)

    assert output.read_text(encoding="utf-8") == "previous"
    work = active_work_directory(tmp_path)
    names = sorted(path.name for path in work.iterdir())
    assert media.SPLIT_COMPLETE_MARKER in names
    lock = pipeline.acquire_lock(work)
    pipeline.release_lock(lock)


def test_successful_refined_generation_publishes_output(tmp_path, monkeypatch):
    config, _tools = prepare_generation(tmp_path, monkeypatch, refine_text=True)
    write_two_chunk_subtitles(monkeypatch)

    def refine(input_path, output_path, *_args, **_kwargs):
        value = webvtt.read(input_path)
        value.captions[0].text = "Refined cue"
        value.save(output_path)

    monkeypatch.setattr(gemini, "global_refine_subtitles", refine)

    pipeline.run_generation(config)

    result = webvtt.read(config.output_path)
    assert [c.text for c in result] == ["Refined cue", "Cue 1"]
    work = active_work_directory(tmp_path)
    assert sorted(path.name for path in work.iterdir()) == [pipeline.LOCK_NAME]
