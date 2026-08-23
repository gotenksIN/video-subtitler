"""Process lock behavior for work-directory ownership."""

import pytest

from modules import core, gemini, media, pipeline
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


def test_second_owner_is_blocked_while_the_lock_is_held(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already using"):
            pipeline.acquire_lock(tmp_path)
    finally:
        pipeline.release_lock(first)


def test_release_allows_a_new_owner(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(first)

    second = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(second)

    assert (tmp_path / pipeline.LOCK_NAME).exists()


def test_generation_holds_the_work_lock_during_api_work(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(
        media, "probe_video_format", lambda _path: (".mp4", "video/mp4", "h264")
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    monkeypatch.setattr(media.subprocess, "run", FakeMediaTools().run)
    monkeypatch.setattr(
        gemini,
        "run_preflight_context",
        lambda *_args, **_kwargs: core.PreflightContext(),
    )
    config = pipeline.GenerationConfig(
        video_path=video,
        output_path=tmp_path / "output.vtt",
        model="model",
        api_key="key",
        audio_refine=False,
        refine_text=False,
    )
    observed = {}

    def process(_key, _base, _chunk, chunk_dir, *_args):
        try:
            pipeline.acquire_lock(chunk_dir)
        except RuntimeError as error:
            observed["error"] = str(error)
            write_chunk_subtitles(
                chunk_dir,
                0,
                [{"id": 0, "start": "0.5", "end": "1.5", "text": "Cue 0"}],
            )
            write_chunk_subtitles(
                chunk_dir,
                1,
                [{"id": 0, "start": "0.5", "end": "1.5", "text": "Cue 1"}],
            )
            return True
        observed["error"] = "the work lock was not held"
        return False

    monkeypatch.setattr(gemini, "process_chunk", process)

    pipeline.run_generation(config)

    assert "already using" in observed["error"]
