"""Process lock behavior for work-directory ownership."""

import os

import pytest

from modules import gemini, media, pipeline
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


def test_second_owner_is_blocked_with_the_owner_pid(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=f"PID {os.getpid()}"):
            pipeline.acquire_lock(tmp_path)
    finally:
        pipeline.release_lock(first)


def test_lock_file_records_the_owner_pid(tmp_path):
    lock = pipeline.acquire_lock(tmp_path)
    try:
        assert (tmp_path / pipeline.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        pipeline.release_lock(lock)


def test_release_allows_a_new_owner(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(first)

    second = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(second)

    assert (tmp_path / pipeline.LOCK_NAME).exists()


def test_stale_pid_text_does_not_block_a_new_owner(tmp_path):
    (tmp_path / pipeline.LOCK_NAME).write_text("999999", encoding="utf-8")

    lock = pipeline.acquire_lock(tmp_path)
    try:
        assert (tmp_path / pipeline.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        pipeline.release_lock(lock)


def test_generation_holds_the_work_lock_during_api_work(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(
        media, "probe_video_format", lambda _path: (".mp4", "video/mp4", "h264")
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    monkeypatch.setattr(media.subprocess, "run", FakeMediaTools().run)
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
