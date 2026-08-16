"""Concurrent chunk scheduling with real threads and scenario fakes."""

import threading
from pathlib import Path

import gemini_subs

CHUNKS = [
    {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2},
    {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4},
]


def run_pipeline(tmp_path, monkeypatch, process, attach=None, overlap=0, workers=2):
    if attach is None:
        attach = gemini_subs.attach_overlap_clip
    monkeypatch.setattr(gemini_subs, "attach_overlap_clip", attach)
    monkeypatch.setattr(gemini_subs, "process_chunk", process)
    return gemini_subs.process_chunks(
        "key",
        None,
        "source.mp4",
        str(tmp_path),
        CHUNKS,
        overlap,
        ".mp4",
        2,
        workers,
        "model",
        "video/mp4",
        "high",
    )


def test_chunks_are_processed_concurrently_by_real_worker_threads(
    tmp_path, monkeypatch
):
    started = []
    barrier = threading.Barrier(2, timeout=10)

    def process(_key, _base, chunk, chunk_dir, *_args):
        started.append(threading.get_ident())
        barrier.wait()
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return True

    failed = run_pipeline(tmp_path, monkeypatch, process, overlap=0)

    assert failed == []
    assert (tmp_path / "done_0").read_text(encoding="utf-8") == "done"
    assert (tmp_path / "done_1").read_text(encoding="utf-8") == "done"
    assert len(set(started)) >= 2


def test_api_failures_are_reported_with_the_stream_copy_chunk_name(
    tmp_path, monkeypatch
):
    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return chunk["idx"] == 0

    failed = run_pipeline(tmp_path, monkeypatch, process, overlap=0)

    assert failed == ["chunk_001.mp4"]
    assert (tmp_path / "done_0").exists()
    assert (tmp_path / "done_1").exists()


def test_no_overlap_forwards_stream_copy_chunks_to_the_api(tmp_path, monkeypatch):
    received = []

    def process(_key, _base, chunk, *_args):
        received.append(chunk["clip_name"])
        return True

    failed = run_pipeline(tmp_path, monkeypatch, process, overlap=0)

    assert failed == []
    assert sorted(received) == ["chunk_000.mp4", "chunk_001.mp4"]


def test_overlap_processing_reports_clip_failures_and_finishes_other_chunks(
    tmp_path, monkeypatch
):
    def attach(_video, _directory, chunk, _overlap, _ext, *_args):
        if chunk["idx"] == 1:
            raise RuntimeError("encode failed")
        return {**chunk, "clip_name": f"context_chunk_{chunk['idx']:03d}.mp4"}

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return True

    failed = run_pipeline(tmp_path, monkeypatch, process, attach=attach, overlap=1)

    assert failed == ["context_chunk_001.mp4"]
    assert (tmp_path / "done_0").read_text(encoding="utf-8") == "done"
    assert not (tmp_path / "done_1").exists()


def test_overlap_processing_reports_api_failures_with_clip_names(tmp_path, monkeypatch):
    def attach(_video, _directory, chunk, _overlap, _ext, *_args):
        return {**chunk, "clip_name": f"context_chunk_{chunk['idx']:03d}.mp4"}

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return chunk["idx"] != 0

    failed = run_pipeline(tmp_path, monkeypatch, process, attach=attach, overlap=1)

    assert failed == ["context_chunk_000.mp4"]
    assert (tmp_path / "done_1").exists()


def test_overlap_forwards_window_metadata_to_the_api(tmp_path, monkeypatch):
    received = []

    def attach(_video, _directory, chunk, _overlap, _ext, *_args):
        return {**chunk, "clip_name": f"context_chunk_{chunk['idx']:03d}.mp4"}

    def process(_key, _base, chunk, *_args):
        received.append(
            (
                chunk["clip_name"],
                chunk["clip_duration"],
                chunk["owner_start_rel"],
                chunk["owner_end_rel"],
            )
        )
        return True

    failed = run_pipeline(tmp_path, monkeypatch, process, attach=attach, overlap=1)

    assert failed == []
    assert sorted(received) == [
        ("context_chunk_000.mp4", 3.0, 0, 2),
        ("context_chunk_001.mp4", 3.0, 1, 3),
    ]
