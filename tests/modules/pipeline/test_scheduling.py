"""Concurrent chunk scheduling with real threads and scenario fakes."""

import threading
from pathlib import Path

from modules import gemini, pipeline

CHUNKS = [
    {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2, "duration": 2},
    {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4, "duration": 2},
]


def run_scheduler(tmp_path, monkeypatch, process, workers=2):
    monkeypatch.setattr(gemini, "process_chunk", process)
    return pipeline.process_chunks(
        "key",
        None,
        str(tmp_path),
        CHUNKS,
        workers,
        "model",
        "video/mp4",
        "high",
    )


def test_chunks_are_processed_concurrently_by_real_worker_threads(
    tmp_path, monkeypatch
):
    barrier = threading.Barrier(2, timeout=10)

    def process(_key, _base, chunk, chunk_dir, *_args):
        barrier.wait()
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return True

    failed = run_scheduler(tmp_path, monkeypatch, process)

    assert failed == []
    assert (tmp_path / "done_0").read_text(encoding="utf-8") == "done"
    assert (tmp_path / "done_1").read_text(encoding="utf-8") == "done"


def test_api_failures_are_reported_with_the_stream_copy_chunk_name(
    tmp_path, monkeypatch
):
    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"done_{chunk['idx']}").write_text("done", encoding="utf-8")
        return chunk["idx"] == 0

    failed = run_scheduler(tmp_path, monkeypatch, process)

    assert failed == ["chunk_001.mp4"]
    assert (tmp_path / "done_0").exists()
    assert (tmp_path / "done_1").exists()


def test_stream_copy_chunks_are_forwarded_with_segment_metadata(tmp_path, monkeypatch):
    received = []

    def process(_key, _base, chunk, *_args):
        received.append(
            (chunk["name"], chunk["start"], chunk["end"], chunk["duration"])
        )
        return True

    failed = run_scheduler(tmp_path, monkeypatch, process)

    assert failed == []
    assert sorted(received) == [
        ("chunk_000.mp4", 0, 2, 2),
        ("chunk_001.mp4", 2, 4, 2),
    ]


def test_source_title_and_candidate_names_reach_every_chunk_worker(
    tmp_path, monkeypatch
):
    received = []

    def process(_key, _base, _chunk, _chunk_dir, _model, _mime, _level, title, names):
        received.append((title, names))
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)

    failed = pipeline.process_chunks(
        "key",
        None,
        str(tmp_path),
        CHUNKS,
        2,
        "model",
        "video/mp4",
        "high",
        "Show Title",
        ["Jane Doe"],
    )

    assert failed == []
    assert received == [("Show Title", ["Jane Doe"])] * len(CHUNKS)
