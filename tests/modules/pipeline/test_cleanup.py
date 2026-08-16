"""Work-directory cleanup outcomes."""

import gemini_subs


def test_completed_work_cleanup_removes_everything_but_the_lock(tmp_path):
    (tmp_path / gemini_subs.LOCK_NAME).write_text("1", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "chunk_000.mp4").write_bytes(b"chunk")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "state.json").write_text("{}", encoding="utf-8")

    gemini_subs.clean_completed_work(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [gemini_subs.LOCK_NAME]


def test_completed_work_cleanup_preserves_the_lock_while_held(tmp_path):
    lock = gemini_subs.acquire_lock(tmp_path)
    try:
        (tmp_path / "artifact.json").write_text("state", encoding="utf-8")

        gemini_subs.clean_completed_work(tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == [
            gemini_subs.LOCK_NAME
        ]
    finally:
        gemini_subs.release_lock(lock)


def test_incomplete_split_cleanup_removes_only_split_artifacts(tmp_path):
    keep = {
        gemini_subs.MANIFEST_NAME: "{}",
        gemini_subs.LOCK_NAME: "1",
        gemini_subs.SPLIT_COMPLETE_MARKER: "ok\n",
        "notes.txt": "user file",
    }
    remove = {
        "chunk_000.mp4": b"chunk",
        "chunk_001.webm": b"chunk",
        "context_chunk_000.mp4": b"clip",
        "context_chunk_000.mp4.tmp": b"tmp",
        "subtitle_chunk_000.json": "[]",
        "subtitle_chunk_000.json.tmp": "[]",
        "segments.csv": "chunk_000.mp4,0,1\n",
    }
    for name, content in {**keep, **remove}.items():
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    gemini_subs.clean_incomplete_split(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(keep)
