"""Work-directory cleanup outcomes."""

from modules import pipeline


def test_completed_work_cleanup_removes_everything_but_the_lock(tmp_path):
    (tmp_path / pipeline.LOCK_NAME).write_text("1", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "chunk_000.mp4").write_bytes(b"chunk")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "state.json").write_text("{}", encoding="utf-8")

    pipeline.clean_completed_work(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [pipeline.LOCK_NAME]


def test_completed_work_cleanup_preserves_the_lock_while_held(tmp_path):
    lock = pipeline.acquire_lock(tmp_path)
    try:
        (tmp_path / "artifact.json").write_text("state", encoding="utf-8")

        pipeline.clean_completed_work(tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == [pipeline.LOCK_NAME]
    finally:
        pipeline.release_lock(lock)
