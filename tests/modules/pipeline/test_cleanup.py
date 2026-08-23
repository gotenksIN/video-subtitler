"""Work-directory cleanup outcomes."""

import pytest

from modules import pipeline


@pytest.mark.parametrize("held", [False, True], ids=["unheld lock", "held lock"])
def test_completed_work_cleanup_removes_artifacts_and_preserves_the_lock(
    tmp_path, held
):
    lock = None
    if held:
        lock = pipeline.acquire_lock(tmp_path)
    else:
        (tmp_path / pipeline.LOCK_NAME).write_text("1", encoding="utf-8")
    try:
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "chunk_000.mp4").write_bytes(b"chunk")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "state.json").write_text("{}", encoding="utf-8")

        pipeline.clean_completed_work(tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == [pipeline.LOCK_NAME]
    finally:
        if lock is not None:
            pipeline.release_lock(lock)
