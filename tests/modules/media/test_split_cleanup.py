"""Incomplete split artifact cleanup outcomes."""

from modules import io, media, pipeline


def test_incomplete_split_cleanup_removes_only_split_artifacts(tmp_path):
    keep = {
        io.MANIFEST_NAME: "{}",
        pipeline.LOCK_NAME: "1",
        media.SPLIT_COMPLETE_MARKER: "ok\n",
        "notes.txt": "user file",
    }
    remove = {
        "chunk_000.mp4": b"chunk",
        "chunk_001.webm": b"chunk",
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

    media.clean_incomplete_split(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(keep)
