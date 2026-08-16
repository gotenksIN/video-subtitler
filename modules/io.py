"""Atomic artifact publication and manifest file I/O."""

import json
import os
import tempfile
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atomic_save_vtt(vtt, path):
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.vtt", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        vtt.save(str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_manifest(chunk_dir):
    manifest_path = os.path.join(chunk_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
