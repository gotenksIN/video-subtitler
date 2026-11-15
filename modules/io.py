"""Atomic artifact publication, manifest file I/O, and work-directory logging."""

import contextlib
import copy
import html
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import webvtt

MANIFEST_NAME = "manifest.json"
LOG_NAME = "pipeline.log"
_log_lock = threading.Lock()
_log_handle = None


@contextlib.contextmanager
def work_log(chunk_dir):
    """Append run diagnostics until all work-directory workers settle."""
    global _log_handle
    with open(Path(chunk_dir) / LOG_NAME, "a", encoding="utf-8") as handle:
        with _log_lock:
            previous_handle = _log_handle
            _log_handle = handle
        try:
            yield
        finally:
            with _log_lock:
                try:
                    handle.flush()
                finally:
                    _log_handle = previous_handle


def log_file(message):
    """Write timestamped lines only when a work log is active."""
    with _log_lock:
        if _log_handle is not None:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            for line in str(message).splitlines():
                _log_handle.write(f"{timestamp} {line}\n")
            _log_handle.flush()


def log(message):
    """Keep console diagnostics and append them to the active work log."""
    print(message)
    log_file(message)


def atomic_write_json(path, data):
    """Write JSON data to a file atomically using a temporary sibling."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_plain_vtt(path):
    """Read strict WebVTT as plain text; reject unsupported cue markup."""
    vtt = webvtt.read(str(path), strict=True)
    for cue in vtt:
        if "<" in cue.raw_text or ">" in cue.raw_text:
            raise ValueError(
                "Tagged WebVTT is unsupported; escape literal angle brackets"
            )
        cue.text = html.unescape(cue.raw_text)
    return vtt


def atomic_save_vtt(vtt, path):
    """Escape plain-text cues and save WebVTT to a destination atomically."""
    serialized = copy.deepcopy(vtt)
    for cue in serialized.captions:
        cue.text = html.escape(cue.raw_text, quote=False)
    expected = [(cue.start, cue.end, cue.raw_text) for cue in serialized.captions]
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.vtt", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        serialized.save(str(tmp_path))
        persisted = [
            (cue.start, cue.end, cue.raw_text)
            for cue in webvtt.read(str(tmp_path), strict=True)
        ]
        if persisted != expected:
            raise ValueError("Serialized VTT does not match the intended captions")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def file_fingerprint(path):
    """Return a file fingerprint dict with path, size, and modification time."""
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
