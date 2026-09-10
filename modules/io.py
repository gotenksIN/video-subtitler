"""Atomic diagnostic and final artifact publication."""

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace a text file atomically through a temporary sibling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_publish_text(path: str | Path, text: str) -> None:
    """Publish text atomically without replacing an existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    except FileExistsError:
        raise FileExistsError(f"Output already exists: {path}") from None
    finally:
        tmp.unlink(missing_ok=True)
