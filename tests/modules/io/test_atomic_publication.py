"""Atomic output publication behavior."""

import pytest

from modules import io


def test_final_publication_refuses_to_replace_existing_content(tmp_path):
    destination = tmp_path / "song.lrc"
    destination.write_text("reference\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        io.atomic_publish_text(destination, "generated\n")

    assert destination.read_text(encoding="utf-8") == "reference\n"
