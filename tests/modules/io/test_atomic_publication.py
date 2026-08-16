"""Atomic publication of JSON artifacts and VTT output."""

import json

import pytest
import webvtt

from modules import io


def test_json_publication_overwrites_the_target_and_cleans_up(tmp_path):
    target = tmp_path / "captions.json"
    target.write_text("old", encoding="utf-8")

    io.atomic_write_json(target, {"text": "plain ascii"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"text": "plain ascii"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["captions.json"]


def test_json_publication_creates_a_new_target(tmp_path):
    target = tmp_path / "captions.json"

    io.atomic_write_json(target, [1, 2])

    assert json.loads(target.read_text(encoding="utf-8")) == [1, 2]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["captions.json"]


def test_vtt_publication_writes_a_readable_webvtt_file(tmp_path):
    output = tmp_path / "output.vtt"
    value = webvtt.WebVTT()
    value.captions.extend(
        [
            webvtt.Caption("00:00:00.000", "00:00:01.000", "One"),
            webvtt.Caption("00:00:01.000", "00:00:02.000", "Two"),
        ]
    )

    io.atomic_save_vtt(value, output)

    result = webvtt.read(output)
    assert [(c.start, c.end, c.text) for c in result] == [
        ("00:00:00.000", "00:00:01.000", "One"),
        ("00:00:01.000", "00:00:02.000", "Two"),
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output.vtt"]


def test_failed_vtt_save_preserves_the_target_and_cleans_up(tmp_path):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")

    class FailingVtt:
        def save(self, _path):
            raise OSError("save failed")

    with pytest.raises(OSError):
        io.atomic_save_vtt(FailingVtt(), output)

    assert output.read_text(encoding="utf-8") == "previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output.vtt"]


def test_failed_vtt_replace_preserves_the_target_and_cleans_up(tmp_path, monkeypatch):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    value = webvtt.WebVTT()
    value.captions.append(webvtt.Caption("00:00:00.000", "00:00:01.000", "New"))

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", fail_replace)

    with pytest.raises(OSError):
        io.atomic_save_vtt(value, output)

    assert output.read_text(encoding="utf-8") == "previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output.vtt"]
