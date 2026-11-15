"""Media and publication contracts through real process and file seams."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import webvtt

from modules import io, media


@pytest.mark.parametrize("codec", ["vp9", "h264", "hevc"])
def test_supported_media_splits_into_decodable_stream_copy_chunks(
    media_files, tmp_path, codec
):
    source = media_files[codec]
    inspection = media.inspect_media(source)
    ext, mime, actual_codec = inspection.video_format
    assert actual_codec == codec
    assert mime == ("video/webm" if codec == "vp9" else "video/mp4")
    assert inspection.duration == pytest.approx(2, abs=0.1)
    work = tmp_path / "work"
    media.split_video(str(source), work, 1, {"chunk_ext": ext})
    chunks = media.list_chunks(work)
    assert len(chunks) == 2
    assert (
        0 == chunks[0]["start"] < chunks[1]["start"] == pytest.approx(chunks[0]["end"])
    )
    for chunk in chunks:
        path = work / chunk["name"]
        assert media.inspect_media(path).video_format == inspection.video_format
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
            check=True,
        )
    before = [(work / chunk["name"]).stat().st_mtime_ns for chunk in chunks]
    media.split_video(str(source), work, 1, {"chunk_ext": ext})
    assert [(work / chunk["name"]).stat().st_mtime_ns for chunk in chunks] == before


def test_unsupported_video_codec_fails(media_files):
    with pytest.raises(RuntimeError, match="not supported"):
        _ = media.inspect_media(media_files["unsupported"]).video_format


def test_embedded_subtitles_do_not_enter_chunks(media_files, tmp_path):
    media.split_video(str(media_files["subtitled"]), tmp_path, 1, {"chunk_ext": ".mp4"})
    for chunk in media.list_chunks(tmp_path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(tmp_path / chunk["name"]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        kinds = {
            stream["codec_type"] for stream in json.loads(result.stdout)["streams"]
        }
        assert kinds == {"video", "audio"}


def test_complete_opus_audio_is_reused_and_corruption_is_regenerated(
    media_files, tmp_path
):
    source = media_files["h264"]
    path, duration, source_duration, reused = media.extract_complete_audio(
        source, tmp_path
    )
    assert not reused
    assert duration == pytest.approx(source_duration, abs=0.1)
    inspection = media.inspect_media(path)
    assert inspection.audio_streams == [
        {
            "codec_name": "opus",
            "codec_type": "audio",
            "sample_rate": "48000",
            "channels": 1,
        },
    ]
    before = Path(path).stat().st_mtime_ns
    assert media.extract_complete_audio(source, tmp_path)[-1]
    assert Path(path).stat().st_mtime_ns == before
    Path(path).write_bytes(b"corrupt")
    assert not media.extract_complete_audio(source, tmp_path)[-1]
    assert media.inspect_media(path).duration > 0
    Path(path).unlink()
    with pytest.raises(RuntimeError, match="No such file or directory") as failure:
        media.extract_complete_audio(
            tmp_path / "missing.mp4",
            tmp_path,
            source_inspection=media.inspect_media(source),
        )
    assert "missing.mp4" in str(failure.value)
    assert not Path(path).exists()
    assert not Path(f"{path}.tmp").exists()


@pytest.mark.parametrize(
    "damage", ["empty-chunk", "malformed-index", "truncated-index"]
)
def test_split_cache_damage_regenerates_real_media(media_files, tmp_path, damage):
    source = str(media_files["h264"])
    manifest = {"chunk_ext": ".mp4"}
    media.split_video(source, tmp_path, 1, manifest)
    chunks = media.list_chunks(tmp_path)
    if damage == "empty-chunk":
        (tmp_path / chunks[0]["name"]).write_bytes(b"")
    else:
        (tmp_path / "segments.csv").write_text(
            "broken\n" if damage == "malformed-index" else f"{chunks[0]['name']},0,1\n",
            encoding="ascii",
        )
    media.split_video(source, tmp_path, 1, manifest)
    rebuilt = media.list_chunks(tmp_path)
    assert len(rebuilt) == 2
    for chunk in rebuilt:
        assert media.inspect_media(tmp_path / chunk["name"]).duration > 0


def test_failed_split_removes_stale_chunks_and_preserves_unrelated_work(
    media_files, tmp_path
):
    media.split_video(str(media_files["h264"]), tmp_path, 1, {"chunk_ext": ".mp4"})
    (tmp_path / "segments.csv").write_text("bad\n", encoding="ascii")
    note = tmp_path / "notes.txt"
    note.write_text("keep", encoding="ascii")
    stale_json = tmp_path / ".subtitle_chunk_000.json.interrupted.tmp"
    stale_json.write_text("partial", encoding="ascii")
    with pytest.raises(subprocess.CalledProcessError):
        media.split_video(
            str(tmp_path / "missing.mp4"), tmp_path, 1, {"chunk_ext": ".mp4"}
        )
    assert not list(tmp_path.glob("chunk_*"))
    assert media.list_chunks(tmp_path) == []
    assert not stale_json.exists()
    assert note.read_text() == "keep"


def test_atomic_json_and_vtt_preserve_targets_on_replace_failure(tmp_path, monkeypatch):
    def fail_replace(*_args):
        raise OSError("publication denied")

    value = webvtt.WebVTT()
    value.captions.append(webvtt.Caption("00:00:00.000", "00:00:01.000", "Hello"))
    for name, publish in (
        ("output.json", lambda path: io.atomic_write_json(path, {"value": 1})),
        ("output.vtt", lambda path: io.atomic_save_vtt(value, path)),
    ):
        path = tmp_path / name
        publish(path)
        previous = path.read_bytes()
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_replace)
            with pytest.raises(OSError, match="publication denied"):
                publish(path)
        assert path.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "output.json",
        "output.vtt",
    ]


def test_plain_vtt_round_trip_and_tagged_input_rejection(tmp_path):
    path = tmp_path / "output.vtt"
    value = webvtt.WebVTT()
    value.captions.append(
        webvtt.Caption(
            "00:00:00.000",
            "00:00:01.000",
            "Host: A & B < C > D",
        )
    )
    io.atomic_save_vtt(value, path)
    assert webvtt.read(path, strict=True)[0].raw_text == "Host: A &amp; B &lt; C &gt; D"
    assert io.read_plain_vtt(path)[0].raw_text == "Host: A & B < C > D"
    for text in ("<v Host>Hello</v>", "<b>Hello</b>"):
        path.write_text(
            f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{text}\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="Tagged WebVTT is unsupported"):
            io.read_plain_vtt(path)
