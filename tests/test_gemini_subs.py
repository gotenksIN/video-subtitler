import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import webvtt

import gemini_subs


def caption(caption_id, start, end, text="Text"):
    return gemini_subs.Caption(id=caption_id, start=start, end=end, text=text)


def manifest(overlap=0, codec="h264"):
    ext = ".webm" if codec == "vp9" else ".mp4"
    return {
        "mode": "generate",
        "overlap": overlap,
        "chunk_ext": ext,
        "process_ext": ext,
        "process_mime": "video/webm" if ext == ".webm" else "video/mp4",
        "video_codec": codec,
    }


def write_chunk_layout(directory, rows, overlap=0):
    Path(directory, gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(manifest(overlap)), encoding="utf-8"
    )
    Path(directory, "segments.csv").write_text(
        "".join(f"{name},{start},{end}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def test_atomic_save_vtt_uses_unique_temporary_paths_in_destination(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "output.vtt"
    sources = []
    real_replace = os.replace

    def record_replace(source, destination):
        sources.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr(gemini_subs.os, "replace", record_replace)
    vtt = webvtt.WebVTT()

    gemini_subs.atomic_save_vtt(vtt, output_path)
    gemini_subs.atomic_save_vtt(vtt, output_path)

    assert len(set(sources)) == 2
    assert all(source.parent == output_path.parent for source in sources)
    assert all(source.name.endswith(".tmp.vtt") for source in sources)


@pytest.mark.parametrize("failure", ["save", "replace"])
def test_atomic_save_vtt_removes_temporary_file_on_failure(
    tmp_path, monkeypatch, failure
):
    output_path = tmp_path / "output.vtt"
    vtt = MagicMock()
    if failure == "save":
        vtt.save.side_effect = OSError("save failed")
    else:
        monkeypatch.setattr(
            gemini_subs.os,
            "replace",
            MagicMock(side_effect=OSError("replace failed")),
        )

    with pytest.raises(OSError, match=f"{failure} failed"):
        gemini_subs.atomic_save_vtt(vtt, output_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("02:03.25", 123.25), ("01:02:03,004", 3723.004)],
)
def test_parse_time_accepts_supported_shapes(value, expected):
    assert gemini_subs.parse_time(value) == expected


def test_parse_time_rejects_too_many_components():
    with pytest.raises(ValueError, match="Invalid timestamp"):
        gemini_subs.parse_time("1:2:3:4")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00.000"),
        (3661.2346, "01:01:01.235"),
        (59.9996, "00:01:00.000"),
    ],
)
def test_format_time_rounds_and_carries(seconds, expected):
    assert gemini_subs.format_time(seconds) == expected


def test_format_time_rejects_negative_values():
    with pytest.raises(ValueError, match="Negative timestamp"):
        gemini_subs.format_time(-0.001)


def test_processing_windows_clip_at_video_edges():
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0.0, "end": 10.0},
        {"idx": 1, "name": "chunk_001.mp4", "start": 10.0, "end": 18.0},
    ]

    windows = gemini_subs.get_processing_windows(chunks, 2.0)

    assert (windows[0]["clip_start"], windows[0]["clip_end"]) == (0.0, 12.0)
    assert (windows[1]["clip_start"], windows[1]["clip_end"]) == (8.0, 18.0)
    assert (windows[1]["owner_start_rel"], windows[1]["owner_end_rel"]) == (
        2.0,
        10.0,
    )


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [(None, 1), (1, 1), (8, 1), (16, 2), (31, 3)],
)
def test_suggested_clip_workers_scales_by_eight_cores(monkeypatch, cpu_count, expected):
    monkeypatch.setattr(gemini_subs.os, "cpu_count", lambda: cpu_count)
    assert gemini_subs.suggested_clip_workers() == expected


@pytest.mark.parametrize(
    ("ext", "codec", "encoder"),
    [
        (".webm", "vp9", "libvpx-vp9"),
        (".mp4", "h264", "libx264"),
        (".mp4", "hevc", "libx265"),
    ],
)
def test_overlap_codec_args_select_current_encoders(ext, codec, encoder):
    assert encoder in gemini_subs.overlap_codec_args(ext, codec)


def test_overlap_codec_args_rejects_mismatched_container():
    with pytest.raises(ValueError, match="H.264 input requires MP4"):
        gemini_subs.overlap_codec_args(".webm", "h264")


def test_validation_sorts_canonicalizes_clamps_and_preserves_text():
    result = gemini_subs.validate_captions(
        [
            caption(2, "4.2", "12", "Second"),
            caption(1, "00:00:01,250", "00:00:02.5", "First"),
        ],
        10,
    )

    assert [item["id"] for item in result] == [1, 2]
    assert result[0]["start"] == "00:00:01.250"
    assert result[1]["end"] == "00:00:10.500"
    assert result[1]["text"] == "Second"


def test_validation_rejects_duplicate_ids():
    with pytest.raises(ValueError, match=r"Duplicate caption IDs: \[1\]"):
        gemini_subs.validate_captions([caption(1, "0", "1"), caption(1, "2", "3")], 5)


@pytest.mark.parametrize(
    "captions",
    [
        [caption(1, "-1", "1")],
        [caption(1, "1", "1")],
        [caption(1, "10.6", "11")],
    ],
)
def test_validation_rejects_invalid_and_out_of_bounds_timing(captions):
    with pytest.raises(ValueError):
        gemini_subs.validate_captions(captions, 10)


def test_validation_heals_overlaps_in_sorted_output():
    result = gemini_subs.validate_captions(
        [caption(2, "1", "3", "Later"), caption(1, "0", "2", "Earlier")], 5
    )

    assert result[0]["end"] == "00:00:01.000"
    assert result[1]["start"] == "00:00:01.000"


def test_validation_pushes_equal_start_caption_forward():
    result = gemini_subs.validate_captions(
        [caption(1, "1", "2"), caption(2, "1", "3")], 5
    )

    assert result[0]["end"] == "00:00:01.001"
    assert result[1]["start"] == "00:00:01.001"


def test_atomic_write_json_replaces_target_with_unicode_json(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("old", encoding="utf-8")

    gemini_subs.atomic_write_json(path, {"text": "안녕"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "안녕"}
    assert not Path(f"{path}.tmp").exists()


def test_list_chunks_uses_physical_line_index_and_duration(tmp_path):
    (tmp_path / "segments.csv").write_text(
        "ignored\nchunk_001.mp4,1.5,4.25,extra\n", encoding="utf-8"
    )

    assert gemini_subs.list_chunks(tmp_path) == [
        {
            "idx": 1,
            "name": "chunk_001.mp4",
            "start": 1.5,
            "end": 4.25,
            "duration": 2.75,
        }
    ]


def test_clean_incomplete_split_preserves_unrelated_files(tmp_path):
    removable = [
        "chunk_000.mp4",
        "context_chunk_000.webm.tmp",
        "subtitle_chunk_000.json.tmp",
        "segments.csv",
    ]
    preserved = [gemini_subs.MANIFEST_NAME, ".split_complete", "notes.txt"]
    for name in removable + preserved:
        (tmp_path / name).write_text("x", encoding="utf-8")

    gemini_subs.clean_incomplete_split(tmp_path)

    assert not any((tmp_path / name).exists() for name in removable)
    assert all((tmp_path / name).exists() for name in preserved)


def test_split_video_removes_stale_marker_before_cleaning(tmp_path, monkeypatch):
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")
    (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,10\n", encoding="utf-8")

    def check_marker_removed(_chunk_dir):
        assert not marker.exists()
        raise RuntimeError("stop before ffmpeg")

    monkeypatch.setattr(gemini_subs, "clean_incomplete_split", check_marker_removed)

    with pytest.raises(RuntimeError, match="stop before ffmpeg"):
        gemini_subs.split_video("video.mp4", tmp_path, 60, manifest())


def test_failed_split_does_not_leave_stale_marker_for_next_run(tmp_path, monkeypatch):
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")
    (tmp_path / "segments.csv").write_text(
        "chunk_000.mp4,0,10\nchunk_001.mp4,10,20\n", encoding="utf-8"
    )
    (tmp_path / "chunk_000.mp4").write_bytes(b"old")
    calls = 0

    def run_split(_cmd, **_kwargs):
        nonlocal calls
        calls += 1
        (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,10\n", encoding="utf-8")
        (tmp_path / "chunk_000.mp4").write_bytes(b"partial")
        if calls == 1:
            raise subprocess.CalledProcessError(1, "ffmpeg")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run_split)

    with pytest.raises(subprocess.CalledProcessError):
        gemini_subs.split_video("video.mp4", tmp_path, 60, manifest())
    assert not marker.exists()

    gemini_subs.split_video("video.mp4", tmp_path, 60, manifest())

    assert calls == 2
    assert marker.read_text(encoding="utf-8") == "ok\n"


def test_cached_captions_load_valid_data(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(
        json.dumps([{"id": 0, "start": "0", "end": "1", "text": "Hi"}]),
        encoding="utf-8",
    )

    result = gemini_subs.load_cached_captions(path, 2)

    assert result[0]["start"] == "00:00:00.000"


def test_cached_captions_delete_invalid_data(tmp_path, capsys):
    path = tmp_path / "captions.json"
    path.write_text("not json", encoding="utf-8")

    assert gemini_subs.load_cached_captions(path, 2) is None
    assert not path.exists()
    assert "Ignoring invalid cached output" in capsys.readouterr().out


def test_acquire_and_release_lock(tmp_path):
    lock_file = gemini_subs.acquire_lock(tmp_path)
    lock_path = tmp_path / gemini_subs.LOCK_NAME

    assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
    with pytest.raises(RuntimeError, match=f"Another run \\(PID {os.getpid()}\\)"):
        gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(lock_file)

    replacement_lock = gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(replacement_lock)
    assert lock_path.exists()


def test_acquire_lock_ignores_stale_pid_file(tmp_path):
    lock_path = tmp_path / gemini_subs.LOCK_NAME
    lock_path.write_text("999999999", encoding="utf-8")

    lock_file = gemini_subs.acquire_lock(tmp_path)

    assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
    gemini_subs.release_lock(lock_file)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("vp9\n", (".webm", "video/webm", "vp9")),
        ("h264\n", (".mp4", "video/mp4", "h264")),
        ("hevc\n", (".mp4", "video/mp4", "hevc")),
    ],
)
def test_probe_video_format_detects_supported_codecs(monkeypatch, output, expected):
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    assert gemini_subs.probe_video_format("video file") == expected
    assert run.call_args.args[0][-1] == "video file"


def test_probe_video_format_selects_primary_video_stream(monkeypatch):
    def run(cmd, **_kwargs):
        selector = cmd[cmd.index("-select_streams") + 1]
        output = {"v:0": "h264\n", "v:1": "vp9\n"}[selector]
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    assert gemini_subs.probe_video_format("multi-video") == (
        ".mp4",
        "video/mp4",
        "h264",
    )


def test_probe_video_format_rejects_unsupported_primary_stream(monkeypatch):
    def run(cmd, **_kwargs):
        selector = cmd[cmd.index("-select_streams") + 1]
        output = {"v:0": "av1\n", "v:1": "h264\n"}[selector]
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        run,
    )
    with pytest.raises(RuntimeError, match="Video format not supported"):
        gemini_subs.probe_video_format("video")


def test_probe_video_format_wraps_subprocess_failure(monkeypatch):
    def missing_ffprobe(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(gemini_subs.subprocess, "run", missing_ffprobe)
    with pytest.raises(RuntimeError, match="Failed to probe video format"):
        gemini_subs.probe_video_format("video")


def test_split_video_skips_valid_completed_split(tmp_path, monkeypatch):
    write_chunk_layout(tmp_path, [("chunk_000.mp4", 0, 2)])
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
    (tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
    run = MagicMock()
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    gemini_subs.split_video("video.mp4", tmp_path, 60, manifest())

    run.assert_not_called()


def test_split_video_runs_stream_copy_and_marks_completion(tmp_path, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    gemini_subs.split_video("video.mp4", tmp_path, 60, manifest())

    command = run.call_args.args[0]
    assert command[0] == "ffmpeg"
    assert "copy" in command
    assert command[-1] == str(tmp_path / "chunk_%03d.mp4")
    assert (tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER).exists()


def test_create_overlap_clip_builds_command_and_atomically_moves_result(
    tmp_path, monkeypatch
):
    (tmp_path / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(manifest()), encoding="utf-8"
    )

    def create_tmp(command, **_kwargs):
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0)

    run = MagicMock(side_effect=create_tmp)
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    name = gemini_subs.create_overlap_clip(
        "source.mp4", tmp_path, 2, 1.25, 4.75, ".mp4"
    )

    assert name == "context_chunk_002.mp4"
    assert (tmp_path / name).exists()
    command = run.call_args.args[0]
    assert command[command.index("-ss") + 1] == "00:00:01.250"
    assert command[command.index("-t") + 1] == "3.500"
    assert "libx264" in command


def test_create_overlap_clip_reuses_positive_cached_webm_clip(tmp_path, monkeypatch):
    clip = tmp_path / "context_chunk_000.webm"
    clip.write_bytes(b"clip")
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, "2.5\n", ""))
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    name = gemini_subs.create_overlap_clip("source.webm", tmp_path, 0, 0, 2.5, ".webm")

    assert name == clip.name
    assert run.call_count == 1
    probe_command = run.call_args.args[0]
    assert probe_command[probe_command.index("-show_entries") + 1] == "format=duration"
    assert probe_command[0] == "ffprobe"


def test_client_forwards_key_and_base_url(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(gemini_subs.genai, "Client", client)

    gemini_subs.create_client("key", "https://example.test")

    client.assert_called_once_with(
        api_key="key", http_options={"base_url": "https://example.test"}
    )


def test_generate_content_config_uses_schema_and_uppercase_thinking():
    config = gemini_subs.generate_content_config("high")

    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    assert config.response_schema == gemini_subs.SubtitleResponse
    assert config.thinking_config.thinking_level == "HIGH"


def test_thinking_level_validation_only_restricts_minimal():
    gemini_subs.validate_thinking_level_for_model("GEMINI-FLASH", "minimal")
    gemini_subs.validate_thinking_level_for_model("gemini-pro", "high")
    with pytest.raises(ValueError, match="only supported by Flash"):
        gemini_subs.validate_thinking_level_for_model("gemini-pro", "minimal")


def test_process_chunk_uses_valid_cache_without_api_or_media_read(monkeypatch):
    chunk = {
        "idx": 0,
        "clip_name": "missing.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    client = MagicMock()
    monkeypatch.setattr(gemini_subs, "load_cached_captions", lambda *_args: [])
    monkeypatch.setattr(gemini_subs, "create_client", client)

    assert gemini_subs.process_chunk(
        "key", None, chunk, "/missing", "model", "video/mp4", "high"
    )
    client.assert_not_called()


def test_process_chunk_streams_validates_and_writes_caption_list(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    chunk = {
        "idx": 3,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content_stream.return_value = [
        SimpleNamespace(text='{"captions": [{"id": 0, '),
        SimpleNamespace(text='"start": "0", "end": "1", "text": "Hi"}]}'),
    ]
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)
    monkeypatch.setattr(gemini_subs, "load_cached_captions", lambda *_args: None)

    result = gemini_subs.process_chunk(
        "key", None, chunk, tmp_path, "model", "video/mp4", "high"
    )

    assert result
    saved = json.loads(
        (tmp_path / "subtitle_chunk_003.json").read_text(encoding="utf-8")
    )
    assert saved[0]["text"] == "Hi"
    call = client.models.generate_content_stream.call_args
    assert call.kwargs["model"] == "model"
    assert call.kwargs["contents"][0].inline_data.mime_type == "video/mp4"


def test_process_chunk_returns_false_for_invalid_response(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    chunk = {
        "idx": 0,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content_stream.return_value = [
        SimpleNamespace(text="invalid")
    ]
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)
    monkeypatch.setattr(gemini_subs, "load_cached_captions", lambda *_args: None)

    assert not gemini_subs.process_chunk(
        "key", None, chunk, tmp_path, "model", "video/mp4", "high"
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()


def test_stitch_offsets_and_filters_context(tmp_path):
    write_chunk_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 10), ("chunk_001.mp4", 10, 20)],
        overlap=2,
    )
    (tmp_path / "subtitle_chunk_000.json").write_text(
        json.dumps(
            [
                {"start": "1", "end": "3", "text": "First"},
                {"start": "10", "end": "12", "text": "Context only"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "subtitle_chunk_001.json").write_text(
        json.dumps(
            [
                {"start": "0", "end": "1", "text": "Prior context"},
                {"start": "2", "end": "4", "text": "Second"},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.vtt"

    gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [cap.text for cap in result] == ["First", "Second"]
    assert result[0].start == "00:00:01.000"
    assert result[1].start == "00:00:10.000"


def test_stitch_rejects_missing_and_unexpected_results(tmp_path):
    write_chunk_layout(tmp_path, [("chunk_000.mp4", 0, 10)])
    (tmp_path / "subtitle_chunk_002.json").write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"missing chunk indices: \[0\].*unexpected chunk indices: \[2\]",
    ):
        gemini_subs.stitch(tmp_path, tmp_path / "output.vtt")


def test_stitch_without_overlap_keeps_all_captions_and_heals_overlap(tmp_path):
    write_chunk_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
    )
    (tmp_path / "subtitle_chunk_000.json").write_text(
        json.dumps([{"start": "1", "end": "5.5", "text": "One"}]),
        encoding="utf-8",
    )
    (tmp_path / "subtitle_chunk_001.json").write_text(
        json.dumps([{"start": "0", "end": "1", "text": "Two"}]),
        encoding="utf-8",
    )
    output = tmp_path / "output.vtt"

    gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert result[0].end == "00:00:05.000"
    assert result[1].start == "00:00:05.000"


def test_global_refinement_updates_only_valid_ids_and_saves_atomically(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input.vtt"
    output_path = tmp_path / "output.vtt"
    source = webvtt.WebVTT()
    source.captions.extend(
        [
            webvtt.Caption("00:00:00.000", "00:00:01.000", "Old\nline"),
            webvtt.Caption("00:00:02.000", "00:00:03.000", "Keep"),
        ]
    )
    source.save(input_path)
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content_stream.return_value = [
        SimpleNamespace(text='{"changes": [{"id": 0, "text": "New"},'),
        SimpleNamespace(text='{"id": 9, "text": "Ignored"}]}'),
    ]
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        input_path, output_path, "key", None, "refiner", "high"
    )

    result = webvtt.read(output_path)
    assert [cap.text for cap in result] == ["New", "Keep"]
    call = client.models.generate_content_stream.call_args
    assert "Old line" in call.kwargs["contents"]
    assert call.kwargs["model"] == "refiner"
    assert call.kwargs["config"].thinking_config.thinking_level == "HIGH"


def test_global_refinement_exits_for_invalid_model_json(tmp_path, monkeypatch):
    input_path = tmp_path / "input.vtt"
    webvtt.WebVTT().save(input_path)
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content_stream.return_value = [
        SimpleNamespace(text="invalid")
    ]
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            input_path,
            tmp_path / "output.vtt",
            "key",
            None,
            "refiner",
            "high",
        )


def test_refine_only_forwards_current_refinement_defaults(tmp_path, monkeypatch):
    input_path = tmp_path / "input.vtt"
    input_path.write_text("WEBVTT\n", encoding="utf-8")
    refine = MagicMock()
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.delenv("GEMINI_REFINE_MODEL", raising=False)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(input_path),
            "--refine-only",
            "--api-key",
            "key",
            "--output",
            "out.vtt",
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        gemini_subs.main()

    refine.assert_called_once_with(
        str(input_path),
        "out.vtt",
        "key",
        None,
        gemini_subs.DEFAULT_REFINE_MODEL,
        gemini_subs.REFINEMENT_THINKING_LEVEL,
    )


def test_refine_only_allows_in_place_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.vtt"
    input_path.write_text("WEBVTT\n", encoding="utf-8")
    refine = MagicMock()
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(input_path),
            "--refine-only",
            "--api-key",
            "key",
            "--output",
            str(input_path),
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        gemini_subs.main()

    refine.assert_called_once()


def test_main_rejects_output_that_resolves_to_source(tmp_path, monkeypatch, capsys):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "video-alias.mp4"
    output_path.symlink_to(video_path)
    build = MagicMock()
    monkeypatch.setattr(gemini_subs, "build_manifest", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(video_path),
            "--output",
            str(output_path),
            "--api-key",
            "key",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert "--output must not resolve to the source video" in capsys.readouterr().out
    build.assert_not_called()


def test_main_rejects_invalid_overlap_before_pipeline(monkeypatch):
    build = MagicMock()
    monkeypatch.setattr(gemini_subs, "build_manifest", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            "video.mp4",
            "--api-key",
            "key",
            "--chunk-dur",
            "5",
            "--overlap",
            "5",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    build.assert_not_called()


def test_main_cleans_before_release_without_removing_new_owner_files(
    tmp_path, monkeypatch
):
    chunks = [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}]
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    old_artifact = work_dir / "manifest.json"
    old_artifact.write_text("old", encoding="utf-8")
    new_owner_artifact = work_dir / "new-owner.json"
    split = MagicMock()
    process = MagicMock(return_value=[])
    stitch = MagicMock()
    events = []
    real_cleanup = gemini_subs.clean_completed_work
    real_release = gemini_subs.release_lock

    def cleanup(path):
        events.append("cleanup")
        real_cleanup(path)

    def release(lock_file):
        events.append("release")
        real_release(lock_file)
        new_owner_lock = gemini_subs.acquire_lock(work_dir)
        new_owner_artifact.write_text("new", encoding="utf-8")
        real_release(new_owner_lock)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(video_path),
            "--api-key",
            "key",
            "--disable-text-refine",
        ],
    )
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _args: (manifest(), str(work_dir))
    )
    monkeypatch.setattr(gemini_subs, "split_video", split)
    monkeypatch.setattr(gemini_subs, "list_chunks", lambda _path: chunks)
    monkeypatch.setattr(gemini_subs, "process_chunks", process)
    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "release_lock", release)
    monkeypatch.setattr(gemini_subs, "clean_completed_work", cleanup)

    gemini_subs.main()

    split.assert_called_once()
    process.assert_called_once()
    stitch.assert_called_once_with(str(work_dir), "output_subtitles.vtt")
    assert events == ["cleanup", "release"]
    assert not old_artifact.exists()
    assert (work_dir / gemini_subs.LOCK_NAME).exists()
    assert new_owner_artifact.read_text(encoding="utf-8") == "new"


def test_main_preserves_output_and_cleans_staging_when_refinement_fails(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "output.vtt"
    output_path.write_text("previous output", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    staging_paths = []

    def stitch(_chunk_dir, path):
        staging_paths.append(Path(path))
        Path(path).write_text("stitched output", encoding="utf-8")

    def refine(input_path, *_args):
        assert Path(input_path).read_text(encoding="utf-8") == "stitched output"
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(video_path),
            "--api-key",
            "key",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _args: (manifest(), str(work_dir))
    )
    monkeypatch.setattr(gemini_subs, "split_video", lambda *_args: None)
    monkeypatch.setattr(
        gemini_subs,
        "list_chunks",
        lambda _path: [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}],
    )
    monkeypatch.setattr(gemini_subs, "process_chunks", lambda *_args: [])
    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert output_path.read_text(encoding="utf-8") == "previous output"
    assert len(staging_paths) == 1
    assert staging_paths[0].parent == output_path.parent
    assert not staging_paths[0].exists()


def test_main_publishes_refined_output_and_cleans_staging(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "output.vtt"
    output_path.write_text("previous output", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    staging_paths = []

    def stitch(_chunk_dir, path):
        staging_paths.append(Path(path))
        Path(path).write_text("stitched output", encoding="utf-8")

    def refine(input_path, refined_output, *_args):
        assert Path(input_path).read_text(encoding="utf-8") == "stitched output"
        Path(refined_output).write_text("refined output", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(video_path),
            "--api-key",
            "key",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _args: (manifest(), str(work_dir))
    )
    monkeypatch.setattr(gemini_subs, "split_video", lambda *_args: None)
    monkeypatch.setattr(
        gemini_subs,
        "list_chunks",
        lambda _path: [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}],
    )
    monkeypatch.setattr(gemini_subs, "process_chunks", lambda *_args: [])
    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)

    gemini_subs.main()

    assert output_path.read_text(encoding="utf-8") == "refined output"
    assert len(staging_paths) == 1
    assert staging_paths[0].parent == output_path.parent
    assert not staging_paths[0].exists()


def test_main_keeps_work_directory_when_chunk_processing_fails(monkeypatch):
    chunks = [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}]
    release = MagicMock()
    cleanup = MagicMock()
    monkeypatch.setattr(
        sys, "argv", ["gemini_subs.py", "video.mp4", "--api-key", "key"]
    )
    monkeypatch.setattr(gemini_subs.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _args: (manifest(), "work-dir")
    )
    monkeypatch.setattr(gemini_subs.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gemini_subs, "acquire_lock", lambda _path: "work-dir/.lock")
    monkeypatch.setattr(gemini_subs, "split_video", lambda *_args: None)
    monkeypatch.setattr(gemini_subs, "list_chunks", lambda _path: chunks)
    monkeypatch.setattr(gemini_subs, "process_chunks", lambda *_args: ["chunk_000.mp4"])
    monkeypatch.setattr(gemini_subs, "release_lock", release)
    monkeypatch.setattr(gemini_subs, "clean_completed_work", cleanup)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    release.assert_called_once_with("work-dir/.lock")
    cleanup.assert_not_called()


def test_build_manifest_records_current_inputs_and_stable_digest(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    args = argparse.Namespace(
        video_file=str(video),
        overlap=5,
        chunk_dur=60,
        model="model",
        chunk_thinking_level="high",
    )
    monkeypatch.setattr(
        gemini_subs,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(gemini_subs, "CHUNK_ROOT", "chunks")

    first = gemini_subs.build_manifest(args)
    second = gemini_subs.build_manifest(args)

    assert first == second
    built, chunk_dir = first
    assert built["video_codec"] == "h264"
    assert built["chunk_thinking_level"] == "high"
    assert Path(chunk_dir).parent == Path("chunks")
    assert len(Path(chunk_dir).name) == 16
    assert all(character in "0123456789abcdef" for character in Path(chunk_dir).name)
