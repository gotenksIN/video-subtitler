import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import webvtt
from pydantic import ValidationError

import gemini_subs


def make_caption(caption_id, start, end, text="Text"):
    return gemini_subs.Caption(id=caption_id, start=start, end=end, text=text)


def make_manifest(overlap=0.0, codec="h264"):
    ext = ".webm" if codec == "vp9" else ".mp4"
    mime = "video/webm" if ext == ".webm" else "video/mp4"
    return {
        "mode": "generate",
        "overlap": overlap,
        "chunk_ext": ext,
        "chunk_mime": mime,
        "process_ext": ext,
        "process_mime": mime,
        "video_codec": codec,
    }


def write_layout(directory, rows, overlap=0.0):
    Path(directory, gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(overlap)), encoding="utf-8"
    )
    Path(directory, "segments.csv").write_text(
        "".join(f"{name},{start},{end}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def write_subtitles(directory, index, captions):
    Path(directory, f"subtitle_chunk_{index:03d}.json").write_text(
        json.dumps(captions), encoding="utf-8"
    )


def write_vtt(path, captions):
    value = webvtt.WebVTT()
    value.captions.extend(
        webvtt.Caption(start, end, text) for start, end, text in captions
    )
    value.save(path)


class ImmediateFuture:
    def __init__(self, function, args):
        try:
            self.value = function(*args)
            self.error = None
        except Exception as error:  # noqa: BLE001 - Emulate Future exception capture.
            self.value = None
            self.error = error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class ImmediateExecutor:
    def __init__(self, max_workers):
        # ThreadPoolExecutor takes max_workers. Immediate execution does not need it.
        del max_workers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args):
        return ImmediateFuture(function, args)


class FakeGeminiClient:
    """Fake SDK boundary that consumes each request and enforces the documented contract."""

    def __init__(self, model_name, response_pieces, verify_contents):
        self.model_name = model_name
        self.response_pieces = response_pieces
        self.verify_contents = verify_contents

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    def generate_content_stream(self, **kwargs):
        if kwargs["model"] != self.model_name:
            raise AssertionError(
                f"request must use model {self.model_name!r}, got {kwargs['model']!r}"
            )
        config = kwargs["config"]
        if config.response_mime_type != "application/json":
            raise AssertionError("request config must require application/json")
        # Consume the response schema like the SDK structured-output boundary does.
        config.response_schema.model_validate_json("".join(self.response_pieces))
        self.verify_contents(kwargs["contents"])
        return iter(SimpleNamespace(text=piece) for piece in self.response_pieces)


@pytest.fixture
def immediate_execution(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor
    )
    monkeypatch.setattr(
        gemini_subs.concurrent.futures, "as_completed", lambda futures: list(futures)
    )


def test_atomic_json_uses_atomic_replacement_and_leaves_no_temporary_files(
    tmp_path, monkeypatch
):
    target = tmp_path / "captions.json"
    target.write_text("old", encoding="utf-8")
    replacements = []
    real_replace = os.replace

    def replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(gemini_subs.os, "replace", replace)

    gemini_subs.atomic_write_json(target, {"text": "plain ascii"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"text": "plain ascii"}
    assert replacements == [(Path(f"{target}.tmp"), target)]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["captions.json"]


def test_atomic_vtt_uses_unique_atomic_temporary_files_in_output_directory(
    tmp_path, monkeypatch
):
    output = tmp_path / "output.vtt"
    sources = []
    real_replace = os.replace

    def replace(source, destination):
        sources.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr(gemini_subs.os, "replace", replace)
    value = webvtt.WebVTT()
    value.captions.extend(
        [
            webvtt.Caption("00:00:00.000", "00:00:01.000", "One"),
            webvtt.Caption("00:00:01.000", "00:00:02.000", "Two"),
        ]
    )

    gemini_subs.atomic_save_vtt(value, output)
    gemini_subs.atomic_save_vtt(value, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["One", "Two"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output.vtt"]
    assert len(set(sources)) == 2
    assert all(path.parent == tmp_path for path in sources)
    assert all(path.name.endswith(".tmp.vtt") for path in sources)


@pytest.mark.parametrize("failure", ["save", "replace"])
def test_atomic_vtt_removes_temporary_file_after_failure(
    tmp_path, monkeypatch, failure
):
    output = tmp_path / "output.vtt"
    value = MagicMock()
    if failure == "save":
        value.save.side_effect = OSError("save failed")
    else:
        monkeypatch.setattr(
            gemini_subs.os, "replace", MagicMock(side_effect=OSError("replace failed"))
        )

    with pytest.raises(OSError, match=f"{failure} failed"):
        gemini_subs.atomic_save_vtt(value, output)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("1.25", 1.25), ("02:03,5", 123.5), ("01:02:03.004", 3723.004)],
)
def test_timestamp_parser_accepts_documented_shapes(value, seconds):
    assert gemini_subs.parse_time(value) == seconds


@pytest.mark.parametrize("value", ["-0.1", "-00:00:00.1", "1:2:3:4"])
def test_timestamp_parser_rejects_negative_or_malformed_values(value):
    with pytest.raises(ValueError):
        gemini_subs.parse_time(value)


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(0, "00:00:00.000"), (3661.2346, "01:01:01.235"), (59.9996, "00:01:00.000")],
)
def test_timestamp_formatter_rounds_to_milliseconds(seconds, formatted):
    assert gemini_subs.format_time(seconds) == formatted


def test_timestamp_formatter_rejects_negative_values():
    with pytest.raises(ValueError, match="Negative timestamp"):
        gemini_subs.format_time(-0.001)


def test_caption_validation_sorts_canonicalizes_and_heals_overlap():
    result = gemini_subs.validate_captions(
        [
            make_caption(2, "1", "3", "Later"),
            make_caption(1, "00:00:00,250", "2", "Earlier"),
        ],
        5,
    )

    assert [item["id"] for item in result] == [1, 2]
    assert result[0] == {
        "id": 1,
        "start": "00:00:00.250",
        "end": "00:00:01.000",
        "text": "Earlier",
    }
    assert result[1]["start"] == "00:00:01.000"


def test_caption_validation_nudges_cues_with_the_same_start():
    result = gemini_subs.validate_captions(
        [make_caption(0, "1", "2"), make_caption(1, "1", "3")], 5
    )

    assert result[0]["end"] == "00:00:01.001"
    assert result[1]["start"] == "00:00:01.001"


@pytest.mark.parametrize(
    "captions",
    [
        [make_caption(0, "0", "1"), make_caption(0, "2", "3")],
        [make_caption(0, "1", "1")],
        [make_caption(0, "9", "10.6")],
        [make_caption(0, "10.2", "10.4")],
        [make_caption(0, "9.9996", "10.4")],
    ],
    ids=[
        "duplicate IDs",
        "non-positive interval",
        "overrun beyond tolerance",
        "clamp makes interval invalid",
        "rounding collapses interval",
    ],
)
def test_caption_validation_rejects_invalid_responses(captions):
    with pytest.raises(ValueError):
        gemini_subs.validate_captions(captions, 10)


def test_caption_validation_clamps_allowed_end_overrun_to_clip_duration():
    result = gemini_subs.validate_captions([make_caption(0, "9", "10.4")], 10)
    assert result[0]["end"] == "00:00:10.000"


@pytest.mark.parametrize(
    ("probe_output", "expected"),
    [
        ("vp9\n", (".webm", "video/webm", "vp9")),
        ("h264\n", (".mp4", "video/mp4", "h264")),
        ("h265\n", (".mp4", "video/mp4", "hevc")),
    ],
)
def test_primary_video_probe_maps_supported_codecs(monkeypatch, probe_output, expected):
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout=probe_output, stderr="")
    )
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    assert gemini_subs.probe_video_format("source") == expected
    command = run.call_args.args[0]
    assert command[command.index("-select_streams") + 1] == "v:0"


def test_primary_video_probe_rejects_unsupported_codec(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="av1\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="Video format not supported"):
        gemini_subs.probe_video_format("source")


def test_primary_video_probe_wraps_tool_failure(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        MagicMock(side_effect=FileNotFoundError("ffprobe")),
    )
    with pytest.raises(RuntimeError, match="Failed to probe video format"):
        gemini_subs.probe_video_format("source")


def test_manifest_contains_documented_identity_and_uses_sorted_hash(
    tmp_path, monkeypatch
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    args = argparse.Namespace(
        video_file=str(video),
        chunk_dur=60,
        model="chunk-model",
        chunk_thinking_level="high",
        overlap=5.0,
    )
    monkeypatch.setattr(
        gemini_subs,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(gemini_subs, "CHUNK_ROOT", str(tmp_path / "work"))

    manifest, work_dir = gemini_subs.build_manifest(args)
    expected_fields = {
        "video",
        "chunk_dur",
        "format",
        "mode",
        "model",
        "chunk_thinking_level",
        "overlap",
        "chunk_ext",
        "chunk_mime",
        "process_ext",
        "process_mime",
        "video_codec",
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    assert set(manifest) == expected_fields
    assert manifest["video"] == {
        "path": str(video.resolve()),
        "size": 5,
        "mtime_ns": video.stat().st_mtime_ns,
    }
    assert manifest["format"] == "stream-copy-v1"
    assert manifest["mode"] == "generate"
    assert Path(work_dir) == tmp_path / "work" / digest


def test_manifest_identity_changes_with_runtime_input(tmp_path, monkeypatch):
    video = tmp_path / "source.webm"
    video.write_bytes(b"video")
    args = argparse.Namespace(
        video_file=str(video),
        chunk_dur=60,
        model="model-a",
        chunk_thinking_level="high",
        overlap=0,
    )
    monkeypatch.setattr(
        gemini_subs,
        "probe_video_format",
        lambda _path: (".webm", "video/webm", "vp9"),
    )

    _, first = gemini_subs.build_manifest(args)
    args.model = "model-b"
    _, second = gemini_subs.build_manifest(args)

    assert first != second


def test_segment_index_is_source_of_chunk_timing_and_physical_index(tmp_path):
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


def test_completed_split_with_nonempty_chunks_is_reused(tmp_path, monkeypatch):
    write_layout(tmp_path, [("chunk_000.mp4", 0, 2)])
    chunk = tmp_path / "chunk_000.mp4"
    chunk.write_bytes(b"cached chunk")
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("a valid completed split must not run ffmpeg")

    monkeypatch.setattr(gemini_subs.subprocess, "run", fail_if_called)

    gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert chunk.read_bytes() == b"cached chunk"
    assert marker.read_text(encoding="utf-8") == "ok\n"


def test_invalid_completed_split_is_cleaned_then_recreated(tmp_path, monkeypatch):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 2), ("chunk_001.mp4", 2, 4)],
    )
    (tmp_path / "chunk_000.mp4").write_bytes(b"only one chunk")
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")

    def run(command, **_kwargs):
        assert not marker.exists()
        assert not (tmp_path / "chunk_000.mp4").exists()
        assert command[command.index("-map") + 1] == "0:v:0"
        assert "0:a?" in command
        assert "-sn" in command
        for name in ("chunk_000.mp4", "chunk_001.mp4"):
            (tmp_path / name).write_bytes(b"fresh chunk")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert marker.read_text(encoding="utf-8") == "ok\n"
    assert (tmp_path / "chunk_000.mp4").read_bytes() == b"fresh chunk"
    assert (tmp_path / "chunk_001.mp4").read_bytes() == b"fresh chunk"


def test_failed_split_leaves_no_completion_marker_for_resume(tmp_path, monkeypatch):
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")
    (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,2\n", encoding="utf-8")
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, "ffmpeg")),
    )

    with pytest.raises(subprocess.CalledProcessError):
        gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert not marker.exists()


def test_processing_windows_clip_context_at_video_edges():
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0.0, "end": 10.0},
        {"idx": 1, "name": "chunk_001.mp4", "start": 10.0, "end": 18.0},
    ]

    windows = gemini_subs.get_processing_windows(chunks, 2.0)

    assert windows[0]["clip_start"] == 0
    assert windows[0]["clip_end"] == 12
    assert windows[1]["clip_start"] == 8
    assert windows[1]["clip_end"] == 18
    assert windows[1]["clip_duration"] == 10
    assert windows[1]["owner_start_rel"] == 2
    assert windows[1]["owner_end_rel"] == 10


@pytest.mark.parametrize(
    ("ext", "codec", "encoder", "audio"),
    [
        (".webm", "vp9", "libvpx-vp9", "libopus"),
        (".mp4", "h264", "libx264", "aac"),
        (".mp4", "hevc", "libx265", "aac"),
    ],
)
def test_overlap_codec_configuration_matches_primary_codec(ext, codec, encoder, audio):
    arguments = gemini_subs.overlap_codec_args(ext, codec)
    assert encoder in arguments
    assert audio in arguments
    assert arguments[arguments.index("-crf") + 1] == "32"


def test_overlap_codec_configuration_rejects_container_mismatch():
    with pytest.raises(ValueError, match="H.264 input requires MP4"):
        gemini_subs.overlap_codec_args(".webm", "h264")


def test_overlap_clip_command_seeks_before_input(tmp_path, monkeypatch):
    (tmp_path / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(codec="h264")), encoding="utf-8"
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"new clip")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    result = gemini_subs.create_overlap_clip(
        "source.mp4", tmp_path, 0, 1.25, 4.75, ".mp4"
    )

    assert result == "context_chunk_000.mp4"
    command = calls[0]
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "00:00:01.250"


def test_valid_overlap_clip_cache_is_reused_without_reencoding(tmp_path, monkeypatch):
    clip = tmp_path / "context_chunk_000.mp4"
    clip.write_bytes(b"cached")
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout="2.5\n", stderr="")
    )
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    result = gemini_subs.create_overlap_clip("source.mp4", tmp_path, 0, 0, 2.5, ".mp4")

    assert result == clip.name
    assert clip.read_bytes() == b"cached"


def test_invalid_overlap_cache_is_regenerated_and_leaves_no_temporary_file(
    tmp_path, monkeypatch
):
    clip = tmp_path / "context_chunk_002.mp4"
    clip.write_bytes(b"invalid")
    (tmp_path / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(codec="h264")), encoding="utf-8"
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad")
        Path(command[-1]).write_bytes(b"new clip")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    result = gemini_subs.create_overlap_clip(
        "source.mp4", tmp_path, 2, 1.25, 4.75, ".mp4"
    )

    assert result == clip.name
    assert clip.read_bytes() == b"new clip"
    assert not Path(f"{clip}.tmp").exists()
    command = calls[1]
    assert command[command.index("-ss") + 1] == "00:00:01.250"
    assert command[command.index("-t") + 1] == "3.500"
    assert "libx264" in command
    assert command[command.index("-crf") + 1] == "32"


def test_subtitle_response_schema_accepts_documented_caption_shape():
    response = gemini_subs.SubtitleResponse.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )

    assert response.captions[0].id == 0
    assert response.captions[0].text == "Hi"


def test_subtitle_response_schema_rejects_malformed_captions():
    with pytest.raises(ValidationError):
        gemini_subs.SubtitleResponse.model_validate(
            {"captions": [{"id": "not-an-int"}]}
        )


def test_generation_config_meets_documented_request_contract(monkeypatch):
    contract = {}

    def make_config(**kwargs):
        contract.update(kwargs)
        return object()

    def make_thinking_config(**kwargs):
        return kwargs

    def make_afc_config(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        gemini_subs,
        "types",
        SimpleNamespace(
            GenerateContentConfig=make_config,
            ThinkingConfig=make_thinking_config,
            AutomaticFunctionCallingConfig=make_afc_config,
        ),
    )

    gemini_subs.generate_content_config("high")

    assert contract["response_mime_type"] == "application/json"
    assert contract["automatic_function_calling"].disable is True
    schema = contract["response_schema"]
    parsed = schema.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )
    assert parsed.captions[0].text == "Hi"
    assert contract["thinking_config"] == {"thinking_level": "HIGH"}


def test_minimal_thinking_is_only_valid_for_flash_models():
    gemini_subs.validate_thinking_level_for_model("Gemini-FLASH", "minimal")
    gemini_subs.validate_thinking_level_for_model("gemini-pro", "high")
    with pytest.raises(ValueError, match="only supported by Flash"):
        gemini_subs.validate_thinking_level_for_model("gemini-pro", "minimal")


def test_generation_prompt_includes_clip_and_owner_timing():
    prompt = gemini_subs.build_generation_prompt(12.0, 2.0, 10.0)

    assert "12.000-second" in prompt
    assert "00:00:02.000 to 00:00:10.000" in prompt


def test_valid_chunk_cache_skips_media_read_and_api(monkeypatch):
    chunk = {
        "idx": 0,
        "clip_name": "missing.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    monkeypatch.setattr(gemini_subs, "load_cached_captions", lambda *_args: [])

    def fail_if_called(*_args):
        raise AssertionError("no API client may be created for cached captions")

    monkeypatch.setattr(gemini_subs, "create_client", fail_if_called)

    assert gemini_subs.process_chunk(
        "key", None, chunk, "/missing", "model", "video/mp4", "high"
    )


def test_invalid_chunk_cache_is_removed_for_regeneration(tmp_path):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text("invalid", encoding="utf-8")

    assert gemini_subs.load_cached_captions(cache, 2) is None
    assert not cache.exists()


def test_chunk_request_saves_canonical_caption_array(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video bytes")
    chunk = {
        "idx": 3,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    pieces = ['{"captions": [{"id": 0,', '"start": "0", "end": "1", "text": "Hi"}]}']

    def verify_contents(contents):
        video_part, prompt = contents
        if video_part.inline_data.mime_type != "video/mp4":
            raise AssertionError(
                "video part must use video/mp4, "
                f"got {video_part.inline_data.mime_type!r}"
            )
        if video_part.inline_data.data != b"video bytes":
            raise AssertionError("video part must contain the clip bytes")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AssertionError("request must include the generation prompt")

    client = FakeGeminiClient("model", pieces, verify_contents)
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    assert gemini_subs.process_chunk(
        "key", "base", chunk, tmp_path, "model", "video/mp4", "high"
    )

    saved = json.loads(
        (tmp_path / "subtitle_chunk_003.json").read_text(encoding="utf-8")
    )
    assert saved == [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "Hi",
        }
    ]


def test_chunk_request_failure_does_not_publish_result(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    chunk = {
        "idx": 0,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    client = FakeGeminiClient("model", ["bad"], lambda _contents: None)
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    assert not gemini_subs.process_chunk(
        "key", None, chunk, tmp_path, "model", "video/mp4", "high"
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()


def test_process_chunks_without_overlap_processes_every_chunk_and_reports_failures(
    tmp_path, monkeypatch, immediate_execution
):
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2},
        {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4},
    ]

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"processed_{chunk['idx']}").write_text(
            "done", encoding="utf-8"
        )
        return chunk["idx"] == 0

    monkeypatch.setattr(gemini_subs, "process_chunk", process)
    failed = gemini_subs.process_chunks(
        "key",
        None,
        "source",
        str(tmp_path),
        chunks,
        0,
        ".mp4",
        7,
        3,
        "model",
        "video/mp4",
        "high",
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "processed_0",
        "processed_1",
    ]
    assert failed == ["chunk_001.mp4"]


def test_process_chunks_overlap_keeps_processing_clips_after_a_clip_failure(
    tmp_path, monkeypatch, immediate_execution
):
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2},
        {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4},
    ]

    def attach(_video, _directory, chunk, _overlap, _ext):
        if chunk["idx"] == 1:
            raise RuntimeError("encode failed")
        return {**chunk, "clip_name": "context_chunk_000.mp4"}

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"processed_{chunk['idx']}").write_text(
            "done", encoding="utf-8"
        )
        return False

    monkeypatch.setattr(gemini_subs, "attach_overlap_clip", attach)
    monkeypatch.setattr(gemini_subs, "process_chunk", process)
    failed = gemini_subs.process_chunks(
        "key",
        None,
        "source",
        str(tmp_path),
        chunks,
        1,
        ".mp4",
        2,
        4,
        "model",
        "video/mp4",
        "high",
    )

    assert (tmp_path / "processed_0").read_text(encoding="utf-8") == "done"
    assert not (tmp_path / "processed_1").exists()
    assert failed == ["context_chunk_001.mp4", "context_chunk_000.mp4"]


def test_stitch_applies_clip_offset_and_half_open_midpoint_ownership(tmp_path):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 10), ("chunk_001.mp4", 10, 20)],
        overlap=2,
    )
    write_subtitles(
        tmp_path,
        0,
        [
            {"start": "9", "end": "11", "text": "Right edge excluded"},
            {"start": "8", "end": "10", "text": "First owner"},
        ],
    )
    write_subtitles(
        tmp_path,
        1,
        [
            {"start": "1", "end": "3", "text": "Left edge included"},
            {"start": "4", "end": "6", "text": "Offset caption"},
        ],
    )
    output = tmp_path / "output.vtt"

    gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == [
        "First owner",
        "Left edge included",
        "Offset caption",
    ]
    assert [caption.start for caption in result] == [
        "00:00:08.000",
        "00:00:09.000",
        "00:00:12.000",
    ]


def test_stitch_rejects_missing_and_unexpected_chunk_results(tmp_path):
    write_layout(tmp_path, [("chunk_000.mp4", 0, 5)])
    write_subtitles(tmp_path, 2, [])

    with pytest.raises(
        ValueError,
        match=r"missing chunk indices: \[0\].*unexpected chunk indices: \[2\]",
    ):
        gemini_subs.stitch(tmp_path, tmp_path / "output.vtt")


def test_stitch_preserves_repeated_text_and_heals_cross_chunk_overlap(tmp_path):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
    )
    write_subtitles(tmp_path, 0, [{"start": "1", "end": "5.5", "text": "Again"}])
    write_subtitles(tmp_path, 1, [{"start": "0", "end": "1", "text": "Again"}])
    output = tmp_path / "output.vtt"

    gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["Again", "Again"]
    assert result[0].end == "00:00:05.000"
    assert result[1].start == "00:00:05.000"


def test_refinement_prompt_contains_complete_indexed_script():
    script = "[37] 12:34:56.789 --> 12:34:58.012: Unique caption text"
    prompt = gemini_subs.build_refinement_prompt(script)

    assert script in prompt


@pytest.mark.parametrize(
    "changes",
    [
        [
            gemini_subs.RefinedCaption(id=0, text="One"),
            gemini_subs.RefinedCaption(id=0, text="Two"),
        ],
        [gemini_subs.RefinedCaption(id=2, text="Out")],
        [gemini_subs.RefinedCaption(id=0, text="  ")],
    ],
    ids=["duplicate", "out of range", "empty"],
)
def test_refinement_change_validation_is_all_or_nothing(changes):
    with pytest.raises(ValueError):
        gemini_subs.validate_refinement_changes(changes, 2)


def test_global_refinement_changes_only_text_and_preserves_timestamps(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(
        source,
        [
            ("00:00:00.000", "00:00:01.000", "Old\nline"),
            ("00:00:02.000", "00:00:03.000", "Keep"),
        ],
    )

    def verify_contents(contents):
        if "[0] 00:00:00.000 --> 00:00:01.000: Old\nline" not in contents:
            raise AssertionError("refinement prompt must contain the indexed script")

    client = FakeGeminiClient(
        "refiner", ['{"changes": [{"id": 0, "text": "New"}]}'], verify_contents
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["New", "Keep"]
    assert [(caption.start, caption.end) for caption in result] == [
        ("00:00:00.000", "00:00:01.000"),
        ("00:00:02.000", "00:00:03.000"),
    ]


def test_invalid_refinement_does_not_mutate_or_publish(tmp_path, monkeypatch):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(
        source,
        [
            ("00:00:00.000", "00:00:01.000", "First"),
            ("00:00:02.000", "00:00:03.000", "Second"),
        ],
    )
    client = FakeGeminiClient(
        "model",
        [
            json.dumps(
                {"changes": [{"id": 0, "text": "Changed"}, {"id": 2, "text": "Bad"}]}
            )
        ],
        lambda _contents: None,
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "model",
            "high",
        )

    result = webvtt.read(source)
    assert [caption.text for caption in result] == ["First", "Second"]
    assert output.read_text(encoding="utf-8") == "previous"


def test_process_lock_blocks_second_owner_and_survives_release(tmp_path):
    first = gemini_subs.acquire_lock(tmp_path)
    lock_path = tmp_path / gemini_subs.LOCK_NAME
    try:
        assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
        with pytest.raises(RuntimeError, match=f"PID {os.getpid()}"):
            gemini_subs.acquire_lock(tmp_path)
    finally:
        gemini_subs.release_lock(first)

    second = gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(second)
    assert lock_path.exists()


def test_process_lock_treats_unlocked_pid_text_as_stale(tmp_path):
    lock_path = tmp_path / gemini_subs.LOCK_NAME
    lock_path.write_text("999999", encoding="utf-8")

    lock = gemini_subs.acquire_lock(tmp_path)
    try:
        assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
    finally:
        gemini_subs.release_lock(lock)


@pytest.mark.parametrize(
    ("arguments", "existing_input", "message"),
    [
        (["missing.mp4", "--api-key", "key"], False, "Video file not found"),
        (["input.mp4"], True, "API key not configured"),
        (["input.mp4", "--api-key", "key", "--chunk-dur", "0"], True, "chunk-dur"),
        (["input.mp4", "--api-key", "key", "--workers", "0"], True, "workers"),
        (["input.mp4", "--api-key", "key", "--overlap", "-1"], True, "overlap"),
        (
            ["input.mp4", "--api-key", "key", "--chunk-dur", "5", "--overlap", "5"],
            True,
            "smaller",
        ),
        (
            [
                "input.mp4",
                "--api-key",
                "key",
                "--model",
                "pro",
                "--thinking-level",
                "minimal",
            ],
            True,
            "Flash",
        ),
    ],
)
def test_cli_rejects_invalid_generation_inputs_before_pipeline(
    tmp_path, monkeypatch, capsys, arguments, existing_input, message
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    input_path = tmp_path / arguments[0]
    if existing_input:
        input_path.write_bytes(b"video")
    arguments = [str(input_path), *arguments[1:]]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pipeline must not start for invalid CLI inputs")

    monkeypatch.setattr(gemini_subs, "build_manifest", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["gemini_subs.py", *arguments])

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert message in capsys.readouterr().out


def test_cli_rejects_generation_output_resolving_to_source(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemini_subs.py", str(source), "--api-key", "key", "--output", str(alias)],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()


@pytest.mark.parametrize(
    "input_exists", [False, True], ids=["missing input", "missing key"]
)
def test_refine_only_validates_input_and_api_key(tmp_path, monkeypatch, input_exists):
    source = tmp_path / "source.vtt"
    if input_exists:
        source.write_text("WEBVTT\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("refinement must not run when CLI validation fails")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", fail_if_called)
    arguments = ["gemini_subs.py", str(source), "--refine-only"]
    if not input_exists:
        arguments.extend(["--api-key", "key"])
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()


def test_refine_only_refines_in_place_without_running_video_pipeline(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.vtt"
    source.write_text("WEBVTT\n", encoding="utf-8")

    def fail_if_called(*_args):
        raise AssertionError("video pipeline must not run in refine-only mode")

    def refine(_input_path, output_path, *_args):
        Path(output_path).write_text("refined\n", encoding="utf-8")

    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)
    monkeypatch.setattr(gemini_subs, "build_manifest", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(source),
            "--refine-only",
            "--api-key",
            "key",
            "--output",
            str(source),
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        gemini_subs.main()

    assert source.read_text(encoding="utf-8") == "refined\n"


def prepare_generation_main(tmp_path, monkeypatch, process_result=None):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    work = tmp_path / "work"
    work.mkdir()
    chunks = [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}]
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _args: (make_manifest(), str(work))
    )
    monkeypatch.setattr(gemini_subs, "split_video", MagicMock())
    monkeypatch.setattr(gemini_subs, "list_chunks", lambda _path: chunks)
    monkeypatch.setattr(
        gemini_subs,
        "process_chunks",
        MagicMock(return_value=[] if process_result is None else process_result),
    )
    return source, work


def test_successful_generation_without_refinement_cleans_work_before_unlock(
    tmp_path, monkeypatch
):
    source, work = prepare_generation_main(tmp_path, monkeypatch)
    artifact = work / "manifest.json"
    artifact.write_text("state", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    real_release = gemini_subs.release_lock

    def release_after_cleanup(lock_file):
        # The documented contract releases the lock only after work cleanup.
        assert sorted(path.name for path in work.iterdir()) == [gemini_subs.LOCK_NAME]
        real_release(lock_file)

    def stitch(_directory, path):
        Path(path).write_text("stitched\n", encoding="utf-8")

    monkeypatch.setattr(gemini_subs, "release_lock", release_after_cleanup)
    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemini_subs.py", str(source), "--api-key", "key", "--disable-text-refine"],
    )

    gemini_subs.main()

    assert (tmp_path / "output_subtitles.vtt").read_text(encoding="utf-8") == (
        "stitched\n"
    )
    assert sorted(path.name for path in work.iterdir()) == [gemini_subs.LOCK_NAME]
    lock = gemini_subs.acquire_lock(work)
    gemini_subs.release_lock(lock)


def test_failed_chunk_processing_keeps_resume_state_and_releases_lock(
    tmp_path, monkeypatch
):
    source, work = prepare_generation_main(
        tmp_path, monkeypatch, process_result=["chunk_000.mp4"]
    )
    artifact = work / "subtitle_chunk_000.json"
    artifact.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["gemini_subs.py", str(source), "--api-key", "key"]
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert artifact.exists()
    lock = gemini_subs.acquire_lock(work)
    gemini_subs.release_lock(lock)


def test_refinement_failure_preserves_output_removes_staging_and_resume_state(
    tmp_path, monkeypatch
):
    source, work = prepare_generation_main(tmp_path, monkeypatch)
    state = work / "manifest.json"
    state.write_text("state", encoding="utf-8")
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    received = {}

    def stitch(_directory, path):
        Path(path).write_text("stitched", encoding="utf-8")

    def refine(input_path, *_args):
        received["path"] = Path(input_path)
        received["content"] = Path(input_path).read_text(encoding="utf-8")
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemini_subs.py", str(source), "--api-key", "key", "--output", str(output)],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert received["content"] == "stitched"
    assert received["path"].name.endswith(".staging.vtt")
    assert received["path"].parent == output.parent
    assert output.read_text(encoding="utf-8") == "previous"
    assert state.exists()
    assert not list(tmp_path.glob("*.staging.vtt"))
