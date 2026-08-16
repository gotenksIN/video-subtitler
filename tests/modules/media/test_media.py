"""Media pipeline behavior against real FFmpeg fixtures.

Tests exercise probing, splitting, segment listing, processing windows,
and overlap clips on tiny generated videos. They assert observable
artifact properties, not command lines or internal representation.
"""

import itertools
import json
import shutil
import subprocess

import pytest

import gemini_subs

VIDEO_WIDTH = 160
VIDEO_HEIGHT = 90
VIDEO_RATE = 25
VIDEO_SECONDS = 6.0
KEYFRAME_INTERVAL = 25  # One keyframe per second at 25 fps.

CODEC_SPECS = {
    "vp9": {
        "ext": ".webm",
        "mime": "video/webm",
        "codec": "vp9",
        "encode_args": [
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "150k",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-g",
            str(KEYFRAME_INTERVAL),
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
        ],
    },
    "h264": {
        "ext": ".mp4",
        "mime": "video/mp4",
        "codec": "h264",
        "encode_args": [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-g",
            str(KEYFRAME_INTERVAL),
            "-c:a",
            "aac",
            "-b:a",
            "24k",
            "-movflags",
            "+faststart",
        ],
    },
    "hevc": {
        "ext": ".mp4",
        "mime": "video/mp4",
        "codec": "hevc",
        "encode_args": [
            "-c:v",
            "libx265",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-x265-params",
            f"keyint={KEYFRAME_INTERVAL}:min-keyint={KEYFRAME_INTERVAL}",
            "-c:a",
            "aac",
            "-b:a",
            "24k",
            "-movflags",
            "+faststart",
            "-tag:v",
            "hvc1",
        ],
    },
}


@pytest.fixture(scope="session")
def media_tools():
    """Resolve ffmpeg and ffprobe or fail with a clear message."""
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing = [name for name, path in tools.items() if not path]
    if missing:
        pytest.fail(
            "media tests require " + " and ".join(missing) + " in PATH; "
            "install them with ./scripts/ffmpeg.sh",
            pytrace=False,
        )
    return tools


def _generate_video(ffmpeg, directory, name, encode_args):
    """Encode one tiny test video and fail clearly when FFmpeg cannot."""
    path = directory / name
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            f"testsrc2=size={VIDEO_WIDTH}x{VIDEO_HEIGHT}:"
            f"rate={VIDEO_RATE}:duration={VIDEO_SECONDS}"
        ),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={VIDEO_SECONDS}",
        "-shortest",
        *encode_args,
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"could not generate the {name} fixture with {ffmpeg}:\n"
            f"{result.stderr.strip()}",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="session", params=list(CODEC_SPECS))
def video_fixture(request, media_tools, tmp_path_factory):
    """One tiny session-scoped VP9, H.264, and HEVC source video."""
    spec = CODEC_SPECS[request.param]
    directory = tmp_path_factory.mktemp(f"fixture-{request.param}")
    path = _generate_video(
        media_tools["ffmpeg"],
        directory,
        f"source{spec['ext']}",
        spec["encode_args"],
    )
    return {
        "codec": spec["codec"],
        "ext": spec["ext"],
        "mime": spec["mime"],
        "path": path,
    }


@pytest.fixture(scope="session")
def mpeg4_video(media_tools, tmp_path_factory):
    """A decodable MPEG-4 video that the pipeline does not support."""
    directory = tmp_path_factory.mktemp("fixture-unsupported")
    return _generate_video(
        media_tools["ffmpeg"],
        directory,
        "unsupported.avi",
        ["-c:v", "mpeg4", "-q:v", "8", "-an", "-f", "avi"],
    )


@pytest.fixture(scope="session")
def h264_video_with_subtitles(media_tools, tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixture-subtitles")
    source = _generate_video(
        media_tools["ffmpeg"],
        directory,
        "source.mp4",
        CODEC_SPECS["h264"]["encode_args"],
    )
    subtitles = directory / "captions.srt"
    subtitles.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nInternal subtitle\n",
        encoding="utf-8",
    )
    output = directory / "source-with-subtitles.mp4"
    command = [
        media_tools["ffmpeg"],
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-i",
        str(subtitles),
        "-map",
        "0",
        "-map",
        "1:0",
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"could not generate a subtitle-stream fixture:\n{result.stderr.strip()}",
            pytrace=False,
        )
    return output


def _ffprobe_lines(ffprobe, path, entries, select_streams=None):
    """Return nonempty ffprobe output lines for one show_entries query."""
    command = [ffprobe, "-v", "error"]
    if select_streams:
        command += ["-select_streams", select_streams]
    command += [
        "-show_entries",
        entries,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"ffprobe failed for {path}:\n{result.stderr.strip()}", pytrace=False
        )
    return [line for line in result.stdout.splitlines() if line]


def _primary_video_codec(ffprobe, path):
    lines = _ffprobe_lines(ffprobe, path, "stream=codec_name", select_streams="v:0")
    if len(lines) != 1:
        pytest.fail(
            f"expected one video stream in {path}, got {lines!r}", pytrace=False
        )
    return lines[0]


def _container_duration(ffprobe, path):
    lines = _ffprobe_lines(ffprobe, path, "format=duration")
    if len(lines) != 1:
        pytest.fail(f"expected one duration for {path}, got {lines!r}", pytrace=False)
    return float(lines[0])


def _stream_types(ffprobe, path):
    return _ffprobe_lines(ffprobe, path, "stream=codec_type")


def test_probe_video_format_identifies_supported_primary_codec(video_fixture):
    assert gemini_subs.probe_video_format(str(video_fixture["path"])) == (
        video_fixture["ext"],
        video_fixture["mime"],
        video_fixture["codec"],
    )


def test_probe_video_format_rejects_unsupported_codec(mpeg4_video):
    with pytest.raises(RuntimeError, match="Video format not supported"):
        gemini_subs.probe_video_format(str(mpeg4_video))


def test_probe_video_format_reports_missing_input(media_tools, tmp_path):
    with pytest.raises(RuntimeError, match="Failed to probe video format"):
        gemini_subs.probe_video_format(str(tmp_path / "missing.webm"))


def test_split_creates_decodable_chunks_and_completion_marker(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    gemini_subs.split_video(
        str(video_fixture["path"]),
        str(chunk_dir),
        2,
        {"chunk_ext": video_fixture["ext"]},
    )

    marker = chunk_dir / gemini_subs.SPLIT_COMPLETE_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    chunks = gemini_subs.list_chunks(str(chunk_dir))
    assert len(chunks) >= 2
    source_duration = _container_duration(media_tools["ffprobe"], video_fixture["path"])
    for position, chunk in enumerate(chunks):
        path = chunk_dir / chunk["name"]
        assert path.is_file() and path.stat().st_size > 0
        assert chunk["idx"] == position
        assert chunk["duration"] > 0
        assert (
            _primary_video_codec(media_tools["ffprobe"], path) == video_fixture["codec"]
        )
        assert _stream_types(media_tools["ffprobe"], path) == ["video", "audio"]
        assert _container_duration(media_tools["ffprobe"], path) > 0

    assert chunks[0]["start"] == 0.0
    for previous, current in itertools.pairwise(chunks):
        assert current["start"] >= previous["end"] - 0.01
    assert chunks[-1]["end"] == pytest.approx(source_duration, abs=0.5)


def _split_artifacts(chunk_dir):
    """Snapshot split artifacts; the manifest is rewritten on every call."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        for path in chunk_dir.iterdir()
        if path.is_file() and path.name != gemini_subs.MANIFEST_NAME
    }


def test_completed_split_is_reused_unchanged(video_fixture, tmp_path):
    chunk_dir = tmp_path / "work"
    manifest = {"chunk_ext": video_fixture["ext"]}
    gemini_subs.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)
    before = _split_artifacts(chunk_dir)

    gemini_subs.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)

    assert _split_artifacts(chunk_dir) == before


def test_corrupted_chunk_triggers_split_regeneration(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    manifest = {"chunk_ext": video_fixture["ext"]}
    gemini_subs.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)
    original_names = [
        chunk["name"] for chunk in gemini_subs.list_chunks(str(chunk_dir))
    ]
    (chunk_dir / original_names[0]).write_bytes(b"")

    gemini_subs.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)

    marker = chunk_dir / gemini_subs.SPLIT_COMPLETE_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    regenerated = gemini_subs.list_chunks(str(chunk_dir))
    assert [chunk["name"] for chunk in regenerated] == original_names
    for chunk in regenerated:
        path = chunk_dir / chunk["name"]
        assert path.stat().st_size > 0
        assert _container_duration(media_tools["ffprobe"], path) > 0


def test_failed_split_leaves_no_completion_marker(video_fixture, tmp_path):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    (chunk_dir / gemini_subs.SPLIT_COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
    (chunk_dir / "segments.csv").write_text("chunk_000.mp4,0,1\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        gemini_subs.split_video(
            str(tmp_path / "missing-source.mp4"),
            str(chunk_dir),
            2,
            {"chunk_ext": video_fixture["ext"]},
        )

    assert not (chunk_dir / gemini_subs.SPLIT_COMPLETE_MARKER).exists()


def test_generated_media_keeps_audio_and_excludes_source_subtitles(
    h264_video_with_subtitles, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    gemini_subs.split_video(
        str(h264_video_with_subtitles),
        str(chunk_dir),
        2,
        {"chunk_ext": ".mp4"},
    )

    for chunk in gemini_subs.list_chunks(str(chunk_dir)):
        assert _stream_types(media_tools["ffprobe"], chunk_dir / chunk["name"]) == [
            "video",
            "audio",
        ]


def test_segment_index_is_the_source_of_chunk_timing(tmp_path):
    (tmp_path / "segments.csv").write_text(
        "chunk_000.mp4,0.0,2.0\nchunk_001.mp4,2.0,4.5,ignored,fields\n",
        encoding="utf-8",
    )

    assert gemini_subs.list_chunks(str(tmp_path)) == [
        {
            "idx": 0,
            "name": "chunk_000.mp4",
            "start": 0.0,
            "end": 2.0,
            "duration": 2.0,
        },
        {
            "idx": 1,
            "name": "chunk_001.mp4",
            "start": 2.0,
            "end": 4.5,
            "duration": 2.5,
        },
    ]


def test_segment_index_missing_means_no_chunks(tmp_path):
    assert gemini_subs.list_chunks(str(tmp_path)) == []


def test_processing_windows_add_context_and_clamp_at_video_edges():
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0.0, "end": 10.0},
        {"idx": 1, "name": "chunk_001.mp4", "start": 10.0, "end": 18.0},
    ]

    windows = gemini_subs.get_processing_windows(chunks, 2.0)

    assert [(w["clip_start"], w["clip_end"]) for w in windows] == [
        (0.0, 12.0),
        (8.0, 18.0),
    ]
    assert [w["clip_duration"] for w in windows] == [12.0, 10.0]
    assert [(w["owner_start_rel"], w["owner_end_rel"]) for w in windows] == [
        (0.0, 10.0),
        (2.0, 10.0),
    ]
    assert [w["name"] for w in windows] == ["chunk_000.mp4", "chunk_001.mp4"]


def test_processing_windows_without_overlap_match_owner_intervals():
    chunks = [{"idx": 0, "name": "chunk_000.mp4", "start": 5.0, "end": 9.0}]

    windows = gemini_subs.get_processing_windows(chunks, 0)

    assert len(windows) == 1
    assert (windows[0]["clip_start"], windows[0]["clip_end"]) == (5.0, 9.0)
    assert (windows[0]["owner_start_rel"], windows[0]["owner_end_rel"]) == (
        0.0,
        4.0,
    )


def test_processing_windows_without_chunks_is_empty():
    assert gemini_subs.get_processing_windows([], 2.0) == []


def _write_overlap_manifest(chunk_dir, codec):
    (chunk_dir / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps({"video_codec": codec}), encoding="utf-8"
    )


def test_overlap_clip_is_decodable_and_keeps_primary_codec(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    _write_overlap_manifest(chunk_dir, video_fixture["codec"])
    clip_start, clip_end = 1.0, 5.0

    name = gemini_subs.create_overlap_clip(
        str(video_fixture["path"]),
        str(chunk_dir),
        0,
        clip_start,
        clip_end,
        video_fixture["ext"],
    )

    clip = chunk_dir / name
    assert name == f"context_chunk_000{video_fixture['ext']}"
    assert _primary_video_codec(media_tools["ffprobe"], clip) == video_fixture["codec"]
    assert _stream_types(media_tools["ffprobe"], clip) == ["video", "audio"]
    assert _container_duration(media_tools["ffprobe"], clip) == pytest.approx(
        clip_end - clip_start, abs=0.25
    )
    assert not list(chunk_dir.glob("*.tmp"))


def test_valid_overlap_clip_is_reused_without_reencoding(video_fixture, tmp_path):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    _write_overlap_manifest(chunk_dir, video_fixture["codec"])
    first = gemini_subs.create_overlap_clip(
        str(video_fixture["path"]), str(chunk_dir), 0, 1.0, 5.0, video_fixture["ext"]
    )
    clip = chunk_dir / first
    before = (clip.stat().st_size, clip.stat().st_mtime_ns, clip.read_bytes())

    second = gemini_subs.create_overlap_clip(
        str(video_fixture["path"]), str(chunk_dir), 0, 1.0, 5.0, video_fixture["ext"]
    )

    assert second == first
    assert (clip.stat().st_size, clip.stat().st_mtime_ns, clip.read_bytes()) == before


def test_corrupted_overlap_clip_is_regenerated_without_tmp_files(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    _write_overlap_manifest(chunk_dir, video_fixture["codec"])
    name = gemini_subs.create_overlap_clip(
        str(video_fixture["path"]), str(chunk_dir), 2, 1.0, 4.0, video_fixture["ext"]
    )
    clip = chunk_dir / name
    clip.write_bytes(b"corrupted clip")
    (chunk_dir / f"{name}.tmp").write_bytes(b"stale temporary")

    regenerated = gemini_subs.create_overlap_clip(
        str(video_fixture["path"]), str(chunk_dir), 2, 1.0, 4.0, video_fixture["ext"]
    )

    assert regenerated == name
    assert clip.stat().st_size > 0
    assert _container_duration(media_tools["ffprobe"], clip) > 0
    assert not list(chunk_dir.glob("*.tmp"))


def test_attach_with_overlap_builds_a_decodable_context_clip(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    _write_overlap_manifest(chunk_dir, video_fixture["codec"])
    chunk = {
        "idx": 0,
        "name": f"chunk_000{video_fixture['ext']}",
        "start": 1.0,
        "end": 3.0,
        "clip_start": 0.0,
        "clip_end": 4.0,
    }

    result = gemini_subs.attach_overlap_clip(
        str(video_fixture["path"]),
        str(chunk_dir),
        chunk,
        1.0,
        video_fixture["ext"],
    )

    clip_name = result["clip_name"]
    assert clip_name == f"context_chunk_000{video_fixture['ext']}"
    assert _container_duration(media_tools["ffprobe"], chunk_dir / clip_name) > 0


def test_attach_without_overlap_selects_the_stream_copy_chunk(tmp_path):
    chunk = {"idx": 1, "name": "chunk_001.mp4", "start": 0.0, "end": 2.0}

    result = gemini_subs.attach_overlap_clip(
        "source.mp4", str(tmp_path), chunk, 0, ".mp4"
    )

    assert result["clip_name"] == "chunk_001.mp4"
    assert not (tmp_path / "context_chunk_001.mp4").exists()


@pytest.mark.parametrize(
    ("ext", "codec", "message"),
    [
        (".webm", "h264", "H.264 input requires MP4"),
        (".webm", "hevc", "HEVC input requires MP4"),
        (".mp4", "vp9", "VP9 input requires WebM"),
        (".mp4", "unknown-codec", "Overlap format not supported"),
    ],
)
def test_overlap_encoding_rejects_container_or_codec_mismatch(ext, codec, message):
    with pytest.raises(ValueError, match=message):
        gemini_subs.overlap_codec_args(ext, codec, threads=2)
