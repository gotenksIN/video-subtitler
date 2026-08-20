"""Scenario builders for pipeline work-directory tests."""

import json
import subprocess
from pathlib import Path

from modules import io


def make_manifest(codec="h264"):
    ext = ".webm" if codec == "vp9" else ".mp4"
    mime = "video/webm" if ext == ".webm" else "video/mp4"
    return {
        "mode": "generate",
        "chunk_ext": ext,
        "chunk_mime": mime,
        "video_codec": codec,
    }


def write_chunk_layout(directory, rows):
    Path(directory, io.MANIFEST_NAME).write_text(
        json.dumps(make_manifest()), encoding="utf-8"
    )
    Path(directory, "segments.csv").write_text(
        "".join(f"{name},{start},{end}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def write_chunk_subtitles(directory, index, captions):
    Path(directory, f"subtitle_chunk_{index:03d}.json").write_text(
        json.dumps(captions), encoding="utf-8"
    )


class FakeMediaTools:
    """Scripted FFmpeg and FFprobe replacement for pipeline scenarios.

    Split commands materialize the configured chunk files and segments.csv.
    Audio extraction commands materialize the complete extracted audio file.
    FFprobe duration checks answer with a configurable duration and status.
    FFprobe audio-stream checks answer with configurable stream metadata.
    Every invocation is recorded in :attr:`calls` for outcome assertions.
    """

    def __init__(
        self,
        chunk_rows=((0, 2), (2, 4)),
        chunk_bytes=b"fake chunk",
        audio_bytes=b"fake audio",
        audio_streams=({"codec_name": "opus", "sample_rate": "48000", "channels": 1},),
        probe_duration="4.0\n",
        probe_ok=True,
    ):
        self.chunk_rows = chunk_rows
        self.chunk_bytes = chunk_bytes
        self.audio_bytes = audio_bytes
        self.audio_streams = audio_streams
        self.probe_duration = probe_duration
        self.probe_ok = probe_ok
        self.calls = []

    def run(self, command, **_kwargs):
        self.calls.append(list(command))
        if command[0] == "ffprobe":
            return self._probe(command)
        if command[0] != "ffmpeg":
            raise AssertionError(f"unexpected subprocess command: {command}")
        if "-segment_list" in command:
            self._materialize_split(command)
        elif "libopus" in command:
            self._materialize_audio(command)
        else:
            raise AssertionError(f"unexpected ffmpeg command: {command}")
        return subprocess.CompletedProcess(command, 0)

    def split_calls(self):
        return [call for call in self.calls if "-segment_list" in call]

    def audio_extraction_calls(self):
        return [
            call for call in self.calls if call[0] == "ffmpeg" and "libopus" in call
        ]

    def _probe(self, command):
        if "-select_streams" in command:
            stdout = json.dumps({"streams": list(self.audio_streams)})
        else:
            stdout = self.probe_duration
        return subprocess.CompletedProcess(
            command,
            0 if self.probe_ok else 1,
            stdout=stdout if self.probe_ok else "",
            stderr="",
        )

    def _materialize_split(self, command):
        segments = Path(command[command.index("-segment_list") + 1])
        template = Path(command[-1])
        directory = template.parent
        lines = []
        for index, (start, end) in enumerate(self.chunk_rows):
            name = f"chunk_{index:03d}{template.suffix}"
            Path(directory, name).write_bytes(self.chunk_bytes)
            lines.append(f"{name},{start},{end}\n")
        segments.write_text("".join(lines), encoding="utf-8")

    def _materialize_audio(self, command):
        output = Path(command[-1])
        contents = (
            self.audio_bytes(output) if callable(self.audio_bytes) else self.audio_bytes
        )
        output.write_bytes(contents)
