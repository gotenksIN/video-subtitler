"""Scenario builders for pipeline work-directory tests."""

import json
import subprocess
from pathlib import Path

import webvtt

import gemini_subs


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


def write_chunk_layout(directory, rows, overlap=0.0):
    Path(directory, gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(overlap)), encoding="utf-8"
    )
    Path(directory, "segments.csv").write_text(
        "".join(f"{name},{start},{end}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def write_chunk_subtitles(directory, index, captions):
    Path(directory, f"subtitle_chunk_{index:03d}.json").write_text(
        json.dumps(captions), encoding="utf-8"
    )


def write_vtt_file(path, captions):
    value = webvtt.WebVTT()
    value.captions.extend(
        webvtt.Caption(start, end, text) for start, end, text in captions
    )
    value.save(path)


class FakeMediaTools:
    """Scripted FFmpeg and FFprobe replacement for pipeline scenarios.

    Split commands materialize the configured chunk files and segments.csv.
    Overlap encode commands materialize the requested context clip.
    FFprobe duration checks answer with a configurable duration and status.
    Every invocation is recorded in :attr:`calls` for outcome assertions.
    """

    def __init__(
        self,
        chunk_rows=((0, 2), (2, 4)),
        chunk_bytes=b"fake chunk",
        clip_bytes=b"fake clip",
        probe_duration="2.0\n",
        probe_ok=True,
    ):
        self.chunk_rows = chunk_rows
        self.chunk_bytes = chunk_bytes
        self.clip_bytes = clip_bytes
        self.probe_duration = probe_duration
        self.probe_ok = probe_ok
        self.calls = []

    def run(self, command, **_kwargs):
        self.calls.append(list(command))
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0 if self.probe_ok else 1,
                stdout=self.probe_duration if self.probe_ok else "",
                stderr="",
            )
        if command[0] != "ffmpeg":
            raise AssertionError(f"unexpected subprocess command: {command}")
        if "-segment_list" in command:
            self._materialize_split(command)
        elif "-ss" in command:
            self._materialize_clip(command)
        else:
            raise AssertionError(f"unexpected ffmpeg command: {command}")
        return subprocess.CompletedProcess(command, 0)

    def split_calls(self):
        return [call for call in self.calls if "-segment_list" in call]

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

    def _materialize_clip(self, command):
        output = Path(command[-1])
        contents = (
            self.clip_bytes(output) if callable(self.clip_bytes) else self.clip_bytes
        )
        output.write_bytes(contents)
