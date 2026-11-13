"""Reusable stateful fakes for the Gemini SDK streaming boundary.

These fakes script streaming responses and record every request the
adapter sends to the SDK. Tests assert on the recorded request state and
on real files, not on internal call sequences or prompt wording.
"""

from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace

import webvtt

from modules import gemini


@dataclass(frozen=True)
class RecordedRequest:
    """One request the adapter sent to the SDK boundary."""

    model: str
    contents: object
    config: object


@dataclass
class StreamCall:
    """One scripted streaming response."""

    pieces: list = field(default_factory=list)
    error: Exception | None = None
    kind: str | None = None


class ScriptedGeminiClient:
    """Stateful fake for genai.Client used as a context manager.

    Consumes calls in order within each request kind. Records each request.
    An unscripted request raises AssertionError, so tests can assert that
    no extra API request was issued.
    """

    def __init__(self, calls=()):
        self._pending = list(calls)
        self.requests = []
        self._lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    def generate_content_stream(self, **kwargs):
        contents = kwargs["contents"]
        kind = None
        if kwargs["config"].tools:
            kind = "research"
        elif isinstance(contents, list) and any(
            getattr(part, "file_data", None) and part.file_data.mime_type == "video/*"
            for part in contents
        ):
            kind = "youtube"
        with self._lock:
            index = next(
                (
                    i
                    for i, call in enumerate(self._pending)
                    if call.kind in (None, kind)
                ),
                None,
            )
            if index is None:
                raise AssertionError("the pipeline issued an unexpected Gemini request")
            call = self._pending.pop(index)
            self.requests.append(
                RecordedRequest(
                    model=kwargs["model"],
                    contents=contents,
                    config=kwargs["config"],
                )
            )
        if call.error is not None:
            raise call.error
        return self._stream(call)

    @staticmethod
    def _stream(call):
        for piece in call.pieces:
            if isinstance(piece, SimpleNamespace):
                yield piece
            else:
                yield SimpleNamespace(text=piece)


def stream_chunk(text=None, finish_reason=None, thought=False):
    """One SDK stream chunk with optional candidate finish metadata."""
    return SimpleNamespace(
        text=None if thought else text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


def chunk_call(pieces, error=None):
    """One scripted chunk-generation response."""
    return StreamCall(pieces=list(pieces), error=error)


def research_call(pieces=("Research text",), error=None):
    """One scripted grounded identity research response."""
    return StreamCall(pieces=list(pieces), error=error, kind="research")


def youtube_call(pieces=("YouTube analysis",), error=None):
    """One scripted direct YouTube analysis response."""
    return StreamCall(pieces=list(pieces), error=error, kind="youtube")


def refinement_call(pieces, error=None):
    """One scripted structured refinement response."""
    return StreamCall(pieces=list(pieces), error=error)


def audio_call(pieces, error=None):
    """One scripted audio-refinement response."""
    return StreamCall(pieces=list(pieces), error=error)


def use_client(monkeypatch, client):
    """Route every create_client call in modules.gemini to one scripted fake."""
    monkeypatch.setattr(gemini, "create_client", lambda *args, **kwargs: client)


def write_vtt(path, captions):
    """Write a real WebVTT file with (start, end, text) caption tuples."""
    value = webvtt.WebVTT()
    value.captions.extend(
        webvtt.Caption(start, end, text) for start, end, text in captions
    )
    value.save(str(path))
    return path


def read_captions(path):
    """Return (start, end, text) tuples from a real WebVTT file."""
    vtt = webvtt.read(str(path))
    return [(caption.start, caption.end, caption.text) for caption in vtt]
