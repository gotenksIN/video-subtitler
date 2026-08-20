"""Reusable stateful fakes for the Gemini SDK streaming boundary.

These fakes script streaming responses and record every request the
adapter sends to the SDK. Tests assert on the recorded request state and
on real files, not on internal call sequences or prompt wording.
"""

from dataclasses import dataclass, field
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


class ScriptedGeminiClient:
    """Stateful fake for genai.Client used as a context manager.

    Consumes scripted calls in order. Records each request in `requests`.
    An unscripted request raises AssertionError, so tests can assert that
    no extra API request was issued.
    """

    def __init__(self, calls=()):
        self._pending = list(calls)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    def generate_content_stream(self, **kwargs):
        if not self._pending:
            raise AssertionError("the pipeline issued an unexpected Gemini request")
        call = self._pending.pop(0)
        self.requests.append(
            RecordedRequest(
                model=kwargs["model"],
                contents=kwargs["contents"],
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


def stream_chunk(text=None, finish_reason=None):
    """One SDK stream chunk with optional candidate finish metadata."""
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


def chunk_call(pieces, error=None):
    """One scripted chunk-generation response."""
    return StreamCall(pieces=list(pieces), error=error)


def research_call(pieces=("Research text",), error=None):
    """One scripted grounded identity research response."""
    return StreamCall(pieces=list(pieces), error=error)


def youtube_call(pieces=("YouTube analysis",), error=None):
    """One scripted direct YouTube analysis response."""
    return StreamCall(pieces=list(pieces), error=error)


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
