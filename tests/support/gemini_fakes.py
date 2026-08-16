"""Reusable stateful fakes for the Gemini SDK streaming boundary.

These fakes script streaming responses and record every request the
adapter sends to the SDK. Tests assert on the recorded request state and
on real files, not on internal call sequences or prompt wording.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import webvtt
from google.genai import types

from modules import gemini


@dataclass(frozen=True)
class RecordedRequest:
    """One request the adapter sent to the SDK boundary."""

    model: str
    contents: object
    config: object


@dataclass
class StreamCall:
    """One scripted streaming response with optional candidate metadata."""

    pieces: list = field(default_factory=list)
    candidate_chunks: list = field(default_factory=list)
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

    @property
    def pending(self):
        return len(self._pending)

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
        for index, piece in enumerate(call.pieces):
            candidates = (
                call.candidate_chunks[index]
                if index < len(call.candidate_chunks)
                else []
            )
            yield SimpleNamespace(text=piece, candidates=list(candidates))


def grounding_candidate(queries=(), sources=(), retrieved=()):
    """Build one candidate carrying search and URL retrieval metadata."""
    grounding = types.GroundingMetadata(
        web_search_queries=list(queries),
        grounding_chunks=[
            types.GroundingChunk(web=types.GroundingChunkWeb(title=title, uri=uri))
            for title, uri in sources
        ],
    )
    url_context = types.UrlContextMetadata(
        url_metadata=[
            types.UrlMetadata(retrieved_url=url, url_retrieval_status=status)
            for url, status in retrieved
        ]
    )
    return [
        types.Candidate(grounding_metadata=grounding, url_context_metadata=url_context)
    ]


def chunk_call(pieces, error=None):
    """One scripted chunk-generation response."""
    return StreamCall(pieces=list(pieces), error=error)


def research_call(
    pieces=("Research text",),
    queries=(),
    sources=(),
    retrieved=(),
    error=None,
):
    """One scripted grounded identity research response."""
    candidates = (
        grounding_candidate(queries, sources, retrieved)
        if queries or sources or retrieved
        else []
    )
    return StreamCall(pieces=list(pieces), candidate_chunks=[candidates], error=error)


def youtube_call(pieces=("YouTube analysis",), error=None):
    """One scripted direct YouTube analysis response."""
    return StreamCall(pieces=list(pieces), error=error)


def refinement_call(pieces, error=None):
    """One scripted structured refinement response."""
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
