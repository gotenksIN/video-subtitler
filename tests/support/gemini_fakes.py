"""Reusable stateful fakes for the Gemini SDK streaming boundary.

These fakes script streaming responses and record every request the
adapter sends to the SDK. Tests assert on the recorded request state and
on real files, not on internal call sequences or prompt wording.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

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


def use_client(monkeypatch, client):
    """Route every create_client call in modules.gemini to one scripted fake."""
    monkeypatch.setattr(gemini, "create_client", lambda *args, **kwargs: client)


def grounded_chunk(text="Verified context"):
    """One response chunk with observable Google Search grounding."""
    web = SimpleNamespace(title="Official source", uri="https://example.com/song")
    metadata = SimpleNamespace(
        web_search_queries=["song artist"],
        grounding_chunks=[SimpleNamespace(web=web)],
    )
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=None, grounding_metadata=metadata)],
    )
