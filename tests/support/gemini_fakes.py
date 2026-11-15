"""Stateful Gemini scenarios through the real SDK and an HTTP transport seam."""

import json
from dataclasses import dataclass, field
from threading import Lock, local

import httpx2
from google import genai
from google.genai import errors, types

from modules import gemini

_REAL_CLIENT = genai.Client


@dataclass(frozen=True)
class RecordedRequest:
    model: str
    contents: object
    config: types.GenerateContentConfig
    kind: str
    url: str
    headers: httpx2.Headers
    body: dict
    http_options: types.HttpOptions


@dataclass
class StreamCall:
    kind: str
    pieces: list = field(default_factory=list)
    error: Exception | None = None
    on_request: object = None


class ScriptedGeminiClient:
    """Drive real SDK clients with scripted HTTP and SSE responses."""

    def __init__(self, calls=()):
        self.pending = list(calls)
        self.requests = []
        self._lock = Lock()

    @staticmethod
    def _kind(config):
        if config.tools:
            kind = "research"
        elif config.response_schema:
            properties = config.response_schema["properties"]
            kind = next(
                name
                for key, name in (
                    ("captions", "chunk"),
                    ("cues", "audio"),
                    ("changes", "text"),
                )
                if key in properties
            )
        else:
            kind = "youtube"
        return kind

    def create_client(self, **kwargs):
        active = local()
        options = dict(kwargs.get("http_options") or {})

        def handle(wire_request):
            model, contents, config = active.request
            request = RecordedRequest(
                model,
                contents,
                config,
                self._kind(config),
                str(wire_request.url),
                wire_request.headers,
                json.loads(wire_request.content),
                types.HttpOptions(**options),
            )
            return self._respond(request)

        options["httpx_client"] = httpx2.Client(transport=httpx2.MockTransport(handle))
        client = _REAL_CLIENT(**{**kwargs, "http_options": options})
        generate = client.models.generate_content_stream

        def generate_content_stream(*, model, contents, config):
            active.request = (model, contents, config)
            try:
                yield from generate(model=model, contents=contents, config=config)
            finally:
                del active.request

        client.models.generate_content_stream = generate_content_stream
        return client

    def _respond(self, request):
        with self._lock:
            self.requests.append(request)
            index = next(
                (i for i, call in enumerate(self.pending) if call.kind == request.kind),
                None,
            )
            if index is None:
                raise AssertionError(f"Unexpected Gemini {request.kind} request")
            call = self.pending.pop(index)
        if call.on_request:
            call.on_request(request)
        if call.error:
            if isinstance(call.error, errors.APIError):
                return httpx2.Response(
                    call.error.code,
                    json={
                        "error": {
                            "code": call.error.code,
                            "message": call.error.message,
                        }
                    },
                )
            raise call.error
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_EventStream(call.pieces),
        )


class _EventStream(httpx2.SyncByteStream):
    def __init__(self, pieces):
        self.pieces = pieces

    def __iter__(self):
        for piece in self.pieces:
            if isinstance(piece, errors.APIError):
                data = {"error": {"code": piece.code, "message": piece.message}}
            elif isinstance(piece, Exception):
                raise piece
            else:
                chunk = stream_chunk(piece) if isinstance(piece, str) else piece
                data = chunk.model_dump(mode="json", by_alias=True, exclude_none=True)
            yield f"data: {json.dumps(data)}\n\n".encode()


def stream_chunk(
    text=None,
    *,
    thought=False,
    finish_reason=None,
    continuation_token=None,
    parts=None,
    grounding=False,
    urls=(),
):
    """Build SDK chunks without testing SDK representation details."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=parts or [types.Part(text=text, thought=thought)]
                ),
                finish_reason=finish_reason,
                continuation_token=continuation_token,
                grounding_metadata=types.GroundingMetadata(
                    web_search_queries=["speaker identities"]
                )
                if grounding
                else None,
                url_context_metadata=types.UrlContextMetadata(
                    url_metadata=[
                        types.UrlMetadata(
                            retrieved_url=url,
                            url_retrieval_status=status,
                        )
                        for url, status in urls
                    ]
                )
                if urls
                else None,
            )
        ]
    )


def research_call(*, grounding=True, urls=(), **kwargs):
    return StreamCall(
        "research",
        [
            stream_chunk(
                "PARTICIPANTS AND SPEAKERS\nHost: Presenter\n"
                "TOPIC TERMINOLOGY AND PROPER NOUNS\nOpus: Audio codec",
                grounding=grounding,
                urls=urls,
            )
        ],
        **kwargs,
    )


def chunk_call(text="HOST: Hello.", *, banner_end=None, **kwargs):
    captions = [{"id": 0, "start": "0.1", "end": "0.5", "text": text}]
    if banner_end is not None:
        captions.append(
            {"id": 1, "start": "0", "end": str(banner_end), "text": "[Title]"}
        )
    return StreamCall(
        "chunk",
        [
            stream_chunk("not JSON", thought=True),
            json.dumps({"captions": captions}),
        ],
        **kwargs,
    )


def audio_call(cues=(), deleted=(), **kwargs):
    return StreamCall(
        "audio",
        [
            json.dumps(
                {
                    "contractVersion": "sparse-patch-v1",
                    "deletedSourceIds": list(deleted),
                    "cues": list(cues),
                }
            )
        ],
        **kwargs,
    )


def text_call(changes=(), **kwargs):
    return StreamCall("text", [json.dumps({"changes": list(changes)})], **kwargs)


def use_client(monkeypatch, client):
    monkeypatch.setattr(gemini.genai, "Client", client.create_client)
