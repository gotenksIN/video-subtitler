"""Fake Gemini SDK boundaries shared by pipeline tests."""

from types import SimpleNamespace

from google.genai import types

import gemini_subs


class StreamPiece(SimpleNamespace):
    def __init__(self, text, candidates=None):
        if candidates is None:
            candidates = []
        elif not isinstance(candidates, list):
            candidates = [candidates]
        super().__init__(text=text, candidates=candidates)


def grounded_candidate(queries=(), sources=(), urls=()):
    """Build one stream candidate carrying grounding and retrieval metadata."""
    grounding = None
    if queries or sources:
        grounding = types.GroundingMetadata(
            web_search_queries=list(queries),
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(title=title, uri=uri))
                for title, uri in sources
            ],
        )
    url_context = None
    if urls:
        url_context = types.UrlContextMetadata(
            url_metadata=[
                types.UrlMetadata(retrieved_url=url, url_retrieval_status=status)
                for url, status in urls
            ]
        )
    return types.Candidate(
        grounding_metadata=grounding, url_context_metadata=url_context
    )


class ScenarioGeminiClient:
    """One fake SDK boundary for chunk generation and refinement requests.

    Chunk requests use the SubtitleResponse schema and the chunk responder.
    Research requests must be plain text with the Google Search tool.
    YouTube requests must be plain text without tools. Refinement requests
    must be structured with the RefinementResponse schema and no tools.
    """

    def __init__(
        self,
        chunk_responder,
        research_spec,
        refinement_spec,
        youtube_spec=None,
    ):
        self.chunk_responder = chunk_responder
        self.research_spec = research_spec
        self.refinement_spec = refinement_spec
        self.youtube_spec = youtube_spec
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    def generate_content_stream(self, **kwargs):
        self.requests.append(kwargs)
        config = kwargs["config"]
        schema = config.response_schema
        if schema is gemini_subs.SubtitleResponse:
            pieces = list(self.chunk_responder(kwargs["contents"]))
            schema.model_validate_json("".join(pieces))
            return iter(StreamPiece(piece) for piece in pieces)
        if schema is gemini_subs.RefinementResponse:
            if config.tools:
                raise AssertionError("structured refinement must not enable tools")
            pieces = list(self.refinement_spec["pieces"])
            schema.model_validate_json("".join(pieces))
            return self._stream(self.refinement_spec, pieces)
        tools = config.tools or []
        if any(tool.google_search is not None for tool in tools):
            if schema is not None or config.response_mime_type is not None:
                raise AssertionError("research request must use plain text")
            return self._stream(self.research_spec)
        if self.youtube_spec is None:
            raise AssertionError("unexpected plain request without Google Search")
        return self._stream(self.youtube_spec)

    def _stream(self, spec, pieces=None):
        if pieces is None:
            pieces = list(spec["pieces"])
        candidates = spec.get("candidates", [])
        return iter(
            StreamPiece(piece, candidates[i] if i < len(candidates) else None)
            for i, piece in enumerate(pieces)
        )
