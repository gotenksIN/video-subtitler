"""Gemini adapters for research, transcription, refinement, and translation."""

import json
from pathlib import Path
from time import monotonic

from google import genai
from google.genai import types
from pydantic import ValidationError

from modules import core, io, media

DEFAULT_FLASH_MODEL = "gemini-3.8-flash"
DEFAULT_PRO_MODEL = "gemini-3.1-pro-preview"
MAX_OUTPUT_TOKENS = 65536
INLINE_AUDIO_WARNING_BYTES = 20 * 1024 * 1024


def create_client(api_key: str | None, base_url: str | None):
    """Create a Gemini client with SDK retries enabled."""
    arguments = {}
    if api_key:
        arguments["api_key"] = api_key
    http_options = {"retry_options": types.HttpRetryOptions()}
    if base_url:
        http_options["base_url"] = base_url
    arguments["http_options"] = http_options
    return genai.Client(**arguments)


def build_content_config(**arguments):
    """Build a direct request config with automatic function calling disabled."""
    arguments["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
        disable=True
    )
    return types.GenerateContentConfig(**arguments)


def _thinking_config():
    return types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True)


def research_config():
    """Build the grounded plain-text research request config."""
    return build_content_config(
        temperature=1.0,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=_thinking_config(),
    )


def _response_schema(model):
    def strip_unsupported(value):
        if isinstance(value, dict):
            return {
                key: strip_unsupported(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [strip_unsupported(item) for item in value]
        return value

    # The endpoint rejects this keyword; local Pydantic validation stays strict.
    return strip_unsupported(model.model_json_schema())


def structured_config(response_model):
    """Build a structured request config for the selected stage response."""
    return build_content_config(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=_response_schema(response_model),
        thinking_config=_thinking_config(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _collect_stream(
    stream, stage: str, model: str, work_dir: str | Path | None = None
) -> tuple[str, list[str], list[core.GroundedSource]]:
    text = ""
    queries = []
    sources = []
    chunks = 0
    thought_chars = 0
    usage = None
    started = last_update = monotonic()
    print(f"{stage} [{model}]: request started", flush=True)
    try:
        for chunk in stream:
            chunks += 1
            usage = getattr(chunk, "usage_metadata", None) or usage
            if chunk.text:
                text += chunk.text
            for candidate in getattr(chunk, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    if getattr(part, "thought", False):
                        thought_chars += len(getattr(part, "text", None) or "")
            now = monotonic()
            if chunks == 1 or now - last_update >= 5:
                print(
                    f"{stage}: {chunks} stream chunks | {len(text)} answer chars | "
                    f"{thought_chars} thought chars | "
                    f"{now - started:.1f}s",
                    flush=True,
                )
                last_update = now
            for candidate in getattr(chunk, "candidates", None) or []:
                reason = getattr(candidate, "finish_reason", None)
                if str(getattr(reason, "value", reason)).upper() == "MAX_TOKENS":
                    raise RuntimeError(
                        "Gemini response exceeded the output token budget"
                    )
                grounding = getattr(candidate, "grounding_metadata", None)
                if not grounding:
                    continue
                queries.extend(
                    query
                    for query in getattr(grounding, "web_search_queries", None) or []
                    if query
                )
                for item in getattr(grounding, "grounding_chunks", None) or []:
                    web = getattr(item, "web", None)
                    url = web and getattr(web, "uri", None)
                    if url:
                        sources.append(
                            core.GroundedSource(
                                title=getattr(web, "title", None) or "", url=url
                            )
                        )
    except Exception:
        if work_dir is not None:
            prefix = Path(work_dir) / stage.lower()
            io.atomic_write_text(prefix.with_suffix(".failed.txt"), text)
            io.atomic_write_text(
                prefix.with_suffix(".failed.json"),
                json.dumps(
                    {
                        "stage": stage,
                        "model": model,
                        "elapsed_seconds": monotonic() - started,
                        "stream_chunks": chunks,
                        "answer_chars": len(text),
                        "thought_chars": thought_chars,
                        "answer_tokens": getattr(usage, "candidates_token_count", None),
                        "thought_tokens": getattr(usage, "thoughts_token_count", None),
                    },
                    indent=2,
                )
                + "\n",
            )
        raise
    finally:
        print(
            f"{stage}: stream closed after {monotonic() - started:.1f}s | "
            f"{chunks} stream chunks | {len(text)} answer chars | "
            f"{thought_chars} thought chars",
            flush=True,
        )
        if usage is not None:
            print(
                f"{stage}: reported tokens: "
                f"answer={getattr(usage, 'candidates_token_count', None)}, "
                f"thought={getattr(usage, 'thoughts_token_count', None)}",
                flush=True,
            )
    unique_queries = list(dict.fromkeys(queries))
    unique_sources = list({source.url: source for source in sources}.values())
    return text, unique_queries, unique_sources


def _audio_part(path: str | Path):
    payload = Path(path).read_bytes()
    if len(payload) > INLINE_AUDIO_WARNING_BYTES:
        print(
            f"Warning: normalized audio is {len(payload) / 1024 / 1024:.1f} MiB; "
            "the inline request may exceed Gemini limits."
        )
    return types.Part.from_bytes(data=payload, mime_type=media.AUDIO_MIME_TYPE)


def research_song_context(
    api_key: str,
    base_url: str | None,
    model: str,
    title: str,
    artist: str | None,
) -> core.SongContext:
    """Research a song and artist and reject ungrounded output."""
    artist_line = artist or "Infer the artist only from reliable search evidence."
    prompt = f"""Research context for this song before an audio transcription.

Song title or source label: {title}
Artist: {artist_line}

Use Google Search. Return concise plain text that identifies the song and artist when evidence supports it. Include official names, language information, release context, and spellings of names or terms likely to occur in the lyrics. Distinguish verified facts from uncertainty. Do not provide, quote, reconstruct, or search for the song lyrics. Do not use lyric websites as transcription evidence.
"""
    with create_client(api_key, base_url) as client:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=research_config(),
        )
        text, queries, sources = _collect_stream(stream, "Research", model)
    if not text.strip():
        raise RuntimeError("Grounded song research returned no context")
    if not queries and not sources:
        raise RuntimeError("Song research returned no Google Search grounding")
    return core.SongContext(
        summary=text.strip(), search_queries=queries, sources=sources
    )


def _parse_lyrics(raw: str, duration: float, stage: str, work_dir: str | Path):
    try:
        response = core.LyricsResponse.model_validate_json(raw)
        return core.validate_lyrics(response.lines, duration)
    except (ValidationError, ValueError) as error:
        io.atomic_write_text(Path(work_dir) / f"{stage}.failed.txt", raw)
        raise RuntimeError(f"{stage} response failed validation: {error}") from error


def transcribe_song(
    api_key: str,
    base_url: str | None,
    model: str,
    audio_path: str | Path,
    duration: float,
    context: core.SongContext,
    work_dir: str | Path,
) -> list[core.LyricLine]:
    """Transcribe complete source-language lyrics from the whole song audio."""
    prompt = f"""Transcribe the sung lyrics in the attached complete song audio.

The audio is authoritative. Use this grounded context only for proper names and terminology:
{context.summary}

Return the actual source-language lyrics. Preserve every language that is sung, including mixed-language lines. Preserve repetitions, ad-libs, interjections, and incomplete phrases that are audible. Do not translate, romanize, summarize, censor, or infer words from known published lyrics. Ignore instrumental and silent passages.

Create line-level timing across the whole song. Start each line at its first audible sung syllable and end it at its last audible syllable. Keep real gaps. Use positive intervals within {duration:.3f} seconds. Segment lines by musical and lyrical phrasing. Return every line in chronological order with unique sequential IDs L1, L2, and so on. Return only the required JSON object.
"""
    with create_client(api_key, base_url) as client:
        stream = client.models.generate_content_stream(
            model=model,
            contents=[_audio_part(audio_path), prompt],
            config=structured_config(core.LyricsResponse),
        )
        raw, _, _ = _collect_stream(stream, "Transcription", model, work_dir)
    return _parse_lyrics(raw, duration, "transcription", work_dir)


def refine_song(
    api_key: str,
    base_url: str | None,
    model: str,
    audio_path: str | Path,
    duration: float,
    context: core.SongContext,
    draft: list[core.LyricLine],
    work_dir: str | Path,
) -> list[core.LyricLine]:
    """Correct existing draft lines against the whole song audio."""
    draft_json = core.LyricsResponse(lines=draft).model_dump_json(indent=2)
    prompt = f"""Proofread the draft source-language lyrics against the attached complete song audio.

Treat every draft line as correct unless the audio supports a specific correction. Fix misheard words, spelling, and incorrect start or end timestamps. Preserve the language actually sung, repetitions, and ad-libs. Keep accurate wording and timing exactly as supplied. Do not translate, romanize, or reconstruct published lyrics.

Use this grounded song and artist context only for verified names and terminology:
{context.summary}

Draft transcription:
{draft_json}

Return a JSON object with a "changes" list. Include only existing line IDs that need correction and only the text, start, or end fields that change. Omit unchanged fields or set them to null. Use the complete corrected line text when changing text. Return {{"changes": []}} when no correction is needed.

Keep every existing line and its ID. Do not add, delete, merge, split, or reorder lines. Corrected times must keep chronological line order and positive intervals within {duration:.3f} seconds. Preserve real gaps. Return only the required JSON object.
"""
    with create_client(api_key, base_url) as client:
        stream = client.models.generate_content_stream(
            model=model,
            contents=[_audio_part(audio_path), prompt],
            config=structured_config(core.RefinementResponse),
        )
        raw, _, _ = _collect_stream(stream, "Refinement", model, work_dir)
    try:
        response = core.RefinementResponse.model_validate_json(raw)
        return core.apply_corrections(draft, response.changes, duration)
    except (ValidationError, ValueError) as error:
        io.atomic_write_text(Path(work_dir) / "refinement.failed.txt", raw)
        raise RuntimeError(f"refinement response failed validation: {error}") from error


def translate_song(
    api_key: str,
    base_url: str | None,
    model: str,
    target_language: str,
    context: core.SongContext,
    source_lines: list[core.LyricLine],
    work_dir: str | Path,
) -> list[core.TranslationLine]:
    """Translate refined lyrics while preserving exact source-line identity."""
    source_json = core.LyricsResponse(lines=source_lines).model_dump_json(indent=2)
    prompt = f"""Translate these final source lyrics into {target_language}.

Use the grounded context for song, artist, proper names, and terminology:
{context.summary}

Preserve meaning, repetition, ad-libs, tone, and line segmentation. Translate each mixed-language source line as one complete line. Do not alter, omit, merge, split, or reorder source lines. Return one nonblank translation for every source ID in the same order. Return only IDs and translated text. Do not return timing.

Source lyrics:
{source_json}
"""
    with create_client(api_key, base_url) as client:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=structured_config(core.TranslationResponse),
        )
        raw, _, _ = _collect_stream(stream, "Translation", model, work_dir)
    try:
        response = core.TranslationResponse.model_validate_json(raw)
        return core.validate_translation(response, source_lines)
    except (ValidationError, ValueError) as error:
        io.atomic_write_text(Path(work_dir) / "translation.failed.txt", raw)
        raise RuntimeError(
            f"translation response failed validation: {error}"
        ) from error
