"""Gemini clients, prompts, configs, chunk requests, and audio/text refinement."""

import hashlib
import json
import os
import time
from pathlib import Path

import webvtt
from google import genai
from google.genai import types
from pydantic import ValidationError

from modules import core, io, media

INLINE_VIDEO_WARNING_BYTES = 20 * 1024 * 1024
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_CHUNK_MODEL = "gemini-3.7-flash"
DEFAULT_REFINE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CHUNK_THINKING_LEVEL = "high"
REFINEMENT_THINKING_LEVEL = "medium"
DEFAULT_AUDIO_REFINE_MODEL = "gemini-3.7-flash"
AUDIO_REFINE_THINKING_LEVEL = "high"
AUDIO_REFINE_MAX_OUTPUT_TOKENS = 65536
AUDIO_REFINE_RESPONSE_CONTRACT = core.AUDIO_REFINE_RESPONSE_CONTRACT
STREAM_REQUEST_ATTEMPTS = 3


def validate_thinking_level_for_model(model_name, thinking_level):
    """Validate thinking level against model capabilities."""
    if thinking_level == "minimal" and "flash" not in model_name.lower():
        raise ValueError(
            "--thinking-level minimal is only supported by Flash models. Use low, medium, or high for this model."
        )


def create_client(api_key, base_url):
    """Create a Gemini API client instance with configured credentials and retries."""
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    http_options = {"retry_options": types.HttpRetryOptions()}
    if base_url:
        http_options["base_url"] = base_url
    kwargs["http_options"] = http_options
    return genai.Client(**kwargs)


def build_content_config(**kwargs):
    """Build a config with AFC disabled for direct Models requests."""
    kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
        disable=True
    )
    return types.GenerateContentConfig(**kwargs)


def build_thinking_config(thinking_level):
    """Build a thinking configuration with thought streaming enabled."""
    if thinking_level is None:
        return None
    return types.ThinkingConfig(
        thinking_level=thinking_level.upper(),
        include_thoughts=True,
    )


def generate_content_config(thinking_level):
    """Build the response configuration for chunk video generation."""
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": core.SubtitleResponse,
    }
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        kwargs["thinking_config"] = thinking_config
    return build_content_config(**kwargs)


def build_generation_prompt(chunk_duration, source_title=None):
    """Build the prompt for per-chunk video subtitle generation."""
    source_block = ""
    if source_title:
        source_block = (
            "SOURCE CONTEXT\n\n"
            f"Source title: {source_title}\n"
            "Names in the source title are candidate identities only. "
            "They do not prove which speaker said a specific line.\n\n"
        )
    return f"""You are an expert subtitle generator and translator.

Watch this {chunk_duration:.3f}-second video clip.

Generate accurate, natural English subtitles for dialogue and meaningful on-screen text throughout the entire clip.

{source_block}TIMING

1. Create timestamps relative to the beginning of the clip, ranging from 00:00:00.000 to {core.format_time(chunk_duration)}.
2. For spoken dialogue, start at the exact first audible syllable and end at the exact end of the last audible syllable.
3. For on-screen text, start when the text becomes visible and end when it disappears.
4. Preserve real silent gaps. Do not stretch captions through silence, reaction shots, or scene changes.
5. Keep captions sorted by start time and do not overlap them.
6. Avoid cues shorter than 500 milliseconds. If a meaningful short utterance cannot fit naturally, combine it with an adjacent utterance from the same speaker only when doing so preserves meaning and timing.

TRANSLATION

7. Translate all spoken dialogue and meaningful on-screen text from the source language into natural English. Never return a source-language transcription instead of an English translation.
8. Prefer faithful, clear English over punchy paraphrases.
9. Preserve every meaningful question, answer, joke, reaction, and product detail. Do not summarize or omit meaningful content.
10. Do not infer missing dialogue or invent facts, product claims, jokes, or cultural explanations.
11. Preserve established names, brands, foods, products, titles, and recurring terms consistently.
12. Preserve useful source-language cultural terms when they express a relationship or concept that English cannot express as precisely.
13. Do not replace understandable English with unexplained romanized source-language terms.
14. Transliterate uncertain proper nouns conservatively instead of inventing a nickname, joke, or English equivalent.
15. Preserve wordplay naturally in English whenever possible. Do not silently replace a pun with unrelated dialogue.

SPEAKER LABELS

16. Use a person's name only when the clip itself establishes attribution: a visible name label or title card, a spoken introduction, or other direct in-clip evidence.
17. Never identify a speaker from appearance alone.
18. When a name cannot be established, prefer a stable descriptive role such as "Host:", "Resident:", "Shop Owner:", or "Producer:" when the role is clear from the clip.
19. Leave dialogue unlabeled when neither a name nor a stable role can be distinguished.
20. Do not use generic numbered labels such as "Speaker 1:".
21. Use the exact format "Name: Dialogue".
22. When multiple identifiable speakers share a cue, place each attributed turn on a separate line.
23. Do not assign speaker labels to on-screen text.

ON-SCREEN TEXT

24. Include meaningful on-screen editorial text when it contributes information, context, humor, branding, or narrative meaning.
25. Ignore decorative text, logos, persistent watermarks, repeated UI, and text unrelated to understanding the video.
26. Keep on-screen text distinct from spoken dialogue.
27. Render on-screen text in square brackets, without mechanical prefixes such as "On-screen text:".
28. Do not combine unrelated dialogue and on-screen text in one caption.
29. Do not describe visible actions such as "(walks)", "(rings bell)", or "(sprays product)" unless corresponding written editorial text actually appears in the video.
30. Translate source-language editorial idioms and visual-caption metaphors into understandable English rather than preserving an incomprehensible literal translation.
31. Do not wrap ordinary spoken dialogue in quotation marks.

FORMATTING

32. Use sequential integer IDs starting at 0.
33. Follow standard subtitle readability rules: no more than 42 characters per line and no more than two lines per caption.
34. Split long speech into readable, natural phrases without changing meaning.
35. Do not use markdown or include explanations outside subtitle captions.
36. Return only a valid JSON object matching the required schema with a "captions" array.
"""


def load_cached_captions(out_json, chunk_duration):
    """Load and validate cached captions from a chunk JSON file."""
    if not os.path.exists(out_json):
        return None
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        response = core.SubtitleResponse(captions=data)
        return core.validate_captions(response.captions, chunk_duration)
    except Exception as e:  # noqa: BLE001 - Invalid cache data must be regenerated.
        print(f"Ignoring invalid cached output {out_json}: {e}")
        os.remove(out_json)
        return None


def process_chunk(
    api_key,
    base_url,
    chunk,
    chunk_dir,
    model_name,
    chunk_mime,
    thinking_level,
    source_title=None,
):
    """Process one video chunk and publish validated caption JSON."""
    chunk_idx = chunk["idx"]
    chunk_name = chunk["name"]
    chunk_duration = chunk["duration"]
    out_json = os.path.join(chunk_dir, f"subtitle_chunk_{chunk_idx:03d}.json")
    chunk_path = os.path.join(chunk_dir, chunk_name)

    cached = load_cached_captions(out_json, chunk_duration)
    if cached is not None:
        print(f"Skipping {chunk_name} - already processed.")
        return True

    prompt = build_generation_prompt(chunk_duration, source_title)

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()
        if len(video_data) > INLINE_VIDEO_WARNING_BYTES:
            print(
                f"[Worker-{chunk_idx:03d}] Warning: {chunk_name} is {len(video_data) / 1024 / 1024:.1f} MB. "
                "Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail."
            )

        print(f"[Worker-{chunk_idx:03d}] Generating {chunk_name} using Gemini API...")

        for attempt in range(STREAM_REQUEST_ATTEMPTS):
            with create_client(api_key, base_url) as client:
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=video_data, mime_type=chunk_mime),
                        prompt,
                    ],
                    config=generate_content_config(thinking_level),
                )
                full_json_text = ""
                for response_chunk in response_stream:
                    if response_chunk.text:
                        full_json_text += response_chunk.text

            # Parsing and caption validation raise ValueError subclasses,
            # so every ValueError here marks an unusable model response.
            try:
                parsed_response = core.SubtitleResponse.model_validate_json(
                    full_json_text
                )
                validated = core.validate_captions(
                    parsed_response.captions, chunk_duration
                )
            except ValueError as error:
                if attempt == STREAM_REQUEST_ATTEMPTS - 1:
                    print(
                        f"[Worker-{chunk_idx:03d}] ERROR processing {chunk_name}: "
                        f"{error}"
                    )
                    return False
                delay = 2**attempt
                print(
                    f"[Worker-{chunk_idx:03d}] Warning: Attempt {attempt + 1} failed "
                    f"for {chunk_name} ({error}); retrying in {delay}s..."
                )
                time.sleep(delay)
                continue

            io.atomic_write_json(out_json, validated)
            print(f"[Worker-{chunk_idx:03d}] Finished {chunk_name}.")
            return True
    except Exception as e:  # noqa: BLE001 - A chunk failure must keep the run resumable.
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {chunk_name}: {e}")
        return False


IDENTITY_RESEARCH_SECTION = "PARTICIPANTS AND SPEAKERS"
TERMINOLOGY_RESEARCH_SECTION = "TOPIC TERMINOLOGY AND PROPER NOUNS"
NON_NAME_LABELS = frozenset({"evidence", "role", "aliases", "sources", "notes"})


def build_identity_research_prompt(source_title=None, context_urls=(), youtube_urls=()):
    """Build the plain-text prompt for the grounded web research pass."""
    title_block = ""
    if source_title:
        title_block = f"\nSOURCE TITLE\n\n{source_title}\n"
    url_block = ""
    if context_urls:
        url_lines = "\n".join(f"- {url}" for url in context_urls)
        url_block = (
            "\nCONTEXT URLS\n\n"
            f"{url_lines}\n"
            "Read the content at these URLs. They may identify the "
            "participants and the topic.\n"
        )
    youtube_block = ""
    if youtube_urls:
        youtube_lines = "\n".join(f"- {url}" for url in youtube_urls)
        youtube_block = (
            "\nYOUTUBE VIDEO URLS\n\n"
            f"{youtube_lines}\n"
            "Do not open these URLs. Their video content is analyzed in a "
            "separate pass. Treat the URLs as identifiers only.\n"
        )
    return f"""You research speaker identities and topic terminology for an English subtitle localization pass.

Use Google Search to research this video and return a concise plain-text summary with exactly two sections, in this order, using these exact section headers.

{IDENTITY_RESEARCH_SECTION}:
Canonical English public names, aliases, and roles for the people who speak in this video.
Begin each entry on its own line with the canonical English public name or the stable role, followed by a colon, then give the aliases, the role, and the cited evidence.

{TERMINOLOGY_RESEARCH_SECTION}:
Canonical English spelling of the recurring proper nouns, program or series titles, organization names, product names, and locations referenced in the source title or the context URLs.
Give one canonical spelling per line with the cited source.
{title_block}{url_block}{youtube_block}REQUIREMENTS

1. Use Google Search at least once and rely on reputable evidence.
2. Cite the source for each attribution so the evidence can be reviewed.
3. Rank evidence: reputable grounded web evidence first, the source title last.
4. Grounded research establishes canonical spelling and verified entities only. It must never infer, invent, or alter spoken dialogue content, meaning, or events.
5. When a participant's identity cannot be established, state one stable descriptive role such as Host, Resident, Shop Owner, or Producer when the role is clear; otherwise state that the speaker stays unlabeled.
6. Return plain text only, with no markdown formatting.
"""


def split_research_sections(research_text):
    """Split research text into identity and terminology section bodies.

    Section headers are matched case-insensitively with an optional trailing
    colon. Lines before the first header stay in the identity section so
    research output without section headers keeps working.
    """
    identity_lines = []
    terminology_lines = []
    target = identity_lines
    for line in research_text.splitlines():
        header = line.strip().upper().removesuffix(":")
        if header == IDENTITY_RESEARCH_SECTION:
            target = identity_lines
        elif header == TERMINOLOGY_RESEARCH_SECTION:
            target = terminology_lines
        else:
            target.append(line)
    identity = "\n".join(identity_lines).strip()
    terminology = "\n".join(terminology_lines).strip()
    return identity, terminology


def extract_grounded_names(identity_section):
    """Collect canonical name entries from the identity research section."""
    names = {}
    for line in identity_section.splitlines():
        match = core.SPEAKER_LABEL_RE.match(line.lstrip(" \t-*"))
        if not match:
            continue
        label = match.group(2).strip()
        if label and label.casefold() not in NON_NAME_LABELS:
            names.setdefault(label.casefold(), label)
    return list(names.values())


def build_youtube_analysis_prompt(source_title=None):
    """Build the plain-text prompt for the direct YouTube analysis pass."""
    title_block = ""
    if source_title:
        title_block = f"\nSOURCE TITLE\n\n{source_title}\n"
    return f"""You analyze public YouTube videos for an English subtitle localization pass.

Watch the attached video content.
Return concise plain text with:
- Each participant's name in official English styling and their role.
- Timestamped speaker-identification observations: when a visible label, title card, or spoken introduction establishes attribution, give the video timestamp and the evidence.

These observations may establish speaker identity and canonical proper-name spelling only.
They must never infer or change dialogue content, meaning, or events.
{title_block}Return plain text only, with no markdown formatting.
"""


def build_refinement_prompt(
    full_script,
    source_title=None,
    identity_context=None,
    terminology_context=None,
    youtube_context=None,
):
    source_block = ""
    if source_title:
        source_block = (
            "\nSource title: "
            f"{source_title}\n"
            "A name in the source title is a candidate identity, not proof "
            "that a specific line was spoken by that person.\n"
        )
    identity_block = ""
    if identity_context:
        identity_block = (
            "\nGROUNDED IDENTITY CONTEXT\n\n"
            f"{identity_context}\n"
            "The identity context above was researched with grounded web "
            "evidence. It ranks below explicit script introductions and title "
            "cards. It may establish speaker identity and canonical "
            "proper-name spelling only. It must never change dialogue "
            "meaning, events, or facts.\n"
        )
    terminology_block = ""
    if terminology_context:
        terminology_block = (
            "\nGROUNDED TERMINOLOGY CONTEXT\n\n"
            f"{terminology_context}\n"
            "The terminology context above was researched with grounded web "
            "evidence. It establishes canonical spelling of proper nouns, "
            "series and program titles, organizations, products, and "
            "locations only. It must never change dialogue meaning, events, "
            "or facts.\n"
        )
    youtube_block = ""
    if youtube_context:
        youtube_block = (
            "\nDIRECT VIDEO IDENTITY ANALYSIS\n\n"
            f"{youtube_context}\n"
            "The analysis above was produced by a separate pass that watched "
            "the source video content. It ranks below explicit script "
            "introductions and title cards. It may establish speaker identity "
            "and canonical proper-name spelling only. It must never change "
            "dialogue meaning, events, or facts.\n"
        )
    return f"""You are an expert English subtitle localization editor.

Below is the complete subtitle script for a video.

You do not have access to the source video or audio. Never infer or reconstruct source content that is not established by the provided script.
{source_block}{identity_block}{terminology_block}{youtube_block}Use the complete script as global context and correct only lines with a clear problem involving:

1. Speaker labels that are missing, inconsistent, conflicting, or attached to on-screen text. Audit speaker labels first, before polishing any text.
2. Inconsistent character names, brands, foods, products, program titles, or recurring terms.
3. Unnatural or ungrammatical English.
4. Literal translations of source-language idioms, slang, or editorial captions that are incomprehensible in English.
5. Clear continuity errors that can be resolved confidently from the script.
6. Formatting artifacts such as stray quotation marks, raw OCR debris, or inconsistent punctuation.

Do not rewrite the entire script. If a line is acceptable, leave it unchanged.

SEMANTIC PRESERVATION

7. Preserve each line's distinct semantic content.
8. Never delete a question, answer, joke, reaction, product detail, qualification, or meaningful on-screen caption.
9. Never replace a line with a duplicate or paraphrase of an adjacent line.
10. Never add dialogue, facts, product qualities, marketing claims, relationships, jokes, or events.
11. Do not infer what the original audio or on-screen text might have said.
12. If a proposed correction is uncertain, leave the line unchanged.
13. Do not merge, split, reorder, add, or remove subtitle entries.
14. Do not alter IDs or timestamps.

TERMINOLOGY AND LOCALIZATION

15. Preserve established names, brands, foods, products, program titles, and recurring terminology consistently.
16. Use the grounded terminology context to ensure consistent, canonical spelling of proper nouns, series and program titles, organization names, and location names.
17. Do not change proper-name romanization unless needed to correct an inconsistency clearly established within the script.
18. Do not replace understandable English with unexplained romanized source-language terms.
19. Preserve useful source-language cultural terms when they communicate a relationship or concept that ordinary English does not express as precisely.
20. Localize source-language idioms and editorial-caption metaphors into understandable English without inventing new meaning.
21. Preserve visible footnote markers such as "*".
22. Preserve meaningful vocalizations when they carry humor or characterization. Clarify them only when their meaning is unambiguous from the script.

SPEAKER LABELS

23. Rank speaker identity evidence in this order: an explicit introduction or title card within the script; the grounded identity context and the direct video analysis; the source title.
24. Use each confidently established person's official English name styling consistently.
25. A speaker label prefix such as "Name:" identifies who is speaking, using their established canonical English name or role.
26. Keep spoken dialogue text faithful to the spoken audio. Never alter spoken names, nicknames, titles, or address terms inside dialogue text merely to match a speaker label prefix.
27. Normalize labels when the evidence confidently establishes the identity.
28. Treat an abrupt label change near a chunk boundary as a likely generation error and normalize it to the established identity.
29. When conflicting identities are attached to one speaker and no identity is confidently established, replace them all with one stable descriptive role when the role is established in the script; otherwise remove the uncertain label.
30. Preserve each speaker's turn when multiple speakers occur in one caption.
31. When consecutive lines within one caption have the same speaker label and form one continuous turn, keep the label only once. Preserve every sentence, its order, and readable line breaks. Do not merge separate captions or alternating speaker turns.
32. Never infer identity from appearance.
33. Never add speaker labels to on-screen text.
34. The grounded identity context and the direct video analysis may establish speaker identity and canonical proper-name spelling only. They must never change dialogue meaning, events, or facts.

ON-SCREEN TEXT

35. Preserve square brackets around on-screen editorial text.
36. Keep on-screen text distinct from dialogue.
37. Do not convert on-screen text into spoken dialogue or accessibility-style action descriptions.
38. Remove mechanical prefixes such as "On-screen text:" while preserving the translated text itself.
39. Correct incomprehensible literal caption idioms only when the intended meaning can be established from the full script.

FORMATTING AND OUTPUT

40. Preserve line breaks when they distinguish multiple speakers.
41. Keep each subtitle to no more than 42 characters per line and two lines where possible without deleting meaning.
42. Return a JSON object containing a "changes" list with only entries that genuinely require correction.
43. Each change must contain the existing numeric subtitle "id" and the complete corrected "text".
44. Do not return unchanged entries.
45. Do not return timestamps, markdown, or explanations.

SCRIPT

{full_script}
"""


def build_research_config(thinking_level, ordinary_urls):
    """Build plain-text config with Google Search grounding."""
    tools = [types.Tool(google_search=types.GoogleSearch())]
    if ordinary_urls:
        tools.append(types.Tool(url_context=types.UrlContext()))
    kwargs = {
        "temperature": 0.0,
        "tools": tools,
    }
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        kwargs["thinking_config"] = thinking_config
    return build_content_config(**kwargs)


def build_youtube_analysis_config(thinking_level):
    """Build plain-text config for direct YouTube analysis without tools."""
    kwargs = {"temperature": 0.0}
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        kwargs["thinking_config"] = thinking_config
    return build_content_config(**kwargs)


def build_refinement_config(thinking_level):
    """Build structured config for script refinement without tools."""
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": core.RefinementResponse,
    }
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        kwargs["thinking_config"] = thinking_config
    return build_content_config(**kwargs)


def collect_stream_metadata(response_stream):
    """Collect response text and grounding metadata in one stream pass."""
    full_text = ""
    search_queries = []
    grounded_sources = []
    retrieved_urls = {}
    for chunk in response_stream:
        if chunk.text:
            full_text += chunk.text
        for candidate in chunk.candidates or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata:
                search_queries.extend(
                    query for query in (metadata.web_search_queries or []) if query
                )
                for grounding_chunk in metadata.grounding_chunks or []:
                    web = getattr(grounding_chunk, "web", None)
                    uri = web and getattr(web, "uri", None)
                    if uri:
                        grounded_sources.append((getattr(web, "title", None), uri))
            url_context = getattr(candidate, "url_context_metadata", None)
            if url_context:
                for entry in url_context.url_metadata or []:
                    url = entry.retrieved_url
                    if url:
                        retrieved_urls[url] = entry.url_retrieval_status
    return full_text, search_queries, grounded_sources, retrieved_urls


def retrieval_status_value(status):
    """Return the plain status string for an enum or raw value."""
    return getattr(status, "value", status)


def verify_refinement_grounding(
    search_queries, grounded_sources, retrieved_urls, context_urls
):
    """Fail refinement before publication when grounding requirements are unmet."""
    if not search_queries and not grounded_sources:
        raise RuntimeError(
            "The identity research response has no Google Search grounding. "
            "Failing without publishing output."
        )

    retrieved_by_identity = {
        core.url_identity(url): status for url, status in retrieved_urls.items()
    }
    for url in context_urls:
        status = retrieved_by_identity.get(core.url_identity(url))
        if status is None:
            raise RuntimeError(
                f"Context URL {url} was not retrieved. "
                "Failing without publishing output."
            )
        if (
            str(retrieval_status_value(status)).upper()
            != "URL_RETRIEVAL_STATUS_SUCCESS"
        ):
            raise RuntimeError(
                f"Context URL {url} retrieval failed with "
                f"{retrieval_status_value(status)}. "
                "Failing without publishing output."
            )


def print_refinement_grounding(
    search_queries, grounded_sources, retrieved_urls, context_urls
):
    """Print summary of search queries, grounded sources, and URL retrieval."""
    unique_queries = list(dict.fromkeys(search_queries))
    if unique_queries:
        print("Search queries:")
        for query in unique_queries:
            print(f"  - {query}")
    unique_sources = list(dict.fromkeys(grounded_sources))
    if unique_sources:
        print("Grounded sources:")
        for title, uri in unique_sources:
            print(f"  - {title or 'Untitled'}: {uri}")
    if context_urls:
        print("Context URL retrieval:")
        retrieved_by_identity = {
            core.url_identity(url): (url, status)
            for url, status in retrieved_urls.items()
        }
        for url in context_urls:
            entry = retrieved_by_identity.get(core.url_identity(url))
            status = retrieval_status_value(entry[1]) if entry else "NOT RETRIEVED"
            print(f"  - {url}: {status}")


def global_refine_subtitles(
    input_vtt,
    output_vtt,
    api_key,
    base_url,
    model_name,
    thinking_level,
    source_title=None,
    context_urls=None,
    grounded_names=None,
):
    """Run the global refinement pipeline on an input WebVTT file."""
    context_urls = core.validate_context_urls(context_urls)
    youtube_urls, ordinary_urls = core.classify_context_urls(context_urls)
    print(f"Loading {input_vtt} for global refinement...")
    vtt = webvtt.read(input_vtt)

    script_lines = []
    for i, caption in enumerate(vtt):
        script_lines.append(f"[{i}] {caption.start} --> {caption.end}: {caption.text}")

    full_script = "\n".join(script_lines)

    # 1. Grounded web research pass. Plain text with Google Search.
    # No video Parts: YouTube content is analyzed in a separate request.
    research_prompt = build_identity_research_prompt(
        source_title, ordinary_urls, youtube_urls
    )
    with create_client(api_key, base_url) as client:
        print(
            "Researching participants and terminology with Google Search "
            "(this may take a minute)..."
        )
        research_stream = client.models.generate_content_stream(
            model=model_name,
            contents=research_prompt,
            config=build_research_config(thinking_level, ordinary_urls),
        )
        (
            research_text,
            search_queries,
            grounded_sources,
            retrieved_urls,
        ) = collect_stream_metadata(research_stream)

    verify_refinement_grounding(
        search_queries, grounded_sources, retrieved_urls, ordinary_urls
    )
    print_refinement_grounding(
        search_queries, grounded_sources, retrieved_urls, ordinary_urls
    )

    # 2. Direct YouTube identity analysis. Only when YouTube context URLs
    # exist. Plain text with video Parts and no tools; request completion is
    # the success signal for public video retrieval.
    youtube_analysis_text = ""
    if youtube_urls:
        print("YouTube video context (direct video input):")
        for url in youtube_urls:
            print(f"  - {url}")
        with create_client(api_key, base_url) as client:
            print(
                "Analyzing YouTube videos for speaker identities "
                "(this may take a minute)..."
            )
            youtube_contents = [
                types.Part.from_uri(file_uri=url, mime_type="video/*")
                for url in youtube_urls
            ]
            youtube_contents.append(build_youtube_analysis_prompt(source_title))
            youtube_stream = client.models.generate_content_stream(
                model=model_name,
                contents=youtube_contents,
                config=build_youtube_analysis_config(thinking_level),
            )
            youtube_analysis_text, *_ = collect_stream_metadata(youtube_stream)

    # 3. Structured refinement pass. No tools; the grounded identity and
    # terminology sections supply context.
    identity_section, terminology_section = split_research_sections(research_text)
    prompt = build_refinement_prompt(
        full_script,
        source_title,
        identity_context=identity_section,
        terminology_context=terminology_section,
        youtube_context=youtube_analysis_text,
    )

    with create_client(api_key, base_url) as client:
        print(
            "Sending script to Gemini for global refinement (this may take a minute)..."
        )
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=build_refinement_config(thinking_level),
        )
        full_json_text = ""
        for response_chunk in response_stream:
            if response_chunk.text:
                full_json_text += response_chunk.text

    try:
        refinements = core.RefinementResponse.model_validate_json(full_json_text)
        core.validate_refinement_changes(refinements.changes, len(vtt))
    except ValueError as e:
        raise RuntimeError(
            "Parsing or validating the model refinement response failed: "
            f"{e}\nRaw response:\n{full_json_text}"
        ) from e

    changes = refinements.changes
    print(f"Model proposed changes to {len(changes)} lines out of {len(vtt)}.")

    for change in changes:
        vtt[change.id].text = change.text

    # Caller-supplied grounded names win over researched names on a
    # case-insensitive collision inside canonicalize_speaker_casing.
    researched_names = extract_grounded_names(identity_section)
    core.canonicalize_speaker_casing(
        vtt, grounded_names=[*researched_names, *(grounded_names or ())]
    )
    io.atomic_save_vtt(vtt, output_vtt)
    print(f"Saved refined subtitles to {output_vtt}")


def load_script_entries(vtt_path):
    """Read a VTT into script entries with sequential request-only IDs."""
    vtt = webvtt.read(str(vtt_path))
    entries = []
    for index, caption in enumerate(vtt):
        entries.append(
            {
                "id": index,
                "start": core.format_time(core.parse_time(caption.start)),
                "end": core.format_time(core.parse_time(caption.end)),
                "text": caption.text,
                "classification": core.classify_cue_text(caption.text),
            }
        )
    return entries


def build_audio_refinement_prompt(
    source_entries, boundaries, audio_duration, source_title=None
):
    boundary_lines = "\n".join(
        f"- {core.format_time(boundary)}" for boundary in boundaries
    )
    script_lines = "\n".join(
        f"[{entry['id']}] {entry['start']} --> {entry['end']} "
        f"[{entry['classification']}]: {entry['text']}"
        for entry in source_entries
    )
    mode_block = (
        "MODE: boundary-limited\n\n"
        "Repair authority is limited to connected regions spanning ten "
        "seconds before and ten seconds after each actual segment boundary.\n"
        "Every cue outside those regions must remain byte-for-byte identical "
        "in text and timing.\n"
        "A changed cue may use one repair region plus the full original time "
        "extents of every referenced source cue that intersects that same "
        "region. It must stay inside that combined envelope. All referenced "
        "source cues must share the same repair region. Recovered cues must "
        "stay inside a repair region."
    )
    title_block = ""
    if source_title:
        title_block = (
            "SOURCE TITLE\n\n"
            f"{source_title}\n\n"
            "Names in the source title are candidate identities only.\n\n"
        )
    return f"""You are an expert audio subtitle repair editor.

Listen to the complete attached audio and compare it with the complete subtitle script below.

{title_block}COMPLETE AUDIO DURATION

{audio_duration:.3f} seconds

ACTUAL SEGMENT BOUNDARIES

{boundary_lines}

The script segments were cut at these source timestamps. A spoken turn or visual cue crossing a boundary may be missing, duplicated, split, mistranslated, or poorly timed in the stitched script.

SCRIPT

{script_lines}

{mode_block}

REPAIR RULES

1. Fix dialogue that disagrees with what the audio says: add missing spoken dialogue, delete duplicated or hallucinated dialogue, merge split turns, split merged turns, rewrite mistranslations, and retime cues to the audible syllables.
2. Preserve silent gaps. Start dialogue at the first audible syllable and end it at the last audible syllable.
3. Keep visual content untouched. The audio cannot establish what appears on screen. Preserve every [bracketed] visual fragment exactly as it appears, attached to its original cue lineage. Pure editorial cues must remain byte-for-byte identical in text and timing.
4. A mixed cue may change timing when its dialogue is repaired, but every bracketed fragment must be preserved exactly once within that lineage.
5. Do not add bracketed text to a recovered cue.
6. Keep speaker labels and their turn structure. Do not guess identities from the audio.
7. Do not describe sounds, music, or actions unless the script already does.
8. Every output cue must contain text.

RESPONSE FORMAT

Return one sparse patch object. Do not return the complete script.
- "contractVersion": exactly "{AUDIO_REFINE_RESPONSE_CONTRACT}".
- "cues": only changed, replacement, split, merged, or recovered cues, in script order. Omit every unchanged source cue; the host preserves omitted cues exactly.
- "deletedSourceIds": IDs of false dialogue cues removed from the script. Omission alone never deletes a source cue.
- One source ID with changed text or timing is a rewrite or retime.
- Multiple strictly increasing source IDs in one cue are a merge.
- One source ID repeated in adjacent output cues is a split; a split cue must carry only that ID.
- An empty "sourceIds" list is recovered spoken dialogue.
- Every changed source ID must appear in a patch cue or in "deletedSourceIds", never both. Unmentioned IDs remain unchanged.
- Timestamps must be source-relative in canonical HH:MM:SS.mmm format.
- Return only valid JSON matching the response schema.
"""


def audio_refinement_config():
    """Build structured config for boundary audio refinement without tools."""
    schema = core.AudioRefinementResponse.model_json_schema()
    cue_schema = schema.pop("$defs")["AudioRefinedCue"]
    schema["properties"]["cues"]["items"] = cue_schema

    def remove_additional_properties(value):
        if isinstance(value, dict):
            return {
                key: remove_additional_properties(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [remove_additional_properties(item) for item in value]
        return value

    return build_content_config(
        temperature=0.0,
        response_mime_type="application/json",
        # The Gemini endpoint rejects Pydantic's additionalProperties keyword.
        # Host-side parsing still enforces extra="forbid" on both models.
        response_json_schema=remove_additional_properties(schema),
        thinking_config=build_thinking_config(AUDIO_REFINE_THINKING_LEVEL),
        max_output_tokens=AUDIO_REFINE_MAX_OUTPUT_TOKENS,
    )


def run_audio_refinement_request(
    api_key,
    base_url,
    model,
    source_entries,
    boundaries,
    audio_duration,
    audio_path,
    source_title=None,
):
    """Run one streamed audio-refinement request and return (raw_text, response)."""
    prompt = build_audio_refinement_prompt(
        source_entries, boundaries, audio_duration, source_title
    )
    with open(audio_path, "rb") as handle:
        audio_bytes = handle.read()
    if len(audio_bytes) > INLINE_VIDEO_WARNING_BYTES:
        print(
            f"Warning: {audio_path} is {len(audio_bytes) / 1024 / 1024:.1f} MB. "
            "Gemini docs recommend inline audio below 20 MB; it will still be sent."
        )
    with create_client(api_key, base_url) as client:
        print("Sending complete audio and script for boundary audio refinement...")
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes, mime_type=media.AUDIO_MIME_TYPE
                ),
                prompt,
            ],
            config=audio_refinement_config(),
        )
        full_json_text = ""
        for response_chunk in response_stream:
            if response_chunk.text:
                full_json_text += response_chunk.text
            for candidate in getattr(response_chunk, "candidates", None) or []:
                reason = getattr(candidate, "finish_reason", None)
                reason_text = str(getattr(reason, "value", reason)).upper()
                if reason_text == "MAX_TOKENS":
                    raise RuntimeError(
                        "Audio refinement response exceeded the configured "
                        "output budget (MAX_TOKENS)."
                    )
    try:
        return full_json_text, core.AudioRefinementResponse.model_validate_json(
            full_json_text
        )
    except ValidationError as error:
        raise RuntimeError(
            "Parsing or validating the audio refinement response failed: "
            f"{error}\nRaw response:\n{full_json_text}"
        ) from error


def audio_refinement_cache_identity(
    script_path, audio_path, audio_duration, boundaries, model
):
    """Return cache identity of one audio-refinement response."""
    return {
        "script_sha256": hashlib.sha256(Path(script_path).read_bytes()).hexdigest(),
        "audio_sha256": hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
        "audio_duration": audio_duration,
        "boundaries": list(boundaries),
        "model": model,
        "thinking_level": AUDIO_REFINE_THINKING_LEVEL,
        "response_contract": AUDIO_REFINE_RESPONSE_CONTRACT,
        "mode": "boundary",
    }


def audio_cache_paths(work_dir):
    """Return boundary audio-refinement cache artifact paths."""
    return (
        Path(work_dir) / "audio_refinement.json",
        Path(work_dir) / "audio_refinement.meta.json",
    )


def store_audio_refinement_cache(work_dir, identity, response):
    """Save validated audio refinement response and metadata to cache."""
    response_path, meta_path = audio_cache_paths(work_dir)
    io.atomic_write_json(response_path, response.model_dump(by_alias=True))
    io.atomic_write_json(meta_path, {"identity": identity})


def load_cached_audio_refinement(work_dir, identity):
    """Load cached audio refinement response matching identity, or return None."""
    response_path, meta_path = audio_cache_paths(work_dir)
    if not response_path.exists() or not meta_path.exists():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict) or stored.get("identity") != identity:
            raise ValueError("cache identity mismatch")
        response = core.AudioRefinementResponse.model_validate(
            json.loads(response_path.read_text(encoding="utf-8"))
        )
    except (ValueError, OSError) as error:
        print(f"Ignoring invalid audio refinement cache {response_path}: {error}")
        discard_audio_refinement_cache(work_dir)
        return None
    return response


def discard_audio_refinement_cache(work_dir):
    """Remove cached audio refinement response and metadata files."""
    for path in audio_cache_paths(work_dir):
        path.unlink(missing_ok=True)


def validated_cached_audio_refinement(
    work_dir, identity, source_entries, audio_duration, boundaries
):
    """Return output for a valid cache, or discard an invalid cache."""
    cached = load_cached_audio_refinement(work_dir, identity)
    if cached is None:
        return None
    try:
        validated = core.validate_audio_refinement_response(
            cached, source_entries, audio_duration, boundaries
        )
    except ValueError as error:
        print(
            f"Ignoring semantically invalid cached audio refinement response: {error}"
        )
        discard_audio_refinement_cache(work_dir)
        return None
    print("Reusing cached boundary audio refinement response.")
    return validated


def revalidate_candidate(candidate_path, output_entries):
    """Verify that VTT serialization preserved every validated cue exactly."""
    vtt = webvtt.read(str(candidate_path))
    actual = [(caption.start, caption.end, caption.text) for caption in vtt]
    expected = [
        (entry["start"], entry["end"], entry["text"]) for entry in output_entries
    ]
    if actual != expected:
        raise ValueError(
            "Serialized candidate does not match the validated audio "
            "refinement response"
        )


def boundary_audio_refine_subtitles(
    stitched_vtt,
    audio_path,
    audio_duration,
    boundaries,
    work_dir,
    output_vtt,
    api_key,
    base_url,
    model_name=DEFAULT_AUDIO_REFINE_MODEL,
    source_title=None,
):
    """Run or reuse boundary audio refinement and publish atomically."""
    source_entries = load_script_entries(stitched_vtt)
    identity = audio_refinement_cache_identity(
        stitched_vtt, audio_path, audio_duration, boundaries, model_name
    )
    validated = validated_cached_audio_refinement(
        work_dir, identity, source_entries, audio_duration, boundaries
    )
    response_to_cache = None
    raw_json = None
    if validated is None:
        raw_json, response = run_audio_refinement_request(
            api_key,
            base_url,
            model_name,
            source_entries,
            boundaries,
            audio_duration,
            audio_path,
            source_title=source_title,
        )
        response_to_cache = response
    candidate_path = Path(work_dir) / "audio_refined.vtt"
    candidate_tmp = Path(work_dir) / "audio_refined.vtt.tmp"
    candidate_tmp.unlink(missing_ok=True)
    try:
        if response_to_cache is not None:
            try:
                validated = core.validate_audio_refinement_response(
                    response_to_cache,
                    source_entries,
                    audio_duration,
                    boundaries,
                )
            except ValueError as error:
                raise RuntimeError(
                    "boundary audio refinement response failed validation: "
                    f"{error}\nRaw response:\n{raw_json}"
                ) from error
        output_entries = validated
        candidate_vtt = webvtt.WebVTT()
        for entry in output_entries:
            candidate_vtt.captions.append(
                webvtt.Caption(entry["start"], entry["end"], entry["text"])
            )
        io.atomic_save_vtt(candidate_vtt, candidate_tmp)
        revalidate_candidate(candidate_tmp, output_entries)
        os.replace(candidate_tmp, candidate_path)
        if response_to_cache is not None:
            store_audio_refinement_cache(work_dir, identity, response_to_cache)
    except Exception:
        candidate_tmp.unlink(missing_ok=True)
        discard_audio_refinement_cache(work_dir)
        raise
    if Path(output_vtt) != candidate_path:
        io.atomic_save_vtt(candidate_vtt, output_vtt)
    print(f"Saved boundary audio-refined subtitles to {output_vtt}")
