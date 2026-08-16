"""Gemini API clients, prompts, request configs, chunk requests, and refinement."""

import json
import os

import webvtt
from google import genai
from google.genai import types

from modules import core, io

INLINE_VIDEO_WARNING_BYTES = 20 * 1024 * 1024
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_CHUNK_MODEL = "gemini-3.7-flash"
DEFAULT_REFINE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CHUNK_THINKING_LEVEL = "high"
REFINEMENT_THINKING_LEVEL = "medium"


def validate_thinking_level_for_model(model_name, thinking_level):
    if thinking_level == "minimal" and "flash" not in model_name.lower():
        raise ValueError(
            "--thinking-level minimal is only supported by Flash models. Use low, medium, or high for this model."
        )


def create_client(api_key, base_url):
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["http_options"] = {"base_url": base_url}
    return genai.Client(**kwargs)


def generate_content_config(thinking_level):
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": core.SubtitleResponse,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_generation_prompt(
    clip_duration, owner_start_rel, owner_end_rel, source_title=None
):
    source_block = ""
    if source_title:
        source_block = (
            "SOURCE CONTEXT\n\n"
            f"Source title: {source_title}\n"
            "Names in the source title are candidate identities only. "
            "They do not prove which speaker said a specific line.\n\n"
        )
    return f"""You are an expert subtitle generator and translator.

Watch this {clip_duration:.3f}-second video clip.

The main chunk window is {core.format_time(owner_start_rel)} to {core.format_time(owner_end_rel)} in this clip. Video before or after that window is context only.

Generate accurate, natural English subtitles for dialogue and meaningful on-screen text throughout the entire clip, including the context windows. Captions outside the main window will be filtered later.

{source_block}TIMING

1. Create timestamps relative to the beginning of the full clip, ranging from 00:00:00.000 to {core.format_time(clip_duration)}.
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
    chunk_idx = chunk["idx"]
    clip_name = chunk["clip_name"]
    clip_duration = chunk["clip_duration"]
    owner_start_rel = chunk["owner_start_rel"]
    owner_end_rel = chunk["owner_end_rel"]
    out_json = os.path.join(chunk_dir, f"subtitle_chunk_{chunk_idx:03d}.json")
    chunk_path = os.path.join(chunk_dir, clip_name)

    cached = load_cached_captions(out_json, clip_duration)
    if cached is not None:
        print(f"Skipping {clip_name} - already processed.")
        return True

    prompt = build_generation_prompt(
        clip_duration, owner_start_rel, owner_end_rel, source_title
    )

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()
        if len(video_data) > INLINE_VIDEO_WARNING_BYTES:
            print(
                f"[Worker-{chunk_idx:03d}] Warning: {clip_name} is {len(video_data) / 1024 / 1024:.1f} MB. "
                "Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail."
            )

        print(f"[Worker-{chunk_idx:03d}] Generating {clip_name} using Gemini API...")

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

        parsed_response = core.SubtitleResponse.model_validate_json(full_json_text)
        validated = core.validate_captions(parsed_response.captions, clip_duration)
        io.atomic_write_json(out_json, validated)

        print(f"[Worker-{chunk_idx:03d}] Finished {clip_name}.")
        return True
    except Exception as e:  # noqa: BLE001 - A chunk failure must keep the run resumable.
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {clip_name}: {e}")
        return False


def build_identity_research_prompt(source_title=None, context_urls=(), youtube_urls=()):
    """Build the plain-text prompt for the grounded web identity research pass."""
    title_block = ""
    if source_title:
        title_block = f"\nSOURCE TITLE\n\n{source_title}\n"
    url_block = ""
    if context_urls:
        url_lines = "\n".join(f"- {url}" for url in context_urls)
        url_block = (
            "\nCONTEXT URLS\n\n"
            f"{url_lines}\n"
            "Read the content at these URLs. They may identify the participants.\n"
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
    return f"""You research speaker identities for an English subtitle localization pass.

Return a concise plain-text summary of the participants who speak in this video.
For each participant return their name in official English styling, their role, and the evidence for that attribution.
Evidence must come from reputable web sources.
{title_block}{url_block}{youtube_block}REQUIREMENTS

1. Use Google Search at least once and rely on reputable evidence.
2. Cite the source for each attribution so the evidence can be reviewed.
3. Rank identity evidence: reputable grounded web evidence first, the source title last.
4. Web evidence may establish speaker identity and canonical proper-name spelling only. It must never infer or change dialogue content, meaning, or events.
5. When identity cannot be established, state one stable descriptive role such as Host, Resident, Shop Owner, or Producer when the role is clear; otherwise state that the speaker stays unlabeled.
6. Return plain text only, with no markdown formatting.
"""


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
    full_script, source_title=None, identity_context=None, youtube_context=None
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
{source_block}{identity_block}{youtube_block}Use the complete script as global context and correct only lines with a clear problem involving:

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
16. Do not change proper-name romanization unless needed to correct an inconsistency clearly established within the script.
17. Do not replace understandable English with unexplained romanized source-language terms.
18. Preserve useful source-language cultural terms when they communicate a relationship or concept that ordinary English does not express as precisely.
19. Localize source-language idioms and editorial-caption metaphors into understandable English without inventing new meaning.
20. Preserve visible footnote markers such as "*".
21. Preserve meaningful vocalizations when they carry humor or characterization. Clarify them only when their meaning is unambiguous from the script.

SPEAKER LABELS

22. Rank speaker identity evidence in this order: an explicit introduction or title card within the script; the grounded identity context and the direct video analysis; the source title.
23. Use each confidently established person's official English name styling consistently.
24. Normalize labels when the evidence confidently establishes the identity.
25. Treat an abrupt label change near a chunk boundary as a likely generation error and normalize it to the established identity.
26. When conflicting identities are attached to one speaker and no identity is confidently established, replace them all with one stable descriptive role when the role is established in the script; otherwise remove the uncertain label.
27. Preserve each speaker's turn when multiple speakers occur in one caption.
28. When consecutive lines within one caption have the same speaker label and form one continuous turn, keep the label only once. Preserve every sentence, its order, and readable line breaks. Do not merge separate captions or alternating speaker turns.
29. Never infer identity from appearance.
30. Never add speaker labels to on-screen text.
31. The grounded identity context and the direct video analysis may establish speaker identity and canonical proper-name spelling only. They must never change dialogue meaning, events, or facts.

ON-SCREEN TEXT

32. Preserve square brackets around on-screen editorial text.
33. Keep on-screen text distinct from dialogue.
34. Do not convert on-screen text into spoken dialogue or accessibility-style action descriptions.
35. Remove mechanical prefixes such as "On-screen text:" while preserving the translated text itself.
36. Correct incomprehensible literal caption idioms only when the intended meaning can be established from the full script.

FORMATTING AND OUTPUT

37. Preserve line breaks when they distinguish multiple speakers.
38. Keep each subtitle to no more than 42 characters per line and two lines where possible without deleting meaning.
39. Return a JSON object containing a "changes" list with only entries that genuinely require correction.
40. Each change must contain the existing numeric subtitle "id" and the complete corrected "text".
41. Do not return unchanged entries.
42. Do not return timestamps, markdown, or explanations.

SCRIPT

{full_script}
"""


def build_research_config(thinking_level, ordinary_urls):
    """Plain-text config that always enables Google Search grounding."""
    tools = [types.Tool(google_search=types.GoogleSearch())]
    if ordinary_urls:
        tools.append(types.Tool(url_context=types.UrlContext()))
    kwargs = {
        "temperature": 0.0,
        "tools": tools,
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_youtube_analysis_config(thinking_level):
    """Plain-text config for the direct YouTube analysis pass. No tools."""
    kwargs = {"temperature": 0.0}
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_refinement_config(thinking_level):
    """Structured config for the script refinement pass. No tools."""
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": core.RefinementResponse,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


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
    """Return the plain status string for a enum or raw value."""
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
    boundary_provenance=None,
):
    context_urls = core.validate_context_urls(context_urls)
    youtube_urls, ordinary_urls = core.classify_context_urls(context_urls)
    print(f"Loading {input_vtt} for global refinement...")
    vtt = webvtt.read(input_vtt)
    if boundary_provenance is not None and len(vtt) != len(boundary_provenance):
        raise ValueError(
            "boundary dedup requires one chunk index per caption: "
            f"{len(vtt)} captions, {len(boundary_provenance)} indices"
        )

    script_lines = []
    for i, caption in enumerate(vtt):
        script_lines.append(f"[{i}] {caption.start} --> {caption.end}: {caption.text}")

    full_script = "\n".join(script_lines)

    # 1. Grounded web identity research pass. Plain text with Google Search.
    # No video Parts: YouTube content is analyzed in a separate request.
    research_prompt = build_identity_research_prompt(
        source_title, ordinary_urls, youtube_urls
    )
    with create_client(api_key, base_url) as client:
        print(
            "Researching speaker identities with Google Search "
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

    # 3. Structured refinement pass. No tools; the identity sections supply
    # context.
    prompt = build_refinement_prompt(
        full_script, source_title, research_text, youtube_analysis_text
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

    if boundary_provenance is not None:
        core.dedup_boundary_overlap(vtt, boundary_provenance)

    io.atomic_save_vtt(vtt, output_vtt)
    print(f"Saved refined subtitles to {output_vtt}")
