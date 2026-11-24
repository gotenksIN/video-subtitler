# Project context and decisions

This document records the architectural choices, domain rules, and design trade-offs behind `video-subtitler`.
See `README.md` for installation and usage, `AGENTS.md` for project-specific coding and testing rules, and `tests/README.md` for the test contract matrix.

## Architecture overview

The tool generates English WebVTT subtitles through a four-pass audio-first pipeline:

0. **Preflight (`Pass 0`):** Runs grounded web research and optional direct YouTube video analysis to build candidate participant identities and topic terminology.
1. **Chunk generation (`Pass 1`):** Subtitles stream-copy video chunks concurrently using the candidate roster.
2. **Boundary audio refinement (`Pass 2`):** Listens to complete extracted audio and repairs faults near chunk boundaries with sparse patches.
3. **Global text refinement (`Pass 3`):** Proofreads the full script with preflight context and optional complete audio without changing timestamps.

The codebase keeps an acyclic module layout: `modules/core.py` and `modules/io.py` have no internal imports, `modules/media.py` depends only on `io`, `modules/gemini.py` depends on `core`, `io`, and `media`, `modules/pipeline.py` orchestrates runs, and `gemini_subs.py` handles CLI dispatch.

## Project decisions

### Models, thinking levels, and Gemini API

- **Default `gemini-4-hydrogen` with `high` thinking instead of `xhigh`:** Chunk generation accepts `--thinking-level xhigh` as an explicit opt-in on `gemini-4-hydrogen`.
  Default chunk generation and fixed Passes 0, 2, and 3 stay on `high` because `xhigh` is substantially more expensive.
- **Static video over `MediaProcessing.AGENTIC` for whole-timeline passes:** Pass 0 YouTube analysis and Pass 1 chunk generation use static video input.
  In cold-cache A/B benchmarking on `gemini-4-hydrogen` (commit `618d123`), agentic Pass 0 produces 2.8x latency with 10.9x more thought and output tokens for only 2.5% fewer total tokens.
- **`generateContent` over Interactions API:** The pipeline uses `client.models.generate_content_stream` with automatic function calling disabled through `_build_content_config()`.
  Passes 0 through 2 are single-turn, avoiding server-side interaction storage, and `generate_content_stream` already provides structured JSON schemas, `automatic_continuation=True` long decode (`1_048_576` tokens on Gemini 4), and multimodal function-result types.
- **Thinking streaming for long requests and strict completion checks:** Requests with a thinking level set `include_thoughts=True` so thought tokens keep HTTP streams alive while host consumers discard thought-only chunks from assembled output.
  Incomplete streams without a `STOP` finish reason, `MAX_TOKENS`, or blocked responses fail or retry before parsing rather than publishing truncated subtitles.
- **Unset `temperature`:** Sampling temperature stays unset across Gemini 4 requests.
- **Inline media with `50 MiB` warnings instead of Files API uploads:** Extracted audio and video chunks stay inline to avoid remote file lifecycle state and `--base-url` proxy incompatibilities.
  Complete audio uses mono Opus at `32 kbps` so multi-hour tracks stay compact inside the 100 MB inline payload ceiling.

### Media, chunking, and timing

- **Stream-copy keyframe splitting over re-encoding:** Pass 1 splits inputs with `-c copy` and uses `segments.csv` as the ground-truth timeline (`-reset_timestamps 1`) rather than assuming exact fixed-duration cuts or paying re-encoding cost.
- **Supported container/codec set:** Probing supports VP9 (`.webm`), H.264 (`.mp4`), and HEVC/H.265 (`.mp4`), rejecting unsupported codecs such as AV1 before API work.
- **Stitch end clamping without cross-chunk drift:** Stitched cue ends clamp to the next segment start before millisecond rounding, rejecting collapsed intervals early during generation and cache reuse so overlapping boundaries never survive into downstream passes.
  Intra-chunk overlapping cues keep their canonical timing.
- **Audio-first boundary repair (`sparse-patch-v1`) over full re-transcription:** Pass 2 repairs only `[boundary - 10s, boundary + 10s]` repair regions using `AudioRefinementResponse` sparse patches, and the host discards out-of-authority edits so unaffected cues copy verbatim.
  Single-chunk runs skip Pass 2 because no repair boundaries exist, though audio is still extracted when Pass 3 is enabled.
- **Visual-cue duration preservation:** Unchanged singleton editorial and mixed cues keep their exact source-video end timing even when audio ends earlier, whereas dialogue cues obey audio duration with a 0.5-second overrun clamp.

### Subtitle content, attribution, and translation

- **Evidence hierarchy for speaker identity:** Attribution follows `explicit script/in-clip evidence > Pass 0 Direct Video Identity Analysis > Pass 0 Grounded Identity Context`.
  Audio voice continuity may confirm an established `1:1` identity, but appearance or voice timbre alone never identifies a speaker.
- **Specific role labels over vague umbrellas and unverified names:** Unattributed speakers use concrete single-person roles (`Host:`, `Presenter:`, `Narrator:`, `Chef:`, etc.) rather than `Person:` or `Participant:`.
  Pass 3 unifies isolated chunk role fallbacks only on clear `1:1` evidence and never adds labels to unlabeled dialogue.
- **Deterministic speaker-casing canonicalization:** Grounded names from Pass 0 take precedence over script frequency when normalizing speaker label casing on final VTT publication; runs without Pass 3 canonicalize using script frequency alone.
- **Intentional preservation of visual graphics:** Bracketed variety captions, echo captions, `[?????]`, and `[...]` graphics stay intact rather than being stripped to lower reading speed.
  Pass 2 forbids editing bracketed visual fragments, and pure-editorial cues split at keyframes merge case-insensitively during stitching.
- **Natural English with preserved cultural terms:** Translation uses natural English SVO phrasing with context-recovered subjects and objects while keeping culturally specific foods, customs, and relational terms instead of domesticating them.
- **Plain-text WebVTT boundary:** Pipeline cues contain semantic plain text without inline WebVTT tags, and `--refine-only` rejects external tagged VTT rather than silently stripping annotations.

### Reliability, caching, and testing

- **Fail-closed preflight grounding:** Pass 0 fails before Pass 1 (or before output publication in `--refine-only`) when Search grounding is absent or a context URL fails retrieval, avoiding silent ungrounded fallback.
  Changing `--context-url` invalidates cached Pass 1 chunks so generation reruns with the updated candidate roster.
- **Resumable work directories and atomic publication:** One non-blocking `fcntl.flock()` covers the entire run lifecycle; failed runs preserve `pipeline.log`, valid chunks, audio, and stage caches for retry, and all JSON, audio, and VTT outputs publish atomically through temporary siblings and read-back verification before `os.replace()`.
- **Stable cache keys:** Cache identities (`preflight-v1`, manifest SHA-256 prefix, and audio patch cache identity) change only intentionally so valid resume artifacts stay reusable across retries.
- **Incremental validation retries in Pass 3:** Invalid `RefinementResponse` patches retain valid edits and deletions across up to 3 attempts, preserve the initial prompt prefix for implicit caching, and carry forward previous structural feedback even after JSON parse failures.
- **Real media and real SDK HTTP transport in tests:** Tests use tiny real FFmpeg fixtures and `ScriptedGeminiClient` over `httpx2.MockTransport` instead of fake FFmpeg runners or monkeypatched pipeline stages, keeping the suite fast and contract-focused.

## Domain terms

- **Cue classes:** `dialogue` (no brackets), `editorial` (only `[...]` fragments and whitespace), and `mixed` (both `[...]` fragments and spoken words).
- **Repair region:** Connected union of 10 seconds before through 10 seconds after segment boundaries.
- **Boundary envelope:** A repair region expanded by the original time extents of referenced source cues plus the repair window on each side.
- **Sparse patch (`sparse-patch-v1`):** Pass 2 contract emitting only changed, split, merged, or recovered cues plus deleted source IDs.
- **Pure-editorial boundary merge:** Stitch pass merging case-insensitive identical `editorial` cues split within 0.5 seconds of a chunk boundary.
