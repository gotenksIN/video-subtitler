# Video subtitler

This context defines the architecture, data schemas, algorithms, prompt contracts, and operational rules required to recreate the video subtitler pipeline from first principles.

## Purpose

This project is a Python command-line tool that creates English WebVTT subtitles from video.
It uses an audio-first pipeline with three Gemini passes:
1. Stream-copy video chunks are subtitled concurrently with Gemini Flash.
2. A boundary-limited audio pass listens to the complete extracted audio and repairs faults near chunk boundaries.
3. A grounded text pass improves the complete subtitle script with identity research.

The design targets accurate speech timing, faithful translation, and useful on-screen text.
It favors throughput and resumability.
FFmpeg performs local media work.
The Gemini API performs video interpretation, audio repair, and translation.
Pydantic validates every structured model response before it reaches the output file.

## System model

The normal generation path runs these stages:

```mermaid
flowchart TD
    A[Source video] --> B[Probe primary video stream]
    B --> C[Build manifest and lock work directory]
    C --> D[Probe first audio stream and extract mono Opus when enabled]
    D --> E[Stream-copy contiguous video chunks]
    E --> F[Generate chunk subtitles concurrently]
    F --> G[Stitch at actual segment offsets]
    G --> H[Merge exact pure-editorial boundary fragments]
    H --> I{Audio refinement enabled?}
    I -->|Yes| J[Boundary-limited audio refinement]
    I -->|No| K[Use stitched VTT]
    J --> L{Text refinement enabled?}
    K --> L
    L -->|Yes| M[Grounded text refinement]
    L -->|No| N[Publish selected artifact atomically]
    M --> N
    N --> O[Clean completed work files]
```

The command holds one process lock for the complete work-directory lifecycle.
The lock covers splitting, audio extraction, API calls, output publication, and success cleanup.

## Repository layout

Every tracked file in this repository has a defined responsibility.

| Path | Responsibility |
| --- | --- |
| `gemini_subs.py` | CLI entry point: dotenv loading, argument parsing, validation, and dispatch. |
| `modules/core.py` | Core schemas, timestamp handling, source titles, context URL policy, caption validation, cue classification, speaker label casing canonicalization, sparse audio-patch reconstruction and validation, and pure-editorial boundary merging. |
| `modules/io.py` | Atomic JSON and VTT publication and manifest file I/O. |
| `modules/media.py` | FFmpeg and FFprobe operations for probing, complete audio extraction, and stream-copy splitting. |
| `modules/gemini.py` | Gemini clients, prompts, request configs, chunk requests, boundary audio refinement, and global text refinement. |
| `modules/pipeline.py` | Generation configuration, locking, scheduling, stitching, and the run lifecycle. |
| `scripts/benchmark.py` | Full-video matrix benchmark across generation, audio refinement, and text refinement models. |
| `scripts/subtitle.sh` | Wrapper script that runs generation with terminal context URL prompting. |
| `scripts/yt-dl.sh` | YouTube downloader that fetches VP9/WebM video via `yt-dlp`. |
| `scripts/ffmpeg.sh` | Headless GPL static FFmpeg and FFprobe installer for Linux hosts. |
| `tests/` | Behavioral test suite organized by module owner. |
| `AGENTS.md` | Authoritative behavioral and development instructions for agents. |
| `CONTEXT.md` | Complete architectural specification and domain glossary. |
| `README.md` | User-facing installation, usage, and development guide. |
| `pyproject.toml` | Python project metadata, tool configuration, and dependency declarations. |
| `uv.lock` | Exact dependency lockfile managed by `uv`. |
| `.env.example` | Environment variable template for credentials and model defaults. |
| `.gitignore` | Ignores credentials, caches, generated media, work directories, and subtitle outputs. |
| `.python-version` | Pins the Python interpreter version for `uv`. |
| `docs/agents/` | Agent skill instructions for issue tracking, triage labels, and domain documentation. |

Keep the five modules in `modules/` on an acyclic dependency graph:
- `modules/core.py` and `modules/io.py` are foundations with no project-internal imports.
- `modules/media.py` depends only on `io`.
- `modules/gemini.py` depends on `core`, `io`, and `media` (for `AUDIO_MIME_TYPE`).
- `modules/pipeline.py` orchestrates media and Gemini and owns the run lifecycle.
- `gemini_subs.py` stays CLI-only: dotenv loading, argument parsing, validation, and dispatch.

## Runtime requirements

The project requires Python 3.14 or newer.
Use `uv` for dependency installation and tool execution.

Runtime dependencies:
- `google-genai>=2.0.0` for Gemini API access.
- `pydantic>=2.13.4` for response schemas and validation.
- `python-dotenv>=1.2.2` for `.env` loading.
- `webvtt-py>=0.5.1` for VTT reading and writing.

The CLI requires `ffmpeg` and `ffprobe` in `PATH`.
Install headless GPL static binaries via `./scripts/ffmpeg.sh`.

## Configuration

`python-dotenv` loads `.env` when `gemini_subs.py` starts.
The shell environment and CLI arguments can also provide these values.
CLI arguments take precedence over environment variables.

| Variable | CLI option | Default | Use |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | `--api-key` | None | Required Gemini credential. |
| `GEMINI_API_BASE` | `--base-url` | SDK default | Optional Gemini-compatible proxy base URL. |
| `GEMINI_MODEL` | `--model` | `gemini-3.7-flash` | Chunk video model. |
| `GEMINI_AUDIO_REFINE_MODEL` | `--audio-refine-model` | `gemini-3.7-flash` | Boundary audio refinement model. |
| `GEMINI_REFINE_MODEL` | `--refine-model` | `gemini-3.1-pro-preview` | Full-script refinement model. |

The chunk thinking level accepts `minimal`, `low`, `medium`, and `high`.
It defaults to `high`.
`minimal` is valid only when the model name contains `flash` (case-insensitive).
The text refinement pass always uses `medium`.
The audio refinement pass always uses `high`.

Direct CLI defaults:
- `--chunk-dur 60` seconds.
- `--workers 7` API workers.
- `--output output_subtitles.vtt`.
- Audio refinement enabled.
- Text refinement enabled.
- No `--context-url` values.

`--disable-audio-refine` disables boundary audio repair.
`--disable-text-refine` disables global text refinement.
`--refine-only` runs global text refinement directly on an input VTT file.

## Artifact publication matrix

Generation publishes one artifact based on the refinement toggles:

| Audio refinement | Text refinement | Published input |
| --- | --- | --- |
| Enabled (default) | Enabled (default) | Text-refined `audio_refined.vtt` |
| Enabled | Disabled (`--disable-text-refine`) | `audio_refined.vtt` |
| Disabled (`--disable-audio-refine`) | Enabled | Text-refined `stitched.vtt` |
| Disabled (`--disable-audio-refine`) | Disabled (`--disable-text-refine`) | `stitched.vtt` |

## Video support

`probe_video_format()` asks FFprobe for the primary video codec:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 <path>
```

Supported codecs:

| Primary codec | Chunk extension | MIME type |
| --- | --- | --- |
| VP9 | `.webm` | `video/webm` |
| H.264 | `.mp4` | `video/mp4` |
| HEVC/H.265 | `.mp4` | `video/mp4` |

Unsupported codecs like AV1 fail with `RuntimeError`.
Audio is mapped with `-map 0:a?` when present.
Subtitle streams are excluded from generated chunks with `-sn`.

## Complete audio extraction

Audio refinement always selects the first audio stream, `0:a:0`.
When audio refinement is enabled, the pipeline probes for `a:0` while holding the work lock.
The run fails before splitting or API calls when the source has no audio stream.

Extraction command:

```bash
ffmpeg -y -i <video_file> -map 0:a:0 -vn -sn -c:a libopus -b:a 64k -ac 1 -ar 48000 -f ogg extracted_audio.ogg.tmp
```

The temporary file is published atomically with `os.replace()`.
A cached `extracted_audio.ogg` is reused only when FFprobe reports one mono Opus stream at 48 kHz, a positive finite duration, and a duration within 2.0 seconds of the source media duration.
Corrupt or inconsistent audio is regenerated.

## Chunking and segmentation

Stream-copy splitting runs:

```bash
ffmpeg -y -i <video_file> -map 0:v:0 -map 0:a? -sn -c copy \
  -f segment -segment_time <chunk_dur> \
  -segment_list segments.csv -reset_timestamps 1 \
  chunk_%03d<ext>
```

Stream copy cuts at keyframes.
`segments.csv` is the source of truth for chunk start and end times.
Each row contains `chunk_NNN.<ext>,start,end`.
When any nonblank row is malformed or has invalid timestamps, the split is regenerated.
A reusable index must match stored `chunk_NNN` files exactly.

### Chunk generation retry

Each chunk worker makes up to 3 attempts per chunk: one initial attempt plus up to 2 retries.
A retry happens only after a transient failure:
- Response parsing or caption validation fails, for example malformed JSON, a schema violation, inverted timestamps, or a collapsed interval.
- The Gemini SDK raises `ServerError` with code 500, 502, 503, or 504.

A retry waits 1 second after the first attempt and 2 seconds after the second.
A successful attempt publishes the chunk JSON atomically.
Permanent failures do not retry.
Examples are 401 and 403 responses, a missing chunk file, and unexpected exceptions.
The chunk fails immediately.
When every attempt fails, the worker logs the final error and publishes no JSON.
The run stays resumable because valid published chunks are skipped on the next run.

## Structured data schemas

### Chunk generation: `SubtitleResponse`

```python
class Caption(BaseModel):
    id: int
    start: str
    end: str
    text: str


class SubtitleResponse(BaseModel):
    captions: list[Caption]
```

### Boundary audio refinement: `AudioRefinementResponse` (`sparse-patch-v1`)

```python
class AudioRefinedCue(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    source_ids: list[int] = Field(alias="sourceIds")
    start: str
    end: str
    text: str


class AudioRefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    contract_version: Literal["sparse-patch-v1"] = Field(alias="contractVersion")
    deleted_source_ids: list[int] = Field(
        default_factory=list, alias="deletedSourceIds"
    )
    cues: list[AudioRefinedCue]
```

Wire fields use camelCase (`contractVersion`, `deletedSourceIds`, `sourceIds`).
The JSON schema sent to Gemini strips unsupported `additionalProperties` keywords while host-side validation keeps `extra="forbid"`.

### Global text refinement: `RefinementResponse`

```python
class RefinedCaption(BaseModel):
    id: int
    text: str


class RefinementResponse(BaseModel):
    changes: list[RefinedCaption]
```

## Timestamp rules

Accepted input shapes: `SS`, `MM:SS`, or `HH:MM:SS`.
The fractional-second part is optional in each shape.
Decimal commas are converted to decimal points.
Output timestamps always use `HH:MM:SS.mmm` rounded to the nearest millisecond.

`validate_captions()`:
- Rejects duplicate IDs and non-positive intervals (`end <= start` or `start < 0`).
- Clamps end overruns up to 0.5s beyond chunk duration.
- Rejects cues that collapse to non-positive intervals after rounding.
- Sorts captions by start time and ID.
- Overlapping cues keep their canonical timing.

## Derivation and classification algorithms

### Source title derivation (`derive_source_title`)
1. Basename check: strip subtitle extension (`.vtt`, `.srt`, `.sub`, `.sbv`). If a trailing language tag matches `^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$`, strip it as well.
2. Strip media extension (`.webm`, `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`).
3. Return trimmed title.

### Context URL policy (`validate_context_urls`, `classify_context_urls`)
- Each URL must be an absolute HTTP/HTTPS URL with a host.
- Deduplication preserves first occurrence.
- YouTube watch URLs (`youtube.com`, `www.youtube.com`, `m.youtube.com` with `/watch` and `v` param) and share URLs (`youtu.be` with one path segment) route to direct video analysis.
- All other URLs route to Google URL Context tool.

### Cue classification (`classify_cue_text`)
- Strip bracketed fragments `\[[^\]]*\]`.
- If no brackets exist in text: `dialogue`.
- If word characters remain outside brackets: `mixed`.
- If only bracketed fragments and whitespace exist: `editorial`.

### Speaker label pattern
`SPEAKER_LABEL_RE` matches `^([ \t]*)([A-Z][\w' -]{1,30})(:[ \t]*)` from the start of a cue line.
The pattern captures leading indentation, the label, and the colon with its trailing spacing.

### Speaker label casing canonicalization

`canonicalize_speaker_casing(vtt, grounded_names=None)` rewrites each speaker label to one canonical spelling per case-insensitive label group.
It accepts a `webvtt.WebVTT` object or a list of `webvtt.Caption` objects, and it modifies the object in place.

The algorithm runs two passes over non-editorial cues:

1. Collection: every line matching `SPEAKER_LABEL_RE` adds one occurrence to its casefolded label group.
2. Rewrite: every matched label becomes its group's canonical spelling.
   Leading indentation, the colon with its trailing spacing, and the remaining line text stay unchanged.

Canonical spelling selection per group:

1. A grounded name from `grounded_names` whose casefold equals the group key wins over script frequency.
2. Otherwise the most frequent script spelling wins.
3. Exact frequency ties keep the first spelling seen in the script.

Publication boundary:

- `gemini.global_refine_subtitles()` extracts grounded names from the identity research section, merges them with its `grounded_names` argument, and canonicalizes before it saves the refined VTT atomically.
- `pipeline.run_generation()` canonicalizes without grounded names when it publishes the final artifact directly without text refinement.

## Stitching and pure-editorial boundary merging

Stitching offsets chunk-relative timestamps by each segment's actual `start` from `segments.csv` and sorts by start time.

`merge_visual_boundary_fragments()` merges exact pure-editorial cues split at keyframe cuts:
- Adjacent owner chunks (`chunk_idx` and `chunk_idx + 1`).
- Both classify as `editorial` (every line is `[Text]`).
- Trimmed texts match exactly.
- First cue ends within 0.5s of the boundary; second starts within 0.5s of the boundary.
- Merged cue spans from first start to second end with the later owner index.
- Merging repeats until stable. Dialogue and mixed cues never merge.

## Boundary-limited audio refinement

Repairs boundary faults by listening to `extracted_audio.ogg`.

Request configuration:
- `DEFAULT_AUDIO_REFINE_MODEL = "gemini-3.7-flash"`
- Fixed thinking level: `HIGH`
- Max output tokens: `65536`
- Response MIME: `application/json` with `AudioRefinementResponse` schema.
- Automatic function calling disabled explicitly via `build_content_config()`.

### Prompt structure
- Header with audio duration and segment boundaries.
- Numbered script entries: `[0] 00:00:00.000 --> 00:00:05.000 [dialogue]: Text`.
- Boundary rules:
  - Outside repair regions (5s before to 5s after boundaries), source cues must survive identically.
  - Rewrites, retimes, splits, and merges may reference only cues intersecting a shared repair region.
  - Recovered cues must have empty lineage (`sourceIds: []`), contain spoken dialogue, have no brackets, and fit inside a repair region.
  - Pure editorial cues must be preserved identically.
  - Mixed cues must preserve their bracketed fragments across descendants.
  - Deletions are allowed only for dialogue cues inside repair regions.

### Candidate reconstruction and publication
1. Host expands the sparse patch by inserting exact copies of omitted source cues.
2. Validates envelope containment, visual fragment multiset equality, and timestamp validity.
3. Serializes to `audio_refined.vtt.tmp`, reads back to verify cue equality, and publishes with `os.replace()`.

## Global text refinement

Refines the full script using grounded identity and terminology research.

1. **Grounded web research pass:**
   - Tool: Google Search + optional URL Context.
   - Output: Plain text with two sections:
     - `PARTICIPANTS AND SPEAKERS`: canonical English public names, aliases, and roles for the people who speak in the video, each entry on its own line starting with the canonical name or stable role followed by a colon.
     - `TOPIC TERMINOLOGY AND PROPER NOUNS`: canonical English spelling of recurring proper nouns, program or series titles, organization names, product names, and locations referenced in the source title or context URLs.
   - Grounded research establishes canonical spelling and verified entities only; it never infers, invents, or alters spoken dialogue content, meaning, or events.
   - Grounding verification: Research stream must contain non-empty search queries, grounded sources, and successful URL retrieval.
2. **Direct YouTube video pass:**
   - Attached video Parts for public YouTube URLs.
   - Retries transient 500, 502, 503, and 504 server errors up to 3 times with exponential backoff.
3. **Structured refinement pass:**
   - Model: `gemini-3.1-pro-preview` (thinking level `medium`, temp `0.0`).
   - Input: Full script with separate `GROUNDED IDENTITY CONTEXT` and `GROUNDED TERMINOLOGY CONTEXT` blocks.
   - Section splitting: The host splits the research text into identity and terminology sections at the section headers. Header matching is case-insensitive with an optional trailing colon, and text before the first header stays in the identity section, so research output without headers keeps working as identity context.
   - Tasks:
     1. Speaker label auditing (intro/title card > web/YouTube research > source title).
     2. Speaker label prefixes (`Name:`) identify who is speaking using the established canonical English name or role; spoken dialogue text stays faithful to the spoken audio and never alters spoken names, nicknames, titles, or address terms merely to match a speaker label prefix.
     3. Terminology consistency: use the grounded terminology context for canonical spelling of proper nouns, series and program titles, organizations, and location names.
     4. English polishing, idiom localization, formatting cleanup.
     5. Forbids retiming, merging, splitting, adding, or deleting cues.
   - Output: `RefinementResponse` applied atomically to target.
4. **Speaker label casing canonicalization:**
   - `core.canonicalize_speaker_casing(vtt, grounded_names)` runs deterministically after the model changes and before the atomic save.
   - `gemini.global_refine_subtitles()` extracts the canonical name entries (leading `Name:` lines) from the identity research section and merges them with its `grounded_names` argument; caller-supplied names win case-insensitive collisions.

## Work state, locking, and recovery

Work directory: `temp_video_chunks/<manifest-sha256-prefix>/` (first 16 hex chars).

Manifest contains:
- `video`: `{path, size, mtime_ns}`
- `chunk_dur`: requested duration
- `format`: `stream-copy-v1`
- `mode`: `generate`
- `model`: chunk model
- `chunk_thinking_level`: thinking level
- `chunk_ext`, `chunk_mime`, `video_codec`

Locking:
- Exclusive non-blocking POSIX `fcntl.flock()` on `.lock`.
- PID written to lock file for diagnostics.
- Released automatically on process exit.

Atomic writes:
- Chunk JSON uses `.tmp` sibling + `os.replace()`.
- Extracted audio uses `.tmp` sibling + `os.replace()`.
- VTT publication uses `io.atomic_save_vtt()` (`.{name}.<random>.tmp.vtt` + `os.replace()`).

Recovery on retry:
- Valid `segments.csv` and chunk files skip splitting.
- Valid `subtitle_chunk_NNN.json` files skip chunk API calls.
- Valid `extracted_audio.ogg` (mono Opus 48kHz within duration tolerance) skips extraction.
- Valid `audio_refinement.json` matching cache identity skips audio refinement.
- Successful runs clean intermediate work files while holding lock.

## Benchmark runner (`scripts/benchmark.py`)

Runs matrix benchmarking across 3 passes:
- `--case GEN:AUDIO:REFINE`: 3-tuple model cases.
- `--model`: Independent chunk-only models.
- `--context-url`: Grounding URLs for text refinement.
- `--reference-vtt`: Optional reference subtitles for comparison.
- Outputs `benchmark_results/benchmark-results.json` and stage VTTs.

### Comparison metrics
- Text normalization: strip speaker label `^\s*[A-Z][\w' -]{1,30}:\s*`, strip punctuation, lowercase words.
- `text_similarity`: `SequenceMatcher(None, ref_words, gen_words).ratio()`.
- `temporal_recall`: `overlap_seconds / reference_seconds`.
- `temporal_precision`: `overlap_seconds / generated_seconds`.
- `temporal_iou`: `overlap_seconds / (ref_seconds + gen_seconds - overlap_seconds)`.

## Glossary

**Audio-first boundary refinement**:
Repairs chunk-boundary faults by listening to the complete extracted audio track.

**Boundary envelope**:
A repair region expanded by the full original time extents of referenced source cues.

**Chunk**:
A contiguous stream-copy segment cut from the source video at a keyframe.

**Pure-editorial boundary merge**:
Stitch pass merging identical pure-editorial cues split across adjacent chunk boundaries.

**Repair region**:
The connected union of 5 seconds before through 5 seconds after segment boundaries.

**Segment boundary**:
The source timestamp where one chunk ends and the next begins.

**Sparse patch (`sparse-patch-v1`)**:
Response contract returning only changed, replacement, split, merged, or recovered cues plus deleted source IDs.
