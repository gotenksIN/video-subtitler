# Gemini subtitle generator

A Python CLI that uses the Google Gemini API to generate English WebVTT subtitles from video.
It splits video into stream-copy chunks, generates subtitles concurrently, repairs chunk boundaries against the complete audio, and refines the complete script.

## Features

- **Generation mode:** Provide a video file to create English subtitles with accurate timestamps.
- **Concurrent processing:** Process video chunks in parallel using multiple Gemini API workers.
- **Boundary audio refinement:** Extract the complete audio track and repair dialogue faults near chunk boundaries with a boundary-limited Gemini Flash pass.
- **Structured outputs:** Validate model responses with Pydantic schemas to catch malformed timestamps, duplicate IDs, and invalid chunk output.
- **Grounded refinement:** Run web identity research with Google Search, optional direct YouTube video analysis, and structured script polish.
  Speaker identities use verified evidence instead of appearance guesses.
- **Resumable failures:** Keep temporary work directories on failure so retries reuse valid completed chunks, extracted audio, and refinement caches.
  Clean up temporary work files on success.
- **Safe outputs:** Write chunk JSON, audio, and final WebVTT files atomically to prevent corrupted output.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Python package installer and resolver.
- [FFmpeg](https://ffmpeg.org/) - Install `ffmpeg` and `ffprobe` in your system `PATH`.
  - **Headless and X11-free installation (WSL/Ubuntu Server):** Install a precompiled GPL static build to avoid X11 and GUI dependencies.
    Use the BtbN static build installer for your user binary directory:
    ```bash
    ./scripts/ffmpeg.sh
    ```
    Add `~/.local/bin` to your `PATH` by adding `export PATH="$HOME/.local/bin:$PATH"` to your shell profile.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Optional tool executed automatically with `uvx`.

## Installation

1. Clone this repository.
2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the root directory and add your Gemini API credentials:
   ```env
   GEMINI_API_KEY=your_api_key_here

   # Optional: set a custom base URL or change default models
   GEMINI_API_BASE=https://main.your-proxy-domain.com/google/v1beta
   GEMINI_MODEL=gemini-3.7-flash
   GEMINI_AUDIO_REFINE_MODEL=gemini-3.7-flash
   GEMINI_REFINE_MODEL=gemini-3.1-pro-preview
   ```

## Usage

Run the CLI using `uv run`.

### Helper scripts

Install or upgrade the BtbN GPL static FFmpeg build in `~/.local/bin`:
```bash
./scripts/ffmpeg.sh
```

Download a YouTube video as VP9 video with audio, falling back to WebM when VP9 is unavailable:
```bash
./scripts/yt-dl.sh "https://youtube.com/watch?v=..."
```

Generate subtitles for a local video:
```bash
./scripts/subtitle.sh "your_video.webm"
```

The subtitle helper writes output to `your_video.webm.vtt`.
When standard input is a terminal, it prompts once for an optional context URL for grounded refinement.
Press Enter with a blank answer to skip it.
Noninteractive usage never prompts.

Benchmark subtitle generation and refinement models across full video runs:
```bash
./scripts/benchmark.py "your_video.webm" --case gemini-3.7-flash:gemini-3.7-flash:gemini-3.1-pro-preview
```

Compare generated subtitles against a reference WebVTT file:
```bash
./scripts/benchmark.py "your_video.webm" --model gemini-3.7-flash --reference-vtt "reference.vtt"
```

The benchmark saves generated subtitles and `benchmark-results.json` to the output directory.

### Generation mode

Generate English subtitles from a video:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```

### Audio-first processing and refinement

By default, the pipeline uses `gemini-3.7-flash` for chunk video generation and boundary audio refinement, and `gemini-3.1-pro-preview` for global text refinement.
Chunk processing limits context to 60 seconds per stream-copy chunk.
The audio refinement pass extracts the complete audio track, listens to it, and repairs dialogue faults near chunk boundaries without touching visual on-screen text.
The global text refinement pass corrects inconsistent character names, terminology, and continuity errors without changing timestamps.

Generation publishes one artifact based on the two refinement toggles:

| Audio refinement | Text refinement | Published input |
| --- | --- | --- |
| Enabled (default) | Enabled (default) | Text-refined `audio_refined.vtt` |
| Enabled | `--disable-text-refine` | `audio_refined.vtt` |
| `--disable-audio-refine` | Enabled | Text-refined `stitched.vtt` |
| `--disable-audio-refine` | `--disable-text-refine` | `stitched.vtt` |

Boundary audio refinement sends the complete audio plus the stitched script to Gemini in one streamed JSON request.
The model returns a sparse patch of only changed cues.
The patch is validated so it cannot change cues outside five-second repair windows around chunk boundaries, cannot delete visual on-screen text, and cannot alter bracketed on-screen fragments.

Text refinement has up to three Gemini requests:

1. Grounded web identity research first: a plain-text, streamed request with Google Search grounding.
   It researches participant names in official English styling, roles, and evidence for speaker-label normalization.
   Grounded research may change speaker identity and proper-name spelling only, never dialogue meaning or events.
2. Direct YouTube analysis second, only when you supply YouTube context URLs: a plain-text, streamed request that watches the attached videos without tools.
   It returns participant identities, official names and roles, and timestamped speaker-identification observations.
   Transient Gemini server errors (500, 502, 503, and 504) retry automatically.
3. Structured refinement last: the streamed JSON request with the `RefinementResponse` schema.
   It receives the grounded research text and the YouTube analysis text as identity context and does not use tools.

Text refinement fails before publication when the research response carries no Google Search grounding, when a supplied context URL is not retrieved successfully, or when a YouTube video cannot be retrieved.
The previous output stays intact.
There is no ungrounded fallback.

To skip the boundary audio refinement pass:
```bash
uv run python gemini_subs.py "your_video.webm" --disable-audio-refine
```

To skip the global text refinement pass:
```bash
uv run python gemini_subs.py "your_video.webm" --disable-text-refine
```

To run global text refinement on an existing WebVTT file without video processing:
```bash
uv run python gemini_subs.py "generated_subtitles.vtt" --refine-only -o "polished_subtitles.vtt"
```

### Additional options

- `--disable-audio-refine`: Disable the boundary audio refinement pass after generation.
- `--disable-text-refine`: Disable the global text refinement pass after generation.
- `--refine-only`: Skip video processing and run global text refinement on an input WebVTT file.
- `--chunk-dur`: Video chunk duration in seconds (default: `60`).
- `--workers`: Maximum concurrent API workers (default: `7`).
- `--thinking-level`: Gemini thinking level for chunk video requests (default: `high`).
  Supported levels are `minimal`, `low`, `medium`, and `high`.
  `minimal` requires a Flash model.
  The global text refinement pass always uses `medium`.
  The boundary audio refinement pass always uses `high`.
- `--api-key`: Override `GEMINI_API_KEY` from `.env` or the environment.
- `--base-url`: Override `GEMINI_API_BASE` for a custom Gemini-compatible proxy.
- `--model`: Override `GEMINI_MODEL` for chunk video generation (default: `gemini-3.7-flash`).
- `--audio-refine-model`: Override `GEMINI_AUDIO_REFINE_MODEL` for boundary audio refinement (default: `gemini-3.7-flash`).
- `--refine-model`: Override `GEMINI_REFINE_MODEL` for global text refinement (default: `gemini-3.1-pro-preview`).
- `--context-url`: Absolute HTTP(S) URL used as grounding context for global refinement.
  Repeat the option to supply several URLs.
  Public YouTube watch or share URLs (`youtube.com`, `www.youtube.com`, or `m.youtube.com` with a `/watch` path and nonempty `v` query, or `youtu.be` with exactly one path segment) become direct video inputs for a separate YouTube analysis pass.
  Other URLs use the URL Context tool.
  Refinement fails if any other URL is not retrieved successfully.
  An invalid, private, or unavailable YouTube video fails the analysis request before publication.

## Notes

- The initial split uses stream copy (`-c copy`) and cuts at keyframes, so chunk boundaries follow the source keyframe layout.
  Supported input codecs are VP9, H.264, and HEVC/H.265.
  VP9 chunks use WebM format, while H.264 and HEVC chunks use MP4 format.
- AV1 input is rejected during probing because the processing pipeline supports VP9, H.264, and HEVC/H.265 only.
- Boundary audio refinement requires an audio stream.
  Generation fails before splitting when the source has no audio and audio refinement is enabled.
  Pass `--disable-audio-refine` to generate subtitles for a silent video.
- Keep inline video and audio requests below 20 MiB.
  Reduce `--chunk-dur` if chunk uploads fail.
- When a chunk fails validation or API processing, stitching stops and the work directory is preserved for retry.
  A malformed segment index or missing chunk file invalidates the split and regenerates it on retry.
  Valid extracted audio and audio refinement responses are reused on retry.
  Successful runs clean up the temporary work directory.
- Output WebVTT files are ignored by Git by default.
  Move or rename files to track specific subtitle outputs.

## Development

Production code is organized into modular components under `modules/`.
`modules/pipeline.py` orchestrates generation and `gemini_subs.py` parses and dispatches CLI requests.
`AGENTS.md` is the authoritative behavioral specification and `tests/README.md` documents the test contract matrix.
Tests under `tests/` mirror these module boundaries (`tests/modules/core/`, `tests/modules/io/`, `tests/modules/media/`, `tests/modules/gemini/`, `tests/modules/pipeline/`, and `tests/cli/`).

Run code quality checks and tests:

```bash
# Lint, format, and static analysis
shellcheck scripts/subtitle.sh scripts/yt-dl.sh scripts/ffmpeg.sh
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q .

# Run targeted module tests or the complete test suite
uv run pytest tests/modules/core
uv run pytest tests/modules/io
uv run pytest tests/modules/media
uv run pytest tests/modules/gemini
uv run pytest tests/modules/pipeline
uv run pytest tests/cli
uv run pytest

# Verify CLI entry points
uv run python gemini_subs.py --help
./scripts/benchmark.py --help
```
