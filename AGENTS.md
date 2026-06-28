# Project Context for AI Agents

This file provides critical context, architectural decisions, and strict rules for any AI assistant working on this repository. Read this before making changes to ensure consistency and prevent regressions.

## Goal
A high-performance Python CLI tool that uses the Google Gemini API to generate completely new English subtitles from scratch. It optimizes for end-to-end throughput and high-quality syllable-level timing.

## Architecture
1. **Chunking**: FFmpeg stream-copies the video into chunks (default 60s) to bypass Gemini inline video limits and proxy timeouts.
2. **Context Overlaps**: Temporary clips are re-encoded (default 5s overlap) to give the model context across chunk boundaries, preserving the input codec family for VP9, H.264, and HEVC/H.265.
3. **Parallel Processing**: Video chunks are sent concurrently to the Gemini API (`gemini-3.1-pro-preview` by default). Overlap clip creation is auto-sized from CPU count.
4. **Structured Output**: Uses `google-genai` SDK and Pydantic schemas (`SubtitleResponse`) to guarantee valid JSON returns.
5. **Stitching & Healing**: Validates timestamps, auto-heals any overlapping cues by nudging boundaries, and stitches chunks back into a final `.vtt`.
6. **Global Refinement**: A second, full-script Gemini pass (`RefinementResponse`) fixes character names, continuity, and grammar without altering the generated timestamps. Saves atomically to prevent corruption.

## Current Defaults
- Model: `gemini-3.1-pro-preview`
- Common base URL: `https://main.your-proxy-domain.com/google/v1beta`
- Chunk duration: `60`
- Overlap: `5`
- Overlap format: derived from input codec (`webm` for VP9, `mp4` for H.264 and HEVC/H.265)
- Chunk thinking level: `minimal` for Flash models, `low` otherwise
- Global refinement thinking level: `high`
- Global refinement: enabled by default

## Strict Rules & Semantics

### 1. Timing Accuracy (CRITICAL)
- **Syllable-level**: For spoken dialogue, `start_time` must be the exact millisecond of the first audible syllable. `end_time` must be the exact end of the last audible syllable.
- **Silence**: Silent gaps between sentences MUST remain real gaps. Do not arbitrarily stretch durations to fill silence.
- **Visual Text**: Editor/flair text must be timed EXACTLY to when it appears and disappears on screen.

### 2. Localization Preferences
- Preserve native cultural terms, foods, and nicknames instead of forcing westernized localization.
- Do not summarize, explain, or infer missing dialogue.

### 3. State & Caching
- **Resumability**: The script writes a `manifest.json` tracking inputs, derived codec/overlap settings, and model configs. Failed runs keep the chunk directory to resume instantly; successful runs remove temporary chunks.

### 4. SDK & Dependencies
- Uses the modern `google-genai` (>= 2.0.0) SDK, NOT the legacy `google-generativeai` package.
- Uses `uv` for dependency management (`uv run`). 
- Schema uses strictly typed Pydantic models.
- Uses Ruff for formatting and lint checks.
- Prefer the Python standard library for tests and small utilities unless a new dependency clearly pays for itself.
- For system dependencies (like FFmpeg/FFprobe), explicitly recommend and document headless/X11-free GPL static builds (from BtbN/FFmpeg-Builds) in environments like WSL or Ubuntu Server to avoid pulling in graphical dependencies.

### 5. Code Style
- Keep `gemini_subs.py` as a single, focused file unless it grows completely unmanageable.
- Avoid adding heavy dependencies (like `moviepy`) when a simple `subprocess` FFmpeg call suffices. 
- Use atomic writes (`.tmp` file renaming) for all final file outputs to prevent corruption on user interrupt.

### 6. Helper Scripts
- Helper scripts live in `scripts/` and must be directly executable as `./scripts/<name>` from the repo root.
- `scripts/subtitle.sh` is the preferred local wrapper for generating subtitles with the repository's tuned worker settings.
- `scripts/yt-dl.sh` downloads YouTube videos as best available VP9 video plus best audio, falling back to best WebM when VP9 is unavailable.
- `scripts/ffmpeg.sh` installs or upgrades the latest BtbN GPL static FFmpeg build into `~/.local/bin` for linux64 and linuxarm64 hosts. It always downloads the current `latest` archive.
- `scripts/benchmark.py` times one overlap clip generation and one real Gemini request, then suggests a `scripts/subtitle.sh` worker count.
- Keep shell scripts compatible with `shellcheck` and explicit about required arguments.

### 7. Validation
Run only the validation commands relevant to the files changed in the current task. Do not run unrelated checks for extra safety unless the user explicitly asks.
- **If only docs or instructions are modified** (`*.md`, including `AGENTS.md` and `README.md`): No code validation is required.
- **If shell scripts in `scripts/*.sh` are modified**: Run `shellcheck` only on the modified shell script(s) (e.g., `shellcheck scripts/subtitle.sh`).
- **If `gemini_subs.py` is modified**: Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run python -m compileall -q .`.
- **If core behavior in `gemini_subs.py` or files in `tests/` are modified**: Run `uv run python -m unittest discover -s tests`.
- **If CLI arguments in `gemini_subs.py` are modified**: Run `uv run gemini_subs.py --help`.
- **If `scripts/benchmark.py` is modified**: No tests or validation are required unless the user explicitly asks.

### 8. Git Workflow
- When the user requests per-task commits, commit each discrete task before starting the next one.
- Before committing, inspect `git status`, `git diff`, and recent commits; stage only files that belong to the current task.
