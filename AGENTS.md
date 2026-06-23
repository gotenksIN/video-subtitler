# Project Context for AI Agents

This file provides critical context, architectural decisions, and strict rules for any AI assistant working on this repository. Read this before making changes to ensure consistency and prevent regressions.

## Goal
A high-performance Python CLI tool that uses the Google Gemini API to either align existing VTT subtitles to a video (fixing timings) or generate completely new English subtitles from scratch. It optimizes for end-to-end throughput and high-quality syllable-level timing.

## Architecture
1. **Chunking**: FFmpeg stream-copies the video into chunks (default 60s) to bypass Gemini inline video limits and proxy timeouts.
2. **Context Overlaps**: Temporary clips are re-encoded (default 5s overlap) to give the model context across chunk boundaries, ensuring accurate midpoint boundary assignments.
3. **Parallel Processing**: Video chunks are sent concurrently to the Gemini API (`gemini-3.1-pro-preview` by default).
4. **Structured Output**: Uses `google-genai` SDK and Pydantic schemas (`AlignmentResponse`) to guarantee valid JSON returns.
5. **Stitching & Healing**: Validates timestamps, auto-heals any overlapping cues by nudging boundaries, and stitches chunks back into a final `.vtt`.
6. **Global Refinement**: A second, full-script Gemini pass (`RefinementResponse`) fixes character names, continuity, and grammar without altering the aligned timestamps. Saves atomically to prevent corruption.

## Current Defaults
- Model: `gemini-3.1-pro-preview`
- Common base URL: `https://main.your-proxy-domain.com/google/v1beta`
- Chunk duration: `60`
- Overlap: `5`
- Overlap format: `mp4`
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
- **Resumability**: The script writes a `manifest.json` tracking inputs, overlap settings, and model configs. Failed runs keep the chunk directory to resume instantly.

### 4. SDK & Dependencies
- Uses the modern `google-genai` (>= 2.0.0) SDK, NOT the legacy `google-generativeai` package.
- Uses `uv` for dependency management (`uv run`). 
- Schema uses strictly typed Pydantic models.
- Prefer the Python standard library for tests and small utilities unless a new dependency clearly pays for itself.
- For system dependencies (like FFmpeg/FFprobe), explicitly recommend and document headless/X11-free GPL static builds (from BtbN/FFmpeg-Builds) in environments like WSL or Ubuntu Server to avoid pulling in graphical dependencies.

### 5. Code Style
- Keep `gemini_subs.py` as a single, focused file unless it grows completely unmanageable.
- Avoid adding heavy dependencies (like `moviepy`) when a simple `subprocess` FFmpeg call suffices. 
- Use atomic writes (`.tmp` file renaming) for all final file outputs to prevent corruption on user interrupt.

### 6. Helper Scripts
- `subtitle.sh` is the preferred local wrapper for generating subtitles with the repository's tuned worker settings.
- `yt-dl.sh` downloads YouTube videos as best available VP9 video plus best audio, falling back to best WebM when VP9 is unavailable.
- `embed-subs.sh` embeds VTT subtitles into a video as a soft subtitle track using `ffmpeg -c copy` (no re-encode).
- Keep shell scripts compatible with `shellcheck` and explicit about required arguments.

### 7. Validation
- Run `shellcheck subtitle.sh yt-dl.sh embed-subs.sh` after changing shell scripts.
- Run `uv run python -m compileall -q .` after Python changes.
- Run `uv run python -m unittest discover -s tests` after code or validation logic changes.
- Run `uv run gemini_subs.py --help` after CLI argument changes.

### 8. Git Workflow
- When the user requests per-task commits, commit each discrete task before starting the next one.
- Before committing, inspect `git status`, `git diff`, and recent commits; stage only files that belong to the current task.
