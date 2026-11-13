# Test contracts

The test suite verifies pipeline behavior end-to-end first and isolates pure domain validators only where combinatorial rules make end-to-end setup impractical.
Keep the suite compact (45 to 70 collected items) so `uv run pytest` finishes quickly.

## Test layout

| Path | Scope |
| --- | --- |
| `tests/conftest.py` | Session-scoped tiny FFmpeg media fixtures for H.264, VP9, HEVC, silent video, and embedded subtitles. |
| `tests/support/gemini_fakes.py` | Stateful `ScriptedGeminiClient` driving real `google.genai.Client` instances over `httpx2.MockTransport`. |
| `tests/test_e2e_pipeline.py` | End-to-end 4-pass `pipeline.run_generation` flows, real SDK and wire schema checks, refinement toggles, single-chunk boundary skip, partial-failure resume, cache invalidation, work-directory locking, stream/validation retries, and `scripts/benchmark.py` coverage. |
| `tests/test_cli.py` | `gemini_subs.py` subprocess help, environment precedence, preflight validation errors, exit codes, and `--refine-only` execution. |
| `tests/test_media_io.py` | Real FFmpeg and FFprobe codec probing, stream-copy splitting, subtitle exclusion, Opus audio extraction, corrupt cache regeneration, atomic JSON/VTT failure safety, HTML entity round-trips, and tagged-VTT rejection. |
| `tests/test_audio_patch.py` | Combinatorial `sparse-patch-v1` repair-region filtering, lineage order and contiguity, pure-editorial and mixed-fragment preservation, overlap rules, and audio/video end clamping. |
| `tests/test_text_contracts.py` | Pass 3 structural-edit guards, speaker-label casing precedence, pure-editorial boundary merging, caption and timestamp validation, source-title derivation, and context URL classification. |

## Automated contracts

| Capability | Observable contract |
| --- | --- |
| End-to-end pipeline | Pass 0 concurrent Search, URL Context, and YouTube analysis overlapping FFmpeg split and Opus extraction; grounded-name forwarding; Pass 1 concurrent chunk generation, thought-chunk exclusion, and stitch offset/banner merging; Pass 2 sparse audio repair; Pass 3 grounded refinement and speaker casing; toggle matrix; single-chunk Pass 2 skip; partial-failure retention and cache-backed resume; advisory lock contention and failure settlement; real SDK `HttpRetryOptions`, disabled AFC, and sanitized wire `responseSchema` checks; benchmark runner execution and reference metrics. |
| CLI | Usage help, exit status (`0`, `1`, `2`), stderr diagnostics without tracebacks, `.env` and environment variable overrides (including empty refinement fallback vars), preflight output-parent and URL checks before media or API work, and `--refine-only` behavior. |
| Media and I/O | VP9, H.264, and HEVC probing and stream-copy splitting; unsupported codec and missing audio rejection; embedded subtitle exclusion; corrupt split, `segments.csv`, and Opus cache regeneration; failure-safe atomic JSON and VTT writes; `&`/`<`/`>` entity escaping; unsupported WebVTT tag rejection. |
| Audio patch invariants | Out-of-authority edit/merge/delete/recovery filtering, fatal unknown IDs, monotonic script lineage, editorial-skipping merges and splits, nested and multi-fragment visual multiset equality, crossed-retime sorting and overlap rules, repair-envelope bounds, and `audio_duration` vs `source_duration` grace rules. |
| Text and core contracts | Pass 3 rejection of cue-class, visual-fragment, and turn/label structure mutations; accepted dialogue re-wrapping; speaker casing tie-breaks and grounded overrides; timestamp parsing and millisecond rounding; chunk overrun clamping; source-title suffix stripping; HTTP(S) and YouTube URL classification. |

## Test boundaries and anti-bloat rules

1. **Real media, never fake FFmpeg:** Session-scoped fixtures in `tests/conftest.py` encode tiny `64x64` test videos with real `ffmpeg` and `ffprobe`.
   The suite fails clearly when `ffmpeg`, `ffprobe`, or required VP9, H.264, and HEVC encoders are missing.
   Parametrize across all three codecs only for codec probing and splitting in `tests/test_media_io.py`; use the shared H.264 fixture everywhere else.
2. **Real Gemini SDK with mock HTTP transport, never internal mocks:** `ScriptedGeminiClient` wraps `google.genai.Client` with `httpx2.MockTransport` so every request runs real SDK parameter validation, schema transformation, wire serialization, and response deserialization.
   Never monkeypatch internal `modules/` functions (`_run_preflight_context`, `process_chunk`, `boundary_audio_refine_subtitles`, `global_refine_subtitles`, or `media.*`) to stub pipeline passes.
3. **One boundary per rule:** Test each contract once at the highest practical layer.
   When `tests/test_e2e_pipeline.py` or `tests/test_cli.py` already covers a behavior, do not add duplicate adapter or helper tests.
4. **Equivalence classes over item proliferation:** Fold related valid or invalid inputs into table assertions by equivalence class instead of creating separate pytest items for each extension, URL variant, `argparse` error, or dictionary key.
   Extend an existing scenario instead of adding single-assertion micro-tests.

## Review-only specifications

The following details remain specified in `CONTEXT.md` or `AGENTS.md` and are not frozen by automated tests:

- Prompt prose and required prompt instructions.
- Manifest JSON representation and SHA-256 truncation details beyond observable work-directory isolation.
- Exact FFmpeg and FFprobe command-line argument arrays.
- Thread-pool sizing formulas.
- Internal staging filenames and private helper implementations.

Review these details against `CONTEXT.md` and `AGENTS.md` when changing their implementation.
Do not add source-text, AST, prompt-snapshot, command-array, or private-helper assertions.

## Commands

Run the complete suite:

```bash
uv run pytest
```

Run a single capability file:

```bash
uv run pytest tests/test_e2e_pipeline.py
uv run pytest tests/test_cli.py
uv run pytest tests/test_media_io.py
uv run pytest tests/test_audio_patch.py
uv run pytest tests/test_text_contracts.py
```
