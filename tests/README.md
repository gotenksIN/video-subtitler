# Test contracts

The test suite is organized around the intended production module boundaries.
Until production extraction, tests import behavior from `gemini_subs.py`.
After extraction, update imports to the owning module without preserving compatibility exports in `gemini_subs.py`.

## Test layout

| Path | Intended owner | Scope |
| --- | --- | --- |
| `tests/modules/core/` | `modules/core.py` | Subtitle payloads, timestamp semantics, source titles, and caption validation. |
| `tests/modules/media/` | `modules/media.py` | Real FFmpeg and FFprobe probing, splitting, stream mapping, windows, and overlap clips. |
| `tests/modules/gemini/` | `modules/gemini.py` | Gemini request contracts, streamed responses, grounding, chunk generation, and refinement. |
| `tests/modules/pipeline/` | `modules/pipeline.py` | Configuration policy, persisted run state, scheduling, stitching, publication, recovery, and cleanup. |
| `tests/cli/` | `gemini_subs.py` | Subprocess argument parsing, validation, exit status, and diagnostics. |
| `tests/support/` | Test infrastructure | Scenario fakes and persistent-state builders shared by behavioral tests. |

## Automated contracts

| Area | Observable contract |
| --- | --- |
| Core | Accepted timestamp forms, exact millisecond rounding, invalid interval rejection, caption ordering, end clamping, payload validation, multiline text preservation, and source-title derivation. |
| Media | Supported codec probing, unsupported input rejection, decodable stream-copy chunks, audio retention, subtitle exclusion, segment timing, split recovery, edge-clamped windows, and decodable overlap clips for VP9, H.264, and HEVC. |
| Gemini | Client and request configuration, tool routing, structured response validation, streamed response assembly, grounding requirements, URL retrieval status, direct YouTube inputs, text-only refinement, and failure-safe output. |
| Pipeline | Runtime validation, URL classification, resumable state selection, valid cache reuse, corrupt-state regeneration, process locking, concurrent outcomes, midpoint ownership, boundary deduplication, provenance, staging, atomic publication, cleanup, and failed-run retention. |
| CLI | Help, parsing errors, validation precedence, environment configuration, mode-specific preflight behavior, stable diagnostic meaning, and exit status. |

Tests assert returned values, files, parsed WebVTT captions, persisted state, diagnostics, exit status, and requests sent across external API boundaries.
Tests do not assert private helper calls or internal execution order unless ordering changes an observable contract.

## Test boundaries

Media integration tests generate tiny session-scoped fixtures with real `ffmpeg` and `ffprobe`.
The suite fails clearly when required binaries or VP9, H.264, and HEVC encoders are unavailable.

Gemini tests use stateful scenario fakes and never contact the live service.
Failure scenarios occur at the process, API, or publication boundary and verify resulting files and resumable state.

CLI tests run `gemini_subs.py` in subprocesses.
They scrub Gemini credentials and use an unreachable base URL to prevent accidental network requests.
Successful generation and refinement lifecycle behavior is tested through the pipeline interface.

## Review-only specifications

The following details remain specified in `AGENTS.md` but are not frozen by automated tests:

- Prompt prose and required prompt instructions.
- Manifest JSON representation and hash construction.
- Exact FFmpeg argument arrays.
- Worker-count and thread-allocation formulas.

Review these details against `AGENTS.md` when their implementation changes.
Do not add source-text, prompt snapshot, complete command-array, private-call, or helper-name assertions.

## Commands

Run the complete suite:

```bash
uv run pytest
```

Run one intended module area:

```bash
uv run pytest tests/modules/core
uv run pytest tests/modules/media
uv run pytest tests/modules/gemini
uv run pytest tests/modules/pipeline
uv run pytest tests/cli
```
