# Test contracts

The test suite is organized around the production module boundaries in `modules/`.
Each test imports behavior from its owning module.
`gemini_subs.py` keeps no compatibility re-exports.

## Test layout

| Path | Intended owner | Scope |
| --- | --- | --- |
| `tests/modules/core/` | `modules/core.py` | Subtitle payloads, timestamp semantics, source titles, context URLs, and boundary deduplication. |
| `tests/modules/io/` | `modules/io.py` | Atomic JSON and VTT publication and manifest file I/O. |
| `tests/modules/media/` | `modules/media.py` | Real FFmpeg and FFprobe probing, splitting, stream mapping, windows, overlap clips, and split cleanup. |
| `tests/modules/gemini/` | `modules/gemini.py` | Gemini request contracts, caption cache, chunk generation, and refinement outcomes. |
| `tests/modules/pipeline/` | `modules/pipeline.py` | Configuration policy, persisted run state, locking, scheduling, stitching, recovery, and cleanup. |
| `tests/cli/` | `gemini_subs.py` | Subprocess argument parsing, validation, exit status, and diagnostics. |
| `tests/support/` | Test infrastructure | Scenario fakes and persistent-state builders shared by behavioral tests. |

## Automated contracts

| Area | Observable contract |
| --- | --- |
| Core | Accepted timestamp forms, exact millisecond rounding, invalid interval rejection, caption ordering, end clamping, payload validation, multiline text preservation, source-title derivation, context URL validation and classification, and boundary deduplication. |
| Io | Atomic JSON and VTT publication with failure-safe target preservation. |
| Media | Supported codec probing, unsupported input rejection, decodable stream-copy chunks, audio retention, subtitle exclusion, segment index validation, split recovery, edge-clamped windows, decodable overlap clips for VP9, H.264, and HEVC, and split artifact cleanup. |
| Gemini | Client boundary translation, outbound request configuration, tool routing, structured response validation, streamed text assembly, direct YouTube inputs, caption cache behavior, text-only refinement, SDK failure propagation, and failure-safe output. |
| Pipeline | Runtime validation, resumable state selection, valid cache reuse, corrupt-state regeneration, process locking, concurrent filesystem outcomes, midpoint ownership, stitch provenance, final publication, recovery, and failed-run retention. |
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
- Nested Gemini SDK response, candidate, grounding, and URL metadata shapes.
- Gemini SDK event normalization into project-owned stream metadata.
- Direct helper outputs that executable adapter and lifecycle tests already cover.
- Thread identities, executor details, and other concurrency mechanisms.
- Internal staging filenames and directories.

Review these details against `AGENTS.md` when their implementation changes.
Do not add source-text, prompt snapshot, complete command-array, private-call, helper-name, or provider event-shape assertions.

## Commands

Run the complete suite:

```bash
uv run pytest
```

Run one intended module area:

```bash
uv run pytest tests/modules/core
uv run pytest tests/modules/io
uv run pytest tests/modules/media
uv run pytest tests/modules/gemini
uv run pytest tests/modules/pipeline
uv run pytest tests/cli
```
