# Agent specification

## Domain specification

`CONTEXT.md` is the authoritative technical and architectural specification for this repository.
It describes the pipeline, schemas, authority rules, and domain model from first principles.
Read `CONTEXT.md` before changing code, tests, scripts, or documentation.
Always keep `CONTEXT.md` up-to-date whenever architecture, schemas, pipeline flow, or domain concepts change.

## Development rules

Use ASCII for documentation, code, and comments unless existing content requires another character set.
Keep comments rare and explain non-obvious behavior.
PEP 758 parenthesis-free `except X, Y:` syntax is valid in this repository.
Use atomic output publication for final files.
Do not use legacy `google-generativeai`.

When a Gemini request does not require automatic function calling, construct its config with `build_content_config()` so AFC is disabled explicitly instead of relying on SDK defaults.
This requirement applies to structured, plain-text, tool-enabled, and tool-free requests that do not require automatic function calling.
Verify adapter calls at external boundaries for meaningful application behavior, such as preventing unintended automatic tool execution.
Do not mandate exhaustive schema or request-shape assertions merely as contracts.

Run each invocation from scratch without caching stage results.
Do not change timing semantics casually.
Preserve failed-run artifacts for diagnosis.

Keep the five modules in `modules/` on an acyclic dependency graph:
- `modules/core.py` and `modules/io.py` are foundations with no project-internal imports.
- `modules/media.py` has no project-internal imports.
- `modules/gemini.py` depends on `core`, `io`, and `media` (for `AUDIO_MIME_TYPE`).
- `modules/pipeline.py` orchestrates media and Gemini and owns the run lifecycle.
- `gemini_lyrics.py` stays CLI-only: dotenv loading, argument parsing, validation, and dispatch.

Use semantic line breaks in Markdown prose: put each complete sentence on its own source line.
Use active voice, present tense, ASD-STE100 short sentences, and sentence-case headings.
Do not use exclamation points in documentation.

## Testing principles

Maintain a minimal, robust test suite focused on behavior rather than contracts.
Do not write new tests unless the user or specification explicitly requests them.
Verify observable outcomes and critical failure modes through public or executable interfaces.
Assert outputs, side effects, errors, and externally visible state.
Remove tests that assert only feature presence, registrations, or internal mechanics.
Let the typechecker enforce static type relationships.
Do not add runtime tests that a typecheck alone satisfies.

Verify adapters at integration boundaries for project-owned translation and behavior.
Do not simulate external providers or test external library mechanics.
Do not assert schema shapes, request payload structures, helper names, prompt prose, exact FFmpeg command arrays, manifest representation, or worker formulas.
Verify media handling with real FFmpeg audio inputs and use stateful Gemini scenario fakes at external boundaries.

## Simplicity (YAGNI)

Implement only current, explicit requirements.
Do not add speculative features, abstractions, configuration, dependencies, or extensibility for hypothetical future use.
Prefer the smallest clear change that reuses existing code and standard facilities.
Delete obsolete code when safe.

Before writing a utility or adding a dependency, search the repository for an existing implementation and its callers.
Then check the standard library and already-declared dependencies.
Reuse an established option when it fits.
Ask before adding a new dependency.

## Validation matrix

Run only checks strictly relevant to the changed files.
Never chain full test suites, unchanged tool checks, or multi-command verification runs for small, focused edits.
For documentation or instruction edits, do not run pytest, ruff, compileall, shellcheck, or CLI help commands.

| Changed files | Required checks |
| --- | --- |
| Documentation or instructions only | No code validation. Check Markdown semantics manually. |
| `scripts/*.sh` | `shellcheck <changed-script>` on the changed script only. |
| `modules/core.py` | `uv run pytest tests/modules/core`. |
| `modules/io.py` | `uv run pytest tests/modules/io`. |
| `modules/media.py` | Minimal manual check of audio probing and normalization with real audio. |
| `modules/gemini.py` | `uv run pytest tests/modules/gemini`. |
| `modules/pipeline.py` | `uv run pytest tests/modules/pipeline`. |
| `gemini_lyrics.py` | `uv run pytest tests/cli` and `uv run python gemini_lyrics.py --help`. |
| `tests/` | Run only the specific changed test file. |
| Python production or test files | `uv run ruff check <changed-file>` and `uv run ruff format --check <changed-file>`. |

Run the full pytest suite only when the user explicitly requests it or when a change touches cross-module boundaries without clear ownership.

## Git workflow

Never create commits unless the user explicitly asks for them.
When the user requests per-task commits, commit each discrete task before starting the next one.
Before every commit, run the exact full commands `git status`, `git diff`, and `git log -10`.
Do not replace these required inspections with abbreviated variants such as `git status --short`, `git diff --stat`, or `git log --oneline`.
Read the full commit messages from `git log -10`, including their bodies and trailers.
Stage only files that belong to the current task.

Format commit messages per repository conventions:
- Use the subject format `<scope>: <Capitalized summary>`. Derive the lowercase scope from the component or directory you changed.
- Write the summary in the imperative mood and do not end it with a period.
- Keep the subject near 50 characters and never longer than 72 characters.
- Add a concise, technical body when the subject does not provide enough context. Explain what changed and why.
- Separate the subject from the body with a blank line and wrap body text at 72 characters.

Check commit signing once per session with `git config commit.gpgsign` and `git config user.signingkey`.
Remember the result for the rest of the session.
If both are set, sign every commit with the configured method and use `git commit --signoff`.
Do not amend commits, push, or rewrite history unless the user explicitly asks.
