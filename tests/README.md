# Test suite

The suite verifies observable behavior and critical failure modes across module boundaries.
Core tests verify lyric validation and LRC and TTML output serialization.
I/O tests verify no-replace final publication.
Gemini tests verify adapter behavior at external boundaries using stateful fakes.
Pipeline tests verify output safety, generation from scratch, and successful work-directory cleanup.
CLI tests verify error reporting and flag validation through the executable boundary.
Media normalization is verified manually using real FFmpeg and FFprobe audio inputs.

Run the complete suite with:

```bash
uv run pytest
```
