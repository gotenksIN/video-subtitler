"""Whole-song lyrics generation lifecycle."""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from modules import core, gemini, io, media

WORK_ROOT = ".lyrics-generator-work"
LANGUAGE_TAG = re.compile(
    r"^(?:"
    r"(?:[A-Za-z]{2,3}(?:-[A-Za-z]{3}){0,3}|[A-Za-z]{4}|[A-Za-z]{5,8})"
    r"(?:-[A-Za-z]{4})?"
    r"(?:-(?:[A-Za-z]{2}|[0-9]{3}))?"
    r"(?:-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*"
    r"(?:-[0-9A-WY-Za-wy-z](?:-[A-Za-z0-9]{2,8})+)*"
    r"(?:-x(?:-[A-Za-z0-9]{1,8})+)?"
    r"|x(?:-[A-Za-z0-9]{1,8})+"
    r"|en-GB-oed"
    r"|i-(?:ami|bnn|default|enochian|hak|klingon|lux|mingo|navajo|pwn|tao|tay|tsu)"
    r"|sgn-(?:BE-FR|BE-NL|CH-DE)"
    r")$",
    re.ASCII | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Inputs that control one lyrics generation run."""

    audio_path: Path
    output_dir: Path | None = None
    api_key: str | None = None
    base_url: str | None = None
    title: str | None = None
    artist: str | None = None
    target_language: str | None = None
    flash_model: str = gemini.DEFAULT_FLASH_MODEL
    pro_model: str = gemini.DEFAULT_PRO_MODEL

    @property
    def resolved_title(self) -> str:
        return (self.title or core.derive_song_title(self.audio_path)).strip()

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.audio_path.parent


def output_paths(config: GenerationConfig) -> tuple[Path, Path | None]:
    """Return same-stem LRC and optional TTML destinations."""
    stem = config.audio_path.stem
    directory = config.resolved_output_dir
    return directory / f"{stem}.lrc", (
        directory / f"{stem}.ttml" if config.target_language is not None else None
    )


def validate_generation_config(config: GenerationConfig) -> tuple[Path, Path | None]:
    """Fail before media conversion or API calls when inputs are unsafe."""
    if not config.audio_path.is_file():
        raise RuntimeError(f"Audio file not found: {config.audio_path}")
    if not config.resolved_title:
        raise ValueError("Song title must not be blank")
    if config.target_language is not None and not LANGUAGE_TAG.fullmatch(
        config.target_language
    ):
        raise ValueError("--target-language must be a BCP 47 language tag")
    lrc_path, ttml_path = output_paths(config)
    destinations = [lrc_path, *([ttml_path] if ttml_path else [])]
    source = config.audio_path.resolve()
    for destination in destinations:
        if destination.resolve() == source:
            raise RuntimeError("An output path must not resolve to the source audio")
        if os.path.lexists(destination):
            raise FileExistsError(f"Output already exists: {destination}")
    if not config.api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY or pass --api-key."
        )
    return lrc_path, ttml_path


def run_generation(config: GenerationConfig) -> None:
    """Generate lyrics from scratch and retain working files only on failure."""
    lrc_path, ttml_path = validate_generation_config(config)
    config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=WORK_ROOT))
    completed = False
    started = monotonic()
    try:
        print(f"Normalizing audio: {config.audio_path}", flush=True)
        audio_path, duration = media.extract_complete_audio(config.audio_path, work_dir)
        print(
            f"Audio ready in {monotonic() - started:.1f}s | "
            f"{duration:.2f}s duration | {audio_path.stat().st_size / 1024 / 1024:.2f} MiB",
            flush=True,
        )
        context = gemini.research_song_context(
            config.api_key,
            config.base_url,
            config.pro_model,
            config.resolved_title,
            config.artist,
        )
        draft = gemini.transcribe_song(
            config.api_key,
            config.base_url,
            config.flash_model,
            audio_path,
            duration,
            context,
            work_dir,
        )
        print(f"Transcription validated: {len(draft)} lines", flush=True)
        refined = gemini.refine_song(
            config.api_key,
            config.base_url,
            config.pro_model,
            audio_path,
            duration,
            context,
            draft,
            work_dir,
        )
        print(f"Refinement validated: {len(refined)} lines", flush=True)

        ttml = None
        if config.target_language is not None:
            translations = gemini.translate_song(
                config.api_key,
                config.base_url,
                config.pro_model,
                config.target_language,
                context,
                refined,
                work_dir,
            )
            ttml = core.serialize_ttml(
                refined,
                translations,
                duration,
                config.target_language,
                config.resolved_title,
            )
        lrc = core.serialize_lrc(refined, duration)

        for destination in (lrc_path, *([ttml_path] if ttml_path else [])):
            if os.path.lexists(destination):
                raise FileExistsError(f"Output already exists: {destination}")
        io.atomic_publish_text(lrc_path, lrc)
        if ttml_path is not None:
            io.atomic_publish_text(ttml_path, ttml)
        completed = True
        print(f"Saved source lyrics to {lrc_path}")
        if ttml_path is not None:
            print(f"Saved bilingual lyrics to {ttml_path}")
        print(f"Generation completed in {monotonic() - started:.1f}s", flush=True)
    finally:
        if completed:
            shutil.rmtree(work_dir)
        else:
            print(f"Retained failed-run files in {work_dir}", flush=True)
