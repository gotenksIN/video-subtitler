import os
import sys
import json
import argparse
import subprocess
import shutil
import webvtt
import concurrent.futures
import hashlib
import re
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define Structured Output Schema
class Caption(BaseModel):
    id: int
    start: str = Field(description="Start time in HH:MM:SS.mmm format")
    end: str = Field(description="End time in HH:MM:SS.mmm format")
    text: str = Field(description="The subtitle text")

class AlignmentResponse(BaseModel):
    captions: list[Caption]

class RefinedCaption(BaseModel):
    id: int = Field(description="The integer ID of the subtitle to change")
    text: str = Field(description="The corrected text")

class RefinementResponse(BaseModel):
    changes: list[RefinedCaption] = Field(
        description="List of subtitles to change. Only include ones that need changes."
    )

CHUNK_ROOT = "temp_video_chunks"
SPLIT_COMPLETE_MARKER = ".split_complete"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".lock"
INLINE_VIDEO_WARNING_BYTES = 20 * 1024 * 1024
OVERLAP_FORMATS = {
    "webm": (".webm", "video/webm"),
    "mp4": (".mp4", "video/mp4"),
}

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def probe_video_format(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=format_name",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        fmt = result.stdout.strip().lower()
        if "webm" in fmt or "matroska" in fmt:
            return ".webm", "video/webm"
        elif "mp4" in fmt or "mov" in fmt:
            return ".mp4", "video/mp4"
        else:
            return ".mp4", "video/mp4"
    except Exception:
        return ".mp4", "video/mp4"

def parse_time(time_str):
    value = str(time_str).strip().replace(",", ".")
    parts = value.split(':')
    if len(parts) == 3:
        h, m, s_ms = parts
    elif len(parts) == 2:
        h = "0"
        m, s_ms = parts
    elif len(parts) == 1:
        h, m = "0", "0"
        s_ms = parts[0]
    else:
        raise ValueError(f"Invalid timestamp: {time_str}")

    if "." in s_ms:
        s, frac = s_ms.split('.', 1)
        frac_seconds = int(frac) / (10 ** len(frac)) if frac else 0
    else:
        s = s_ms
        frac_seconds = 0

    return int(h) * 3600 + int(m) * 60 + int(s) + frac_seconds

def format_time(seconds):
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {seconds}")

    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def build_manifest(args):
    ext, mime = probe_video_format(args.video_file)
    if args.overlap:
        process_ext, process_mime = OVERLAP_FORMATS[args.overlap_format]
    else:
        process_ext, process_mime = ext, mime

    manifest = {
        "video": file_fingerprint(args.video_file),
        "vtt": file_fingerprint(args.vtt_file) if args.vtt_file else None,
        "chunk_dur": args.chunk_dur,
        "format": "stream-copy-v1",
        "mode": "align" if args.vtt_file else "generate",
        "model": args.model,
        "thinking_budget": args.thinking_budget,
        "overlap": args.overlap,
        "overlap_format": args.overlap_format if args.overlap else None,
        "clip_workers": args.clip_workers if args.overlap else 0,
        "chunk_ext": ext,
        "chunk_mime": mime,
        "process_ext": process_ext,
        "process_mime": process_mime,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return manifest, os.path.join(CHUNK_ROOT, digest)

def acquire_lock(chunk_dir):
    lock_path = os.path.join(chunk_dir, LOCK_NAME)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Another run is already using {chunk_dir}")

    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)
    return lock_path

def release_lock(lock_path):
    if lock_path and os.path.exists(lock_path):
        os.remove(lock_path)

def clean_incomplete_split(chunk_dir):
    for name in os.listdir(chunk_dir):
        if (
            re.fullmatch(r"chunk_\d+\.(mp4|webm)", name)
            or re.fullmatch(r"context_chunk_\d+\.(mp4|webm)(\.tmp)?", name)
            or re.fullmatch(r"aligned_chunk_\d+\.json(\.tmp)?", name)
            or name == "segments.csv"
        ):
            os.remove(os.path.join(chunk_dir, name))

def split_video(video_file, chunk_dir, chunk_dur_sec, manifest):
    print(f"Splitting video into {chunk_dur_sec}-second chunks (stream copy mode)...")
    os.makedirs(chunk_dir, exist_ok=True)
    atomic_write_json(os.path.join(chunk_dir, MANIFEST_NAME), manifest)

    complete_marker = os.path.join(chunk_dir, SPLIT_COMPLETE_MARKER)
    if os.path.exists(complete_marker) and list_chunks(chunk_dir):
        print("Chunks already exist, skipping splitting.")
        return

    clean_incomplete_split(chunk_dir)
    ext = manifest["chunk_ext"]

    cmd = [
        "ffmpeg", "-y", "-i", video_file,
        "-map", "0:v:0", "-map", "0:a?", "-sn",
        "-c", "copy",
        "-f", "segment", "-segment_time", str(chunk_dur_sec),
        "-segment_list", os.path.join(chunk_dir, "segments.csv"),
        "-reset_timestamps", "1", os.path.join(chunk_dir, f"chunk_%03d{ext}")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Path(complete_marker).write_text("ok\n", encoding="utf-8")
    print("Splitting complete.")

def list_chunks(chunk_dir):
    csv_path = os.path.join(chunk_dir, "segments.csv")
    if not os.path.exists(csv_path):
        return []

    chunks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) >= 3:
                name = parts[0]
                start = float(parts[1])
                end = float(parts[2])
                chunks.append({
                    "idx": i,
                    "name": name,
                    "start": start,
                    "end": end,
                    "duration": end - start
                })
    return chunks

def get_processing_windows(chunks, overlap_sec):
    if not chunks:
        return []

    video_end = chunks[-1]["end"]
    windows = []
    for chunk in chunks:
        owner_start = chunk["start"]
        owner_end = chunk["end"]
        clip_start = max(0.0, owner_start - overlap_sec)
        clip_end = min(video_end, owner_end + overlap_sec)
        windows.append({
            **chunk,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "clip_duration": clip_end - clip_start,
            "owner_start": owner_start,
            "owner_end": owner_end,
            "owner_start_rel": owner_start - clip_start,
            "owner_end_rel": owner_end - clip_start,
        })
    return windows

def suggested_clip_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 8 or 1))

def overlap_codec_args(ext):
    if ext == ".webm":
        return [
            "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-threads", "8", "-tile-columns", "2", "-row-mt", "1", "-frame-parallel", "1",
            "-c:a", "libopus", "-b:a", "128k",
        ]

    return [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
    ]

def create_overlap_clip(video_file, chunk_dir, chunk_idx, clip_start, clip_end, clip_ext):
    clip_name = f"context_chunk_{chunk_idx:03d}{clip_ext}"
    clip_path = os.path.join(chunk_dir, clip_name)
    if os.path.exists(clip_path):
        return clip_name
    tmp_path = f"{clip_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    duration = clip_end - clip_start
    if duration <= 0:
        raise ValueError(f"Invalid overlap clip duration for chunk {chunk_idx}: {duration}")

    print(f"Creating overlap clip {clip_name} ({format_time(clip_start)} to {format_time(clip_end)})...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-ss", format_time(clip_start), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a?", "-sn",
        *overlap_codec_args(clip_ext),
        "-f", "webm" if clip_ext == ".webm" else "mp4",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp_path, clip_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return clip_name

def attach_overlap_clip(video_file, chunk_dir, chunk, overlap_sec, clip_ext):
    if overlap_sec > 0:
        clip_name = create_overlap_clip(video_file, chunk_dir, chunk["idx"], chunk["clip_start"], chunk["clip_end"], clip_ext)
    else:
        clip_name = chunk["name"]

    return {
        **chunk,
        "clip_name": clip_name,
    }

def collect_api_results(futures):
    failed = []
    for future in concurrent.futures.as_completed(futures):
        chunk_name = futures[future]
        try:
            if not future.result():
                failed.append(chunk_name)
        except Exception as e:
            print(f"ERROR processing {chunk_name}: {e}")
            failed.append(chunk_name)
    return failed

def process_chunks(
    api_key, base_url, video_file, chunk_dir, chunks, overlap_sec, clip_ext,
    clip_workers, api_workers, vtt_file, model_name, chunk_mime, thinking_budget,
):
    windows = get_processing_windows(chunks, overlap_sec)
    if overlap_sec <= 0 or len(windows) <= 1:
        processing_chunks = [attach_overlap_clip(video_file, chunk_dir, chunk, overlap_sec, clip_ext) for chunk in windows]
        print(f"Processing {len(processing_chunks)} chunks using {api_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=api_workers) as executor:
            futures = {
                executor.submit(
                    process_chunk,
                    api_key, base_url,
                    chunk, chunk_dir,
                    vtt_file, model_name, chunk_mime, thinking_budget,
                ): chunk["clip_name"]
                for chunk in processing_chunks
            }
            return collect_api_results(futures)

    print(
        f"Creating {len(windows)} overlap clips using {clip_workers} workers "
        f"and processing them using {api_workers} API workers..."
    )
    failed = []
    api_futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=clip_workers) as clip_executor, \
            concurrent.futures.ThreadPoolExecutor(max_workers=api_workers) as api_executor:
        clip_futures = {
            clip_executor.submit(attach_overlap_clip, video_file, chunk_dir, chunk, overlap_sec, clip_ext): chunk
            for chunk in windows
        }
        for future in concurrent.futures.as_completed(clip_futures):
            chunk = clip_futures[future]
            chunk_name = f"context_chunk_{chunk['idx']:03d}{clip_ext}"
            try:
                processing_chunk = future.result()
            except Exception as e:
                print(f"ERROR creating {chunk_name}: {e}")
                failed.append(chunk_name)
                continue

            api_futures[
                api_executor.submit(
                    process_chunk,
                    api_key, base_url,
                    processing_chunk, chunk_dir,
                    vtt_file, model_name, chunk_mime, thinking_budget,
                )
            ] = processing_chunk["clip_name"]

        failed.extend(collect_api_results(api_futures))

    return failed

def get_captions_for_chunk(vtt_path, owner_start_sec, owner_end_sec, clip_start_sec, clip_duration):
    vtt = webvtt.read(vtt_path)
    chunk_cues = []

    for i, caption in enumerate(vtt):
        cap_start = parse_time(caption.start)
        cap_end = parse_time(caption.end)
        if cap_end <= cap_start:
            continue

        # Assign boundary captions by midpoint instead of raw start time so lines
        # that straddle a chunk edge stay with the chunk that contains most of them.
        midpoint = (cap_start + cap_end) / 2
        if not (owner_start_sec <= midpoint < owner_end_sec):
            continue

        rel_start = clamp(cap_start - clip_start_sec, 0.0, clip_duration)
        rel_end = clamp(cap_end - clip_start_sec, 0.0, clip_duration)
        if rel_end <= rel_start:
            rel_end = clamp(rel_start + 0.1, 0.1, clip_duration)
            if rel_end <= rel_start:
                rel_start = clamp(clip_duration - 0.1, 0.0, clip_duration)
                rel_end = clip_duration

        chunk_cues.append({
            "id": i,
            "start": format_time(rel_start),
            "end": format_time(rel_end),
            "text": caption.text
        })
    return chunk_cues


def validate_captions(captions, chunk_duration, original_cues=None):
    validated = []

    if original_cues is not None:
        originals_by_id = {cue["id"]: cue for cue in original_cues}

        unique_captions = {}
        for cap in captions:
            if cap.id in originals_by_id and cap.id not in unique_captions:
                unique_captions[cap.id] = cap

        for expected_id in sorted(originals_by_id.keys()):
            orig_cue = originals_by_id[expected_id]

            if expected_id not in unique_captions:
                print(f"      [Auto-fix] Missing caption id={expected_id}. Restoring original.")
                validated.append({
                    "id": expected_id,
                    "start": orig_cue["start"],
                    "end": orig_cue["end"],
                    "text": orig_cue["text"],
                })
                continue

            cap = unique_captions[expected_id]

            try:
                start = parse_time(cap.start)
            except Exception:
                start = parse_time(orig_cue["start"])

            try:
                end = parse_time(cap.end)
            except Exception:
                end = parse_time(orig_cue["end"])

            if start < 0:
                start = 0
            if end <= start:
                orig_dur = parse_time(orig_cue["end"]) - parse_time(orig_cue["start"])
                end = start + max(orig_dur, 0.1)

            max_end = chunk_duration + 0.5
            if end > max_end:
                end = max_end
            if start > max_end:
                start = max_end - 0.5

            validated.append({
                "id": expected_id,
                "start": format_time(start),
                "end": format_time(end),
                "text": orig_cue["text"],
            })

    else:
        seen_ids = set()
        duplicate_ids = set()
        for cap in captions:
            if cap.id in seen_ids:
                duplicate_ids.add(cap.id)
            seen_ids.add(cap.id)
        if duplicate_ids:
            raise ValueError(f"Duplicate caption IDs: {sorted(duplicate_ids)}")

        for cap in captions:
            start = parse_time(cap.start)
            end = parse_time(cap.end)
            if start < 0 or end <= start:
                raise ValueError(f"Invalid caption timing for id={cap.id}: {cap.start} --> {cap.end}")

            max_end = chunk_duration + 0.5
            if end > max_end:
                end = max_end

            validated.append({
                "id": cap.id,
                "start": format_time(start),
                "end": format_time(end),
                "text": cap.text,
            })

        validated = sorted(validated, key=lambda item: (parse_time(item["start"]), item["id"]))
        
        # Auto-heal overlaps instead of crashing
        for i in range(1, len(validated)):
            prev_cap = validated[i-1]
            curr_cap = validated[i]
            
            prev_end = parse_time(prev_cap["end"])
            curr_start = parse_time(curr_cap["start"])
            
            if curr_start < prev_end:
                # If they overlap, adjust the previous caption's end time to match the current's start time
                # ensuring it doesn't go below its own start time
                prev_start = parse_time(prev_cap["start"])
                new_prev_end = max(prev_start + 0.001, curr_start)
                
                # If adjusting the previous end makes the current start invalid (curr_start < prev_start), 
                # push the current start forward instead.
                if new_prev_end > curr_start:
                    curr_start = new_prev_end
                    curr_end = parse_time(curr_cap["end"])
                    curr_end = max(curr_start + 0.001, curr_end)
                    curr_cap["start"] = format_time(curr_start)
                    curr_cap["end"] = format_time(curr_end)
                
                prev_cap["end"] = format_time(new_prev_end)

    return sorted(validated, key=lambda item: (parse_time(item["start"]), item["id"]))

def load_cached_captions(out_json, chunk_duration, original_cues):
    if not os.path.exists(out_json):
        return None
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        response = AlignmentResponse(captions=data)
        return validate_captions(response.captions, chunk_duration, original_cues, )
    except Exception as e:
        print(f"Ignoring invalid cached output {out_json}: {e}")
        os.remove(out_json)
        return None

def create_client(api_key, base_url):
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["http_options"] = {"base_url": base_url}
    return genai.Client(**kwargs)

def generate_content_config(thinking_budget):
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": AlignmentResponse,
    }
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kwargs)

def process_chunk(api_key, base_url, chunk, chunk_dir, vtt_file, model_name, chunk_mime, thinking_budget):
    chunk_idx = chunk["idx"]
    clip_name = chunk["clip_name"]
    clip_start = chunk["clip_start"]
    clip_duration = chunk["clip_duration"]
    owner_start_rel = chunk["owner_start_rel"]
    owner_end_rel = chunk["owner_end_rel"]
    out_json = os.path.join(chunk_dir, f"aligned_chunk_{chunk_idx:03d}.json")
    chunk_path = os.path.join(chunk_dir, clip_name)

    if vtt_file:
        # Alignment Mode
        original_cues = get_captions_for_chunk(
            vtt_file,
            chunk["owner_start"],
            chunk["owner_end"],
            clip_start,
            clip_duration,
        )
        if not original_cues:
            print(f"No captions for {clip_name}, skipping.")
            atomic_write_json(out_json, [])
            return True

        cues_json_str = json.dumps(original_cues, ensure_ascii=False, indent=2)
        cached = load_cached_captions(out_json, clip_duration, original_cues, )
        if cached is not None:
            print(f"Skipping {clip_name} - already processed.")
            return True

        prompt = f"""
        You are an expert subtitle aligner.
        Watch this {clip_duration:.3f}-second video clip.
        Below is a JSON list of the original subtitles assigned to this chunk.
        The main chunk window is {format_time(owner_start_rel)} to {format_time(owner_end_rel)} in this clip. Video before or after that window is context only.

        Your task:
        1. Fix the timestamps of these captions so they align perfectly with the video.
        2. For spoken dialogue, start_time must be the exact millisecond of the first audible syllable/word, and end_time must be the exact end of the last audible syllable.
        3. Silent gaps between spoken sentences must remain real gaps in the timestamps; do not arbitrarily stretch durations to fill silence.
        4. Crucially, if a caption corresponds to visual text (editors' flair text), match the timing with EXACTLY when the text appears and disappears visually on screen.
        5. Preserve every original 'id' exactly once.
        6. Do NOT change, translate, correct, split, merge, or reorder the original text. Return exactly what was provided in the 'text' field.
        7. Return a complete list of all captions, ensuring none are dropped. ALL provided IDs MUST be returned, even if your corrected timing places them in the context window.
        8. Keep captions sorted by start time.
        9. Use timestamps relative to this full clip, from 00:00:00.000 to {format_time(clip_duration)}.
        10. Some captions were assigned by midpoint because they may straddle chunk boundaries. Use the context video to place them correctly.

        Return ONLY the valid JSON object matching the required schema with a 'captions' array. Do not include markdown formatting or explanations.

        Original Captions:
        {cues_json_str}
        """
    else:
        # Generation Mode
        cached = load_cached_captions(out_json, clip_duration, None)
        if cached is not None:
            print(f"Skipping {clip_name} - already processed.")
            return True

        prompt = f"""
        You are an expert subtitle generator and translator.
        Watch this {clip_duration:.3f}-second video clip.
        The main chunk window is {format_time(owner_start_rel)} to {format_time(owner_end_rel)} in this clip. Video before or after that window is context only.

        Your task:
        1. Generate accurate English subtitles for dialogue and relevant on-screen text for the ENTIRE clip, including the context windows. (We will filter them later).
        2. Create accurate timestamps relative to the start of this full clip, ranging from 00:00:00.000 to {format_time(clip_duration)}.
        3. For spoken dialogue, start_time must be the exact millisecond of the first audible syllable/word, and end_time must be the exact end of the last audible syllable.
        4. Silent gaps between spoken sentences must remain real gaps in the timestamps; do not arbitrarily stretch durations to fill silence.
        5. Prefer faithful, clear English over punchy paraphrases when dialogue is not English.
        6. Preserve names and recurring terms consistently within the chunk. Keep original native nicknames and do not translate them.
        7. If a proper noun is uncertain, transliterate conservatively instead of inventing a nickname or joke.
        8. Preserve native cultural terms and foods rather than over-localizing them.
        9. Do not summarize, explain, or infer missing dialogue.
        10. Include meaningful on-screen text when it matters for understanding the video, timing it exactly to when it appears and disappears.
        11. Ignore decorative text, logos, watermarks, and unrelated UI.
        12. Use sequential integer IDs starting at 0.
        13. Keep captions sorted by start time and do not overlap them.
        14. Follow standard subtitle rules: max 42 characters per line, max 2 lines per caption.
        15. Split long speech into readable, natural phrases.

        Return ONLY the valid JSON object matching the required schema with a 'captions' array. Do not include markdown formatting or explanations.
        """

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()
        if len(video_data) > INLINE_VIDEO_WARNING_BYTES:
            print(
                f"[Worker-{chunk_idx:03d}] Warning: {clip_name} is {len(video_data) / 1024 / 1024:.1f} MB. "
                "Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail."
            )

        mode_str = "aligning" if vtt_file else "generating"
        print(f"[Worker-{chunk_idx:03d}] {mode_str.capitalize()} {clip_name} using Gemini API...")

        with create_client(api_key, base_url) as client:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=video_data, mime_type=chunk_mime),
                    prompt
                ],
                config=generate_content_config(thinking_budget)
            )

        parsed_response = AlignmentResponse.model_validate_json(response.text)
        validated = validate_captions(
            parsed_response.captions,
            clip_duration,
            original_cues if vtt_file else None
        )
        atomic_write_json(out_json, validated)

        print(f"[Worker-{chunk_idx:03d}] Finished {clip_name}.")
        return True
    except Exception as e:
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {clip_name}: {e}")
        return False

def load_manifest(chunk_dir):
    manifest_path = os.path.join(chunk_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def stitch(chunk_dir, output_vtt):
    print("Stitching chunks into final VTT...")
    final_vtt = webvtt.WebVTT()
    captions_to_write = []

    manifest = load_manifest(chunk_dir)
    chunks = list_chunks(chunk_dir)
    windows = get_processing_windows(chunks, float(manifest.get("overlap") or 0.0))
    window_map = {c["idx"]: c for c in windows}
    filter_generated_context = manifest.get("mode") == "generate" and float(manifest.get("overlap") or 0.0) > 0

    json_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('aligned_chunk_') and f.endswith('.json')])

    for json_name in json_files:
        chunk_idx = int(json_name.replace("aligned_chunk_", "").replace(".json", ""))
        window = window_map.get(chunk_idx)
        if not window:
            continue
        offset_sec = window["clip_start"]

        with open(os.path.join(chunk_dir, json_name), "r") as f:
            captions = json.load(f)

        for cap in captions:
            rel_start = parse_time(cap["start"])
            rel_end = parse_time(cap["end"])
            if filter_generated_context:
                midpoint = (rel_start + rel_end) / 2
                if not (window["owner_start_rel"] <= midpoint < window["owner_end_rel"]):
                    continue

            abs_start = rel_start + offset_sec
            abs_end = rel_end + offset_sec
            if abs_end <= abs_start:
                raise ValueError(f"Invalid caption timing in {json_name}: {cap}")

            captions_to_write.append({
                "start": abs_start,
                "end": abs_end,
                "text": cap["text"],
            })

    for cap in sorted(captions_to_write, key=lambda item: item["start"]):
        final_vtt.captions.append(webvtt.Caption(
            format_time(cap["start"]),
            format_time(cap["end"]),
            cap["text"]
        ))

    output_path = Path(output_vtt)
    tmp_output = output_path.with_name(f"{output_path.name}.tmp.vtt")
    final_vtt.save(str(tmp_output))
    os.replace(tmp_output, output_path)
    print(f"Successfully saved to {output_vtt} with {len(final_vtt.captions)} total captions.")

def global_refine_subtitles(input_vtt, output_vtt, api_key, base_url, model_name, thinking_budget):
    print(f"Loading {input_vtt} for global refinement...")
    vtt = webvtt.read(input_vtt)

    script_lines = []
    for i, caption in enumerate(vtt):
        text = caption.text.replace('\n', ' ')
        script_lines.append(f"[{i}] {caption.start} --> {caption.end}: {text}")

    full_script = "\n".join(script_lines)

    prompt = f"""
You are an expert subtitle localization editor.
Below is an entire subtitle script for a video.

Your task is to read the whole script to understand the global context and fix:
1. Inconsistent character names or nicknames (ensure they are uniform from start to finish).
2. Over-localized terms (revert to native cultural terms where appropriate).
3. Awkward grammar or unnatural phrasing.
4. Glaring continuity errors in dialogue.

DO NOT REWRITE THE ENTIRE SCRIPT. Only modify lines that genuinely need correction for consistency or natural flow. If a line is acceptable, leave it alone.

Return a JSON object containing a 'changes' list with ONLY the lines you want to change. Each change must have the 'id' (found in brackets like [ID]) and the 'text' (the new, corrected text).
Do not change the timestamps. Do not merge or split lines.

Script:
{full_script}
"""

    with create_client(api_key, base_url) as client:
        print("Sending script to Gemini for global refinement (this may take a minute)...")
        config_kwargs = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": RefinementResponse,
        }
        if thinking_budget is not None and thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs)
        )

    try:
        refinements = RefinementResponse.model_validate_json(response.text)
    except Exception as e:
        print(f"Error parsing model response: {e}")
        print("Raw response:")
        print(response.text)
        sys.exit(1)

    changes = refinements.changes
    print(f"Model proposed changes to {len(changes)} lines out of {len(vtt)}.")

    for change in changes:
        try:
            if 0 <= change.id < len(vtt):
                vtt[change.id].text = change.text
        except ValueError:
            pass

    output_path = Path(output_vtt)
    tmp_output = output_path.with_name(f"{output_path.name}.tmp.vtt")
    vtt.save(str(tmp_output))
    os.replace(tmp_output, output_path)
    print(f"Saved refined subtitles to {output_vtt}")

def main():
    parser = argparse.ArgumentParser(description="Align or Generate VTT subtitles for a video using Gemini API.")
    parser.add_argument("video_file_or_vtt", help="Path to the original video file (OR path to input VTT if --refine-only is used)")
    parser.add_argument("vtt_file", nargs="?", default=None, help="Path to the original VTT subtitle file (optional). If omitted, generates from scratch.")
    parser.add_argument("--output", "-o", default="output_subtitles.vtt", help="Output path for the generated/aligned VTT file")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--base-url", default=os.environ.get("GEMINI_API_BASE"), help="Base URL for Gemini API (optional)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"), help="Gemini model to use")
    parser.add_argument("--disable-text-refine", action="store_true", help="Disable the global text refinement pass after alignment/generation")
    parser.add_argument("--refine-only", action="store_true", help="Skip video processing entirely; only run global text refinement on the input VTT file")
    parser.add_argument("--chunk-dur", type=int, default=60, help="Chunk duration in seconds (default: 60)")
    parser.add_argument("--overlap", type=float, default=5.0, help="Seconds of context to add before and after each chunk (default: 5)")
    parser.add_argument("--overlap-format", choices=sorted(OVERLAP_FORMATS.keys()), default="mp4", help="Container to use for re-encoded overlap clips (default: mp4)")
    parser.add_argument("--clip-workers", type=int, default=0, help="Parallel overlap clip encode workers. 0 means auto.")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent API workers")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Gemini thinking token budget (default: 0).")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep the per-input work directory after successful processing")

    args = parser.parse_args()

    if args.refine_only:
        if not os.path.exists(args.video_file_or_vtt):
            print(f"Error: Input VTT file not found: {args.video_file_or_vtt}")
            sys.exit(1)
        if not args.api_key:
            print("Error: Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key.")
            sys.exit(1)
        global_refine_subtitles(
            args.video_file_or_vtt, args.output, args.api_key, args.base_url, args.model, args.thinking_budget
        )
        sys.exit(0)

    # Map back to video_file for standard pipeline processing
    args.video_file = args.video_file_or_vtt

    if args.chunk_dur <= 0:
        print("Error: --chunk-dur must be greater than 0")
        sys.exit(1)

    if args.workers <= 0:
        print("Error: --workers must be greater than 0")
        sys.exit(1)

    if args.clip_workers < 0:
        print("Error: --clip-workers must be greater than or equal to 0")
        sys.exit(1)

    if args.thinking_budget is not None and args.thinking_budget < 0:
        print("Error: --thinking-budget must be greater than or equal to 0")
        sys.exit(1)

    if args.overlap < 0:
        print("Error: --overlap must be greater than or equal to 0")
        sys.exit(1)

    if args.overlap >= args.chunk_dur:
        print("Error: --overlap must be smaller than --chunk-dur")
        sys.exit(1)

    clip_workers = args.clip_workers or suggested_clip_workers()

    if not args.vtt_file:
        print("Warning: generation mode without VTT.")

    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if args.vtt_file and not os.path.exists(args.vtt_file):
        print(f"Error: VTT file not found: {args.vtt_file}")
        sys.exit(1)

    if not args.api_key:
        print("Error: Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key.")
        sys.exit(1)

    manifest, chunk_dir = build_manifest(args)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_path = None
    completed = False

    try:
        lock_path = acquire_lock(chunk_dir)
        print(f"Using work directory: {chunk_dir}")

        # 1. Split Video
        split_video(args.video_file, chunk_dir, args.chunk_dur, manifest)

        chunks = list_chunks(chunk_dir)
        if not chunks:
            raise RuntimeError("No video chunks were created")

        # 2. Process chunks concurrently. Overlap runs pipeline clip creation into API calls.
        failed = process_chunks(
            args.api_key,
            args.base_url,
            args.video_file,
            chunk_dir,
            chunks,
            args.overlap,
            manifest["process_ext"],
            clip_workers,
            args.workers,
            args.vtt_file,
            args.model,
            manifest["process_mime"],
            args.thinking_budget,
        )
        if failed:
            raise RuntimeError(
                f"Failed to process {len(failed)} chunk(s): {', '.join(sorted(failed))}. "
                f"Keeping {chunk_dir} so you can retry."
            )

        # 3. Stitch chunks together
        stitch(chunk_dir, args.output)

        # 4. Optional Global Refinement Pass
        if not args.disable_text_refine:
            global_refine_subtitles(
                args.output, args.output, args.api_key, args.base_url, args.model, args.thinking_budget
            )

        completed = True

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        release_lock(lock_path)
        # 4. Cleanup
        if completed and not args.keep_chunks and os.path.exists(chunk_dir):
            print(f"Cleaning up temporary directory: {chunk_dir}")
            shutil.rmtree(chunk_dir)

if __name__ == "__main__":
    main()
