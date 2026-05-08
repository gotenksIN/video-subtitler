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

CHUNK_ROOT = "temp_video_chunks"
SPLIT_COMPLETE_MARKER = ".split_complete"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".lock"
INLINE_VIDEO_WARNING_BYTES = 20 * 1024 * 1024

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
    manifest = {
        "video": file_fingerprint(args.video_file),
        "vtt": file_fingerprint(args.vtt_file) if args.vtt_file else None,
        "chunk_dur": args.chunk_dur,
        "format": "stream-copy-v1",
        "mode": "align" if args.vtt_file else "generate",
        "text_mode": args.text_mode if args.vtt_file else None,
        "chunk_ext": ext,
        "chunk_mime": mime,
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
        if re.fullmatch(r"chunk_\d+\.(mp4|webm)", name) or re.fullmatch(r"aligned_chunk_\d+\.json(\.tmp)?", name) or name == "segments.csv":
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

def get_captions_for_chunk(vtt_path, start_sec, end_sec):
    vtt = webvtt.read(vtt_path)
    chunk_cues = []
    chunk_duration = max(end_sec - start_sec, 0.0)

    for i, caption in enumerate(vtt):
        cap_start = parse_time(caption.start)
        cap_end = parse_time(caption.end)
        if cap_end <= cap_start:
            continue

        # Assign boundary captions by midpoint instead of raw start time so lines
        # that straddle a chunk edge stay with the chunk that contains most of them.
        midpoint = (cap_start + cap_end) / 2
        if not (start_sec <= midpoint < end_sec):
            continue

        rel_start = clamp(cap_start - start_sec, 0.0, chunk_duration)
        rel_end = clamp(cap_end - start_sec, 0.0, chunk_duration)
        if rel_end <= rel_start:
            rel_end = clamp(rel_start + 0.1, 0.1, chunk_duration)
            if rel_end <= rel_start:
                rel_start = clamp(chunk_duration - 0.1, 0.0, chunk_duration)
                rel_end = chunk_duration

        chunk_cues.append({
            "id": i,
            "start": format_time(rel_start),
            "end": format_time(rel_end),
            "text": caption.text
        })
    return chunk_cues

def select_alignment_text(cap, orig_cue, text_mode):
    if text_mode == "preserve":
        return orig_cue["text"]

    text = cap.text.strip()
    return text or orig_cue["text"]

def validate_captions(captions, chunk_duration, original_cues=None, text_mode="preserve"):
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

            max_end = max(chunk_duration + 5.0, parse_time(orig_cue["end"]) + 5.0)
            if end > max_end:
                end = max_end
            if start > max_end:
                start = max_end - 0.5

            validated.append({
                "id": expected_id,
                "start": format_time(start),
                "end": format_time(end),
                "text": select_alignment_text(cap, orig_cue, text_mode),
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
        previous_end = None
        for cap in validated:
            start = parse_time(cap["start"])
            end = parse_time(cap["end"])
            if previous_end is not None and start < previous_end - 0.05:
                raise ValueError(f"Generated captions overlap near id={cap['id']}")
            previous_end = max(previous_end or 0, end)

    return sorted(validated, key=lambda item: (parse_time(item["start"]), item["id"]))

def load_cached_captions(out_json, chunk_duration, original_cues, text_mode="preserve"):
    if not os.path.exists(out_json):
        return None
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        response = AlignmentResponse(captions=data)
        return validate_captions(response.captions, chunk_duration, original_cues, text_mode=text_mode)
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

def process_chunk(api_key, base_url, chunk_idx, chunk_name, chunk_dir, vtt_file, start_sec, end_sec, chunk_duration, model_name, chunk_mime, text_mode):
    out_json = os.path.join(chunk_dir, f"aligned_chunk_{chunk_idx:03d}.json")
    chunk_path = os.path.join(chunk_dir, chunk_name)

    if vtt_file:
        # Alignment Mode
        original_cues = get_captions_for_chunk(vtt_file, start_sec, end_sec)
        if not original_cues:
            print(f"No captions for {chunk_name}, skipping.")
            atomic_write_json(out_json, [])
            return True

        cues_json_str = json.dumps(original_cues, ensure_ascii=False, indent=2)
        cached = load_cached_captions(out_json, chunk_duration, original_cues, text_mode=text_mode)
        if cached is not None:
            print(f"Skipping {chunk_name} - already processed.")
            return True

        text_instruction = (
            "4. Do NOT change, translate, correct, split, merge, or reorder the original text. Return exactly what was provided in the 'text' field.\n"
            if text_mode == "preserve" else
            "4. You may correct awkward or incorrect English, but keep each caption's meaning faithful to the spoken line or visible text.\n"
            "5. Do not hallucinate missing dialogue, punch up jokes, or guess uncertain names. Prefer conservative transliteration over invented wording.\n"
        )
        prompt = f"""
        You are an expert subtitle aligner.
        Watch this {chunk_duration:.3f}-second video chunk.
        Below is a JSON list of the original subtitles assigned to this chunk.

        Your task:
        1. Fix the timestamps of these captions so they align perfectly with the video.
        2. Crucially, match the timing with WHEN THE TEXT APPEARS VISUALLY ON SCREEN (editors' flair text) OR when the dialogue is spoken.
        3. Preserve every original 'id' exactly once.
        {text_instruction}6. Return a complete list of all captions, ensuring none are dropped.
        7. Keep captions sorted by start time.
        8. Use timestamps relative to this chunk, from 00:00:00.000 to {format_time(chunk_duration)}.
        9. Some captions were assigned by midpoint because they may straddle chunk boundaries. Use the visible audio/video in this chunk to place them correctly.

        Return the result as a JSON object matching the required schema with a 'captions' array.

        Original Captions:
        {cues_json_str}
        """
    else:
        # Generation Mode
        cached = load_cached_captions(out_json, chunk_duration, None)
        if cached is not None:
            print(f"Skipping {chunk_name} - already processed.")
            return True

        prompt = f"""
        You are an expert subtitle generator and translator.
        Watch this {chunk_duration:.3f}-second video chunk.

        Your task:
        1. Generate accurate English subtitles for the dialogue and any relevant on-screen text.
        2. Create accurate timestamps for each caption relative to the start of this chunk (ranging from 00:00:00.000 to {format_time(chunk_duration)}).
        3. Use natural English translations when dialogue is not English.
        4. Do not summarize, explain, or infer missing dialogue.
        5. Include meaningful on-screen text when it matters for understanding the video.
        6. Ignore decorative text, logos, watermarks, and unrelated UI.
        7. Use sequential integer IDs starting at 0.
        8. Keep captions sorted by start time and do not overlap them.
        9. Split long speech into readable captions.

        Return the result as a JSON object matching the required schema with a 'captions' array.
        """

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()
        if len(video_data) > INLINE_VIDEO_WARNING_BYTES:
            print(
                f"[Worker-{chunk_idx:03d}] Warning: {chunk_name} is {len(video_data) / 1024 / 1024:.1f} MB. "
                "Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail."
            )

        mode_str = "aligning" if vtt_file else "generating"
        print(f"[Worker-{chunk_idx:03d}] {mode_str.capitalize()} {chunk_name} using Gemini API...")

        with create_client(api_key, base_url) as client:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=video_data, mime_type=chunk_mime),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=AlignmentResponse,
                )
            )

        parsed_response = AlignmentResponse.model_validate_json(response.text)
        validated = validate_captions(
            parsed_response.captions,
            chunk_duration,
            original_cues if vtt_file else None,
            text_mode=text_mode,
        )
        atomic_write_json(out_json, validated)

        print(f"[Worker-{chunk_idx:03d}] Finished {chunk_name}.")
        return True
    except Exception as e:
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {chunk_name}: {e}")
        return False

def stitch(chunk_dir, output_vtt):
    print("Stitching chunks into final VTT...")
    final_vtt = webvtt.WebVTT()
    captions_to_write = []

    chunks = list_chunks(chunk_dir)
    offset_map = {c["idx"]: c["start"] for c in chunks}

    json_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('aligned_chunk_') and f.endswith('.json')])

    for json_name in json_files:
        chunk_idx = int(json_name.replace("aligned_chunk_", "").replace(".json", ""))
        offset_sec = offset_map.get(chunk_idx, 0.0)

        with open(os.path.join(chunk_dir, json_name), "r") as f:
            captions = json.load(f)

        for cap in captions:
            abs_start = parse_time(cap["start"]) + offset_sec
            abs_end = parse_time(cap["end"]) + offset_sec
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

def main():
    parser = argparse.ArgumentParser(description="Align or Generate VTT subtitles for a video using Gemini API.")
    parser.add_argument("video_file", help="Path to the original video file (e.g. .webm, .mp4)")
    parser.add_argument("vtt_file", nargs="?", default=None, help="Path to the original VTT subtitle file (optional). If omitted, generates from scratch.")
    parser.add_argument("--output", "-o", default="output_subtitles.vtt", help="Output path for the generated/aligned VTT file")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--base-url", default=os.environ.get("GEMINI_API_BASE"), help="Base URL for Gemini API (optional)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"), help="Gemini model to use")
    parser.add_argument("--chunk-dur", type=int, default=60, help="Chunk duration in seconds (default: 60)")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent API workers")
    parser.add_argument("--text-mode", choices=["preserve", "fix"], default="preserve", help="Whether alignment mode preserves original subtitle text or lets the model fix awkward translation")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep the per-input work directory after successful processing")

    args = parser.parse_args()

    if args.chunk_dur <= 0:
        print("Error: --chunk-dur must be greater than 0")
        sys.exit(1)

    if args.workers <= 0:
        print("Error: --workers must be greater than 0")
        sys.exit(1)

    if not args.vtt_file and args.text_mode != "preserve":
        print("Warning: --text-mode is ignored in generation mode because no VTT file was provided.")

    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if args.vtt_file and not os.path.exists(args.vtt_file):
        print(f"Error: VTT file not found: {args.vtt_file}")
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

        # 2. Process chunks concurrently
        print(f"Processing {len(chunks)} chunks using {args.workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_chunk,
                    args.api_key, args.base_url,
                    chunk["idx"], chunk["name"], chunk_dir,
                    args.vtt_file, chunk["start"], chunk["end"],
                    chunk["duration"], args.model, manifest["chunk_mime"], args.text_mode if args.vtt_file else "preserve"
                ): chunk["name"]
                for chunk in chunks
            }
            failed = []
            for future in concurrent.futures.as_completed(futures):
                chunk_name = futures[future]
                try:
                    if not future.result():
                        failed.append(chunk_name)
                except Exception as e:
                    print(f"ERROR processing {chunk_name}: {e}")
                    failed.append(chunk_name)

            if failed:
                raise RuntimeError(
                    f"Failed to process {len(failed)} chunk(s): {', '.join(sorted(failed))}. "
                    f"Keeping {chunk_dir} so you can retry."
                )

        # 3. Stitch chunks together
        stitch(chunk_dir, args.output)
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
