import os
import sys
import json
import argparse
import subprocess
import shutil
import webvtt
import concurrent.futures
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from datetime import timedelta
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

def parse_time(time_str):
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s_ms = parts
    elif len(parts) == 2:
        h = "0"
        m, s_ms = parts
    else:
        h, m = "0", "0"
        s_ms = parts[0]

    s, ms = s_ms.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def format_time(seconds):
    td = timedelta(seconds=seconds)
    time_str = str(td)
    if "." in time_str:
        base, ms = time_str.split(".")
        ms = ms[:3].ljust(3, "0")
    else:
        base = time_str
        ms = "000"

    parts = base.split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h = "0"
        m, s = parts

    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms}"

def split_video(video_file, chunk_dir, chunk_dur_sec):
    print(f"Splitting video into {chunk_dur_sec}-second chunks (this might take a minute)...")
    os.makedirs(chunk_dir, exist_ok=True)
    if len([f for f in os.listdir(chunk_dir) if f.endswith(('.webm', '.mp4'))]) > 0:
        print("Chunks already exist, skipping splitting.")
        return

    cmd = [
        "ffmpeg", "-y", "-i", video_file,
        "-f", "segment", "-segment_time", str(chunk_dur_sec),
        "-c", "copy", os.path.join(chunk_dir, "chunk_%03d.webm")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Splitting complete.")

def get_captions_for_chunk(vtt_path, chunk_idx, chunk_dur_sec):
    start_sec = chunk_idx * chunk_dur_sec
    end_sec = (chunk_idx + 1) * chunk_dur_sec

    vtt = webvtt.read(vtt_path)
    chunk_cues = []

    for i, caption in enumerate(vtt):
        cap_start = parse_time(caption.start)
        if start_sec <= cap_start < end_sec:
            rel_start = cap_start - start_sec
            rel_end = parse_time(caption.end) - start_sec
            chunk_cues.append({
                "id": i,
                "start": format_time(rel_start),
                "end": format_time(rel_end),
                "text": caption.text
            })
    return chunk_cues

def process_chunk(client, chunk_idx, chunk_name, chunk_dir, vtt_file, chunk_dur_sec, model_name):
    out_json = os.path.join(chunk_dir, f"aligned_chunk_{chunk_idx:03d}.json")
    if os.path.exists(out_json):
        print(f"Skipping {chunk_name} - already processed.")
        return True

    chunk_path = os.path.join(chunk_dir, chunk_name)

    if vtt_file:
        # Alignment Mode
        original_cues = get_captions_for_chunk(vtt_file, chunk_idx, chunk_dur_sec)
        if not original_cues:
            print(f"No captions for {chunk_name}, skipping.")
            with open(out_json, "w") as f:
                json.dump([], f)
            return True

        cues_json_str = json.dumps(original_cues, ensure_ascii=False, indent=2)

        prompt = f"""
        You are an expert subtitle aligner.
        Watch this 1-minute video chunk.
        Below is a JSON list of the original English subtitles.

        Your task:
        1. Fix the timestamps of these captions so they align perfectly with the video.
        2. Crucially, match the timing with WHEN THE TEXT APPEARS VISUALLY ON SCREEN (editors' flair text) OR when the dialogue is spoken.
        3. Do NOT change the original text. Return exactly what was provided in the 'text' field.
        4. Return a complete list of all captions, ensuring none are dropped.

        Original Captions:
        {cues_json_str}
        """
    else:
        # Generation Mode
        prompt = f"""
        You are an expert subtitle generator and translator.
        Watch this 1-minute video chunk.

        Your task:
        1. Generate accurate English subtitles for the dialogue and any relevant on-screen text.
        2. Create accurate timestamps for each caption relative to the start of this 1-minute chunk (ranging from 00:00:00.000 to 00:01:00.000).
        3. Return a complete JSON list of all generated captions with their translated 'text'.
        """

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()

        mode_str = "aligning" if vtt_file else "generating"
        print(f"[Worker-{chunk_idx:03d}] {mode_str.capitalize()} {chunk_name} using Gemini API...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=video_data, mime_type='video/webm'),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=AlignmentResponse,
            )
        )

        parsed_response = json.loads(response.text)
        with open(out_json, "w") as f:
            json.dump(parsed_response.get("captions", []), f, ensure_ascii=False, indent=2)

        print(f"[Worker-{chunk_idx:03d}] Finished {chunk_name}.")
        return True
    except Exception as e:
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {chunk_name}: {e}")
        return False

def stitch(chunk_dir, chunk_dur_sec, output_vtt):
    print("Stitching chunks into final VTT...")
    final_vtt = webvtt.WebVTT()

    json_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('aligned_chunk_') and f.endswith('.json')])

    for json_name in json_files:
        chunk_idx = int(json_name.replace("aligned_chunk_", "").replace(".json", ""))
        offset_sec = chunk_idx * chunk_dur_sec

        with open(os.path.join(chunk_dir, json_name), "r") as f:
            captions = json.load(f)

        for cap in captions:
            try:
                abs_start = parse_time(cap["start"]) + offset_sec
                abs_end = parse_time(cap["end"]) + offset_sec

                vtt_caption = webvtt.Caption(
                    format_time(abs_start),
                    format_time(abs_end),
                    cap["text"]
                )
                final_vtt.captions.append(vtt_caption)
            except Exception as e:
                print(f"Error parsing caption in {json_name}: {e}")

    final_vtt.save(output_vtt)
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
    parser.add_argument("--keep-chunks", action="store_true", help="Keep the temporary chunk directory after processing")

    args = parser.parse_args()

    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if args.vtt_file and not os.path.exists(args.vtt_file):
        print(f"Error: VTT file not found: {args.vtt_file}")
        sys.exit(1)

    if args.base_url:
        client = genai.Client(
            api_key=args.api_key,
            http_options=types.HttpOptions(base_url=args.base_url)
        )
    else:
        client = genai.Client(
            api_key=args.api_key
        )

    chunk_dir = "temp_video_chunks"

    try:
        # 1. Split Video
        split_video(args.video_file, chunk_dir, args.chunk_dur)

        chunks = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith(('.webm', '.mp4'))])

        # 2. Process chunks concurrently
        print(f"Processing {len(chunks)} chunks using {args.workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_chunk, client, i, name, chunk_dir,
                    args.vtt_file, args.chunk_dur, args.model
                )
                for i, name in enumerate(chunks)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # 3. Stitch chunks together
        stitch(chunk_dir, args.chunk_dur, args.output)

    finally:
        # 4. Cleanup
        if not args.keep_chunks and os.path.exists(chunk_dir):
            print(f"Cleaning up temporary directory: {chunk_dir}")
            shutil.rmtree(chunk_dir)

if __name__ == "__main__":
    main()
