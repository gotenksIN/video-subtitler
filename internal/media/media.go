package media

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/gotenksIN/video-subtitler/internal/storage"
)

const AudioMIMEType = "audio/ogg"
const ExtractedAudioName = "extracted_audio.ogg"
const splitMarker = ".split_complete"

var chunkRE = regexp.MustCompile(`^chunk_\d+\.(mp4|webm)$`)

type Chunk struct {
	Idx                  int
	Name                 string
	Start, End, Duration float64
}

func runOutput(name string, args ...string) (string, error) {
	command := exec.Command(name, args...)
	output, err := command.Output()
	return strings.TrimSpace(string(output)), err
}
func ProbeVideoFormat(path string) (ext, mime, codec string, err error) {
	codecName, err := runOutput(
		"ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=codec_name",
		"-of", "default=noprint_wrappers=1:nokey=1",
		path,
	)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to probe video format: %w", err)
	}
	switch strings.ToLower(codecName) {
	case "vp9":
		return ".webm", "video/webm", "vp9", nil
	case "h264":
		return ".mp4", "video/mp4", "h264", nil
	case "hevc", "h265":
		return ".mp4", "video/mp4", "hevc", nil
	}
	return "", "", "", fmt.Errorf("video format not supported: %s", path)
}
func FormatDuration(path string) float64 {
	output, err := runOutput("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path)
	if err != nil {
		return 0
	}
	duration, err := strconv.ParseFloat(output, 64)
	if err != nil || math.IsNaN(duration) || math.IsInf(duration, 0) || duration <= 0 {
		return 0
	}
	return duration
}

type audioProbe struct {
	Streams []struct {
		CodecName  string `json:"codec_name"`
		SampleRate string `json:"sample_rate"`
		Channels   int    `json:"channels"`
	} `json:"streams"`
}

func audioStreams(path string) []struct {
	CodecName  string `json:"codec_name"`
	SampleRate string `json:"sample_rate"`
	Channels   int    `json:"channels"`
} {
	output, err := runOutput(
		"ffprobe",
		"-v", "error",
		"-select_streams", "a",
		"-show_entries", "stream=codec_name,sample_rate,channels",
		"-of", "json",
		path,
	)
	if err != nil {
		return nil
	}
	var probe audioProbe
	if json.Unmarshal([]byte(output), &probe) != nil {
		return nil
	}
	return probe.Streams
}
func HasAudio(path string) bool { return len(audioStreams(path)) > 0 }
func ValidAudio(path string) bool {
	streams := audioStreams(path)
	return FormatDuration(path) > 0 &&
		len(streams) == 1 &&
		streams[0].CodecName == "opus" &&
		streams[0].SampleRate == "48000" &&
		streams[0].Channels == 1
}
func ExtractAudio(video, work string) (string, float64, float64, bool, error) {
	sourceDuration := FormatDuration(video)
	if sourceDuration <= 0 {
		return "", 0, 0, false, fmt.Errorf("failed to determine source media duration")
	}
	target := filepath.Join(work, ExtractedAudioName)
	if _, err := os.Stat(target); err == nil && ValidAudio(target) {
		duration := FormatDuration(target)
		if math.Abs(duration-sourceDuration) <= 2 {
			fmt.Println("Complete audio already exists, skipping extraction.")
			return target, duration, sourceDuration, true, nil
		}
		fmt.Println("Cached extracted audio is inconsistent with the source; removing it.")
		if err := os.Remove(target); err != nil {
			return "", 0, sourceDuration, false, err
		}
	}
	temporaryPath := target + ".tmp"
	if err := os.Remove(temporaryPath); err != nil && !os.IsNotExist(err) {
		return "", 0, sourceDuration, false, err
	}
	fmt.Println("Extracting complete mono Ogg Opus audio...")
	command := exec.Command(
		"ffmpeg",
		"-y",
		"-i", video,
		"-map", "0:a:0",
		"-vn",
		"-sn",
		"-c:a", "libopus",
		"-b:a", "64k",
		"-ac", "1",
		"-ar", "48000",
		"-f", "ogg",
		temporaryPath,
	)
	command.Stdout = nil
	command.Stderr = nil
	if err := command.Run(); err != nil {
		_ = os.Remove(temporaryPath)
		return "", 0, sourceDuration, false, fmt.Errorf("failed to extract complete audio; the source may not contain an audio stream")
	}
	duration := FormatDuration(temporaryPath)
	if !ValidAudio(temporaryPath) || math.Abs(duration-sourceDuration) > 2 {
		_ = os.Remove(temporaryPath)
		return "", 0, sourceDuration, false, fmt.Errorf("extracted audio validation failed")
	}
	if err := os.Rename(temporaryPath, target); err != nil {
		return "", 0, sourceDuration, false, err
	}
	fmt.Println("Complete audio extraction finished.")
	return target, duration, sourceDuration, false, nil
}

func ListChunks(dir string) []Chunk {
	contents, err := os.ReadFile(filepath.Join(dir, "segments.csv"))
	if err != nil {
		return nil
	}
	chunks := []Chunk{}
	seen := map[string]bool{}
	for index, line := range strings.Split(string(contents), "\n") {
		row := strings.TrimSpace(line)
		if row == "" {
			continue
		}
		fields := strings.Split(row, ",")
		if len(fields) < 3 || !chunkRE.MatchString(fields[0]) || seen[fields[0]] {
			return nil
		}
		start, startErr := strconv.ParseFloat(fields[1], 64)
		end, endErr := strconv.ParseFloat(fields[2], 64)
		if startErr != nil || endErr != nil ||
			math.IsNaN(start) || math.IsNaN(end) ||
			math.IsInf(start, 0) || math.IsInf(end, 0) ||
			start < 0 || end <= start {
			return nil
		}
		chunks = append(chunks, Chunk{
			Idx:      index,
			Name:     fields[0],
			Start:    start,
			End:      end,
			Duration: end - start,
		})
		seen[fields[0]] = true
	}
	return chunks
}
func CleanIncomplete(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	subRE := regexp.MustCompile(`^subtitle_chunk_\d+\.json(\.tmp)?$`)
	for _, entry := range entries {
		name := entry.Name()
		if chunkRE.MatchString(name) || subRE.MatchString(name) || name == "segments.csv" {
			if err := os.Remove(filepath.Join(dir, name)); err != nil {
				return err
			}
		}
	}
	return nil
}
func SplitVideo(video, dir string, duration int, manifest map[string]any) error {
	fmt.Printf("Splitting video into %d-second chunks (stream copy mode)...\n", duration)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := storage.AtomicWriteJSON(filepath.Join(dir, storage.ManifestName), manifest); err != nil {
		return err
	}
	chunks := ListChunks(dir)
	listed := map[string]bool{}
	valid := len(chunks) > 0
	for _, chunk := range chunks {
		listed[chunk.Name] = true
		info, err := os.Stat(filepath.Join(dir, chunk.Name))
		if err != nil || info.Size() <= 0 {
			valid = false
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if chunkRE.MatchString(entry.Name()) && !listed[entry.Name()] {
			valid = false
		}
	}
	marker := filepath.Join(dir, splitMarker)
	if _, err := os.Stat(marker); err == nil && valid {
		fmt.Println("Chunks already exist, skipping splitting.")
		return nil
	}
	if err := os.Remove(marker); err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := CleanIncomplete(dir); err != nil {
		return err
	}
	extension := manifest["chunk_ext"].(string)
	command := exec.Command(
		"ffmpeg",
		"-y",
		"-i", video,
		"-map", "0:v:0",
		"-map", "0:a?",
		"-sn",
		"-c", "copy",
		"-f", "segment",
		"-segment_time", strconv.Itoa(duration),
		"-segment_list", filepath.Join(dir, "segments.csv"),
		"-reset_timestamps", "1",
		filepath.Join(dir, "chunk_%03d"+extension),
	)
	if err := command.Run(); err != nil {
		return err
	}
	if err := os.WriteFile(marker, []byte("ok\n"), 0o644); err != nil {
		return err
	}
	fmt.Println("Splitting complete.")
	return nil
}
