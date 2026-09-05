package media

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestRealMediaSplitAndAudioExtraction(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg is not installed")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe is not installed")
	}
	dir := t.TempDir()
	video := filepath.Join(dir, "source.mp4")
	command := exec.Command(
		"ffmpeg",
		"-v", "error",
		"-f", "lavfi",
		"-i", "color=c=black:s=320x180:r=25:d=3",
		"-f", "lavfi",
		"-i", "sine=frequency=440:sample_rate=48000:duration=3",
		"-c:v", "libx264",
		"-g", "25",
		"-c:a", "aac",
		"-shortest",
		video,
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("create fixture: %v: %s", err, output)
	}
	extension, mimeType, codec, err := ProbeVideoFormat(video)
	if err != nil || extension != ".mp4" || mimeType != "video/mp4" || codec != "h264" {
		t.Fatalf("probe = %q, %q, %q, %v", extension, mimeType, codec, err)
	}
	work := filepath.Join(dir, "work")
	manifest := map[string]any{"chunk_ext": extension}
	if err := SplitVideo(video, work, 1, manifest); err != nil {
		t.Fatal(err)
	}
	chunks := ListChunks(work)
	if len(chunks) < 2 {
		t.Fatalf("split produced %d chunks", len(chunks))
	}
	for _, chunk := range chunks {
		info, err := os.Stat(filepath.Join(work, chunk.Name))
		if err != nil || info.Size() == 0 {
			t.Fatalf("chunk %s is invalid: %v", chunk.Name, err)
		}
	}
	audio, duration, sourceDuration, reused, err := ExtractAudio(video, work)
	if err != nil {
		t.Fatal(err)
	}
	if reused || !ValidAudio(audio) || duration <= 0 || sourceDuration <= 0 {
		t.Fatalf("audio = %s, %v, %v, %v", audio, duration, sourceDuration, reused)
	}
	_, _, _, reused, err = ExtractAudio(video, work)
	if err != nil || !reused {
		t.Fatalf("valid audio cache was not reused: %v, %v", reused, err)
	}
}
