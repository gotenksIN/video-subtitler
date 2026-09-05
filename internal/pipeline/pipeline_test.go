package pipeline

import (
	"github.com/gotenksIN/video-subtitler/internal/vtt"
	"os"
	"path/filepath"
	"testing"
)

func TestLockCleanupAndStitching(t *testing.T) {
	dir := t.TempDir()
	l, err := AcquireLock(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = AcquireLock(dir); err == nil {
		t.Fatal("second lock acquired")
	}
	if err = os.WriteFile(filepath.Join(dir, "artifact"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err = CleanCompleted(dir); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, ".lock")); err != nil {
		t.Fatal("lock inode removed")
	}
	l.Close()
	l2, err := AcquireLock(dir)
	if err != nil {
		t.Fatal(err)
	}
	l2.Close()
	if err = os.WriteFile(filepath.Join(dir, "segments.csv"), []byte("chunk_000.mp4,0,20\nchunk_001.mp4,20,40\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "subtitle_chunk_000.json"), []byte(`[{"id":0,"start":"00:00:18.000","end":"00:00:20.100","text":"[Banner]"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "subtitle_chunk_001.json"), []byte(`[{"id":0,"start":"00:00:00.000","end":"00:00:02.000","text":"[Banner]"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out.vtt")
	if err = Stitch(dir, out); err != nil {
		t.Fatal(err)
	}
	f, err := vtt.Read(out)
	if err != nil || len(f.Cues) != 1 || f.Cues[0].Start != "00:00:18.000" || f.Cues[0].End != "00:00:22.000" {
		t.Fatalf("stitched = %#v, %v", f, err)
	}
}
