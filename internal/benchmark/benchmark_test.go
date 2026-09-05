package benchmark_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gotenksIN/video-subtitler/internal/benchmark"
)

func TestCompareTextAndTimingMetrics(t *testing.T) {
	for _, test := range []struct {
		name      string
		reference string
		generated string
		want      float64
	}{
		{"substitution", "a b c", "a x c", 2.0 / 3.0},
		{"empty", "", "", 1},
		{"repeated sequence", "a b a c", "b a b c", 0.75},
		{"bold markup", "Hello", "<b>Hello</b>", 1},
		{"voice and class markup", "Hello there", "<v Jane><c.highlight>Hello</c> there</v>", 1},
		{"timestamp markup", "Hello there", "Hello <00:00:00.500>there", 1},
		{"speaker label inside markup", "Hello", "<b>Jane Doe: Hello</b>", 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			reference := filepath.Join(dir, "reference.vtt")
			generated := filepath.Join(dir, "generated.vtt")
			writeSubtitle(t, reference, "00:00:00.000", "00:00:02.000", test.reference)
			writeSubtitle(t, generated, "00:00:01.000", "00:00:03.000", test.generated)
			metrics, err := benchmark.Compare(generated, reference)
			if err != nil {
				t.Fatal(err)
			}
			if got := metrics["text_similarity"].(float64); math.Abs(got-test.want) > 1e-15 {
				t.Errorf("text similarity = %.17g, want %.17g", got, test.want)
			}
			if test.reference != "" {
				for name, want := range map[string]float64{
					"reference_active_seconds": 2,
					"generated_active_seconds": 2,
					"temporal_overlap_seconds": 1,
					"temporal_recall":          0.5,
					"temporal_precision":       0.5,
					"temporal_iou":             1.0 / 3.0,
				} {
					if got := metrics[name].(float64); math.Abs(got-want) > 1e-15 {
						t.Errorf("%s = %v, want %v", name, got, want)
					}
				}
			}
		})
	}
}

func writeSubtitle(t *testing.T, path, start, end, text string) {
	t.Helper()
	data := "WEBVTT\n"
	if text != "" {
		data += "\n" + start + " --> " + end + "\n" + text + "\n"
	}
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}
