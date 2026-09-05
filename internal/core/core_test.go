package core

import (
	"reflect"
	"testing"

	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

func TestTimeAndCaptionContracts(t *testing.T) {
	for input, want := range map[string]float64{"0": 0, "1.25": 1.25, "12:34": 754, "02:03,5": 123.5, "01:02:03.004": 3723.004} {
		got, err := ParseTime(input)
		if err != nil || got != want {
			t.Fatalf("ParseTime(%q) = %v, %v", input, got, err)
		}
	}
	for _, input := range []string{"-0.1", "1:2:3:4", "abc", "", "1..5"} {
		if _, err := ParseTime(input); err == nil {
			t.Errorf("ParseTime(%q) succeeded", input)
		}
	}
	if got := MustFormatTime(3661.2346); got != "01:01:01.235" {
		t.Fatalf("format = %s", got)
	}
	caps, err := ValidateCaptions([]Caption{{ID: 2, Start: "1", End: "3", Text: "Later"}, {ID: 1, Start: "00:00:00,250", End: "2", Text: "Earlier"}}, 5)
	if err != nil || caps[0].ID != 1 || caps[0].Start != "00:00:00.250" {
		t.Fatalf("captions = %#v, %v", caps, err)
	}
}

func TestTitlesURLsBracketsAndUnicodeSpeakerCasing(t *testing.T) {
	if got := DeriveSourceTitle("full/path/movie.webm.en.vtt"); got != "movie" {
		t.Fatal(got)
	}
	urls, err := ValidateContextURLs([]string{"https://example.com/a", "https://example.com/a", "https://youtu.be/abc?t=2"})
	if err != nil || len(urls) != 2 {
		t.Fatalf("urls = %v, %v", urls, err)
	}
	if !IsYouTubeURL(urls[1]) || IsYouTubeURL("https://youtu.be/a/b") {
		t.Fatal("YouTube classification mismatch")
	}
	text := "Host: Ready\n[[Mission Rule] Two chances]"
	if ClassifyCueText(text) != "mixed" || !reflect.DeepEqual(VisualFragments(text), []string{"[[Mission Rule] Two chances]"}) {
		t.Fatal("bracket parsing mismatch")
	}
	cues := []vtt.Cue{{Text: "AẞA: One"}, {Text: "ASSA: Two"}, {Text: "[Opening title]"}}
	got := CanonicalizeSpeakerCasing(cues, []string{"ASSA"})
	if got[0].Text != "ASSA: One" {
		t.Fatalf("Unicode case fold failed: %q", got[0].Text)
	}
}

func sources() []ScriptEntry {
	return []ScriptEntry{{0, "00:00:00.000", "00:00:05.000", "Host: Welcome.", "dialogue"}, {1, "00:00:05.000", "00:00:10.000", "Host: First topic.", "dialogue"}, {2, "00:00:09.500", "00:00:15.000", "Guest: Hello.", "dialogue"}, {3, "00:00:19.500", "00:00:20.500", "[Title Card]", "editorial"}, {4, "00:00:25.000", "00:00:30.000", "Host: See [Chapter 1] now.", "mixed"}}
}
func TestSparseAudioAuthorityAndVisualIntegrity(t *testing.T) {
	r := AudioRefinementResponse{ContractVersion: AudioRefineResponseContract, DeletedSourceIDs: []int{2}, Cues: []AudioRefinedCue{{[]int{1}, "00:00:05.000", "00:00:09.000", "Host: Refined topic."}, {nil, "00:00:07.000", "00:00:09.000", "Host: Recovered remark."}}}
	got, err := ValidateAudioRefinement(r, sources(), 35, []float64{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 {
		t.Fatalf("got %d cues", len(got))
	}
	bad := AudioRefinementResponse{ContractVersion: AudioRefineResponseContract, Cues: []AudioRefinedCue{{[]int{4}, "00:00:25.000", "00:00:30.000", "Host: See Chapter 1 now."}}}
	if _, err = ValidateAudioRefinement(bad, sources(), 35, []float64{20}); err == nil {
		t.Fatal("visual deletion accepted")
	}
	outside := AudioRefinementResponse{ContractVersion: AudioRefineResponseContract, DeletedSourceIDs: []int{0}}
	got, err = ValidateAudioRefinement(outside, sources(), 35, []float64{21})
	if err != nil || len(got) != 5 {
		t.Fatalf("authority filter = %d, %v", len(got), err)
	}
}
