package vtt

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestReadWritePreservesUnicodeMultilineAndSettings(t *testing.T) {
	p := filepath.Join(t.TempDir(), "in.vtt")
	raw := "\ufeffWEBVTT title\n\nNOTE ignored\nmetadata\n\ncue-id\n00:00:01.000 --> 00:00:02.500 align:start position:10%\n안녕 😀\nSecond line\n\n"
	if err := os.WriteFile(p, []byte(raw), 0644); err != nil {
		t.Fatal(err)
	}
	f, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Cues) != 1 || f.Cues[0].Text != "안녕 😀\nSecond line" || f.Cues[0].Settings != "align:start position:10%" {
		t.Fatalf("cue = %#v", f.Cues)
	}
	out := filepath.Join(filepath.Dir(p), "out.vtt")
	if err = f.SaveAtomic(out); err != nil {
		t.Fatal(err)
	}
	again, err := Read(out)
	if err != nil || again.Cues[0].Text != f.Cues[0].Text {
		t.Fatalf("round trip: %#v %v", again, err)
	}
}

func TestRoundTripPreservesStyles(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input.vtt")
	output := filepath.Join(dir, "output.vtt")
	raw := "WEBVTT\n\nSTYLE\n::cue { color: lime; }\n::cue(b) { font-weight: bold; }\n\nSTYLE\n::cue(.warning) { color: red; }\n\ncue-id\n00:00:00.000 --> 00:00:01.000 align:start\n<b>Hello</b>\n"
	if err := os.WriteFile(input, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := Read(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.SaveAtomic(output); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, style := range []string{"STYLE\n::cue { color: lime; }\n::cue(b) { font-weight: bold; }", "STYLE\n::cue(.warning) { color: red; }"} {
		if !strings.Contains(string(data), style) {
			t.Fatalf("subtitle style was lost:\n%s", data)
		}
	}
	again, err := Read(output)
	if err != nil || !reflect.DeepEqual(again.Cues, file.Cues) {
		t.Fatalf("cue content changed: %#v, %v", again, err)
	}
}
