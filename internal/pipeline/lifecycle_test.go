package pipeline_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gotenksIN/video-subtitler/internal/pipeline"
	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

type pipelineScenario struct {
	mutex     sync.Mutex
	counts    map[string]int
	failStage string
}

func (scenario *pipelineScenario) serve(writer http.ResponseWriter, request *http.Request) {
	var body struct {
		Tools            []json.RawMessage `json:"tools"`
		GenerationConfig struct {
			ResponseJSONSchema json.RawMessage `json:"responseJsonSchema"`
		} `json:"generationConfig"`
		Contents []struct {
			Parts []struct {
				Text       string `json:"text"`
				InlineData *struct {
					MIMEType string `json:"mimeType"`
				} `json:"inlineData"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		writer.WriteHeader(http.StatusBadRequest)
		return
	}
	stage := "text"
	if len(body.Tools) > 0 {
		stage = "preflight"
	} else if len(body.GenerationConfig.ResponseJSONSchema) > 0 {
		stage = "audio"
	} else {
		for _, content := range body.Contents {
			for _, part := range content.Parts {
				if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "video/") {
					stage = "chunk"
				}
			}
		}
	}
	scenario.mutex.Lock()
	defer scenario.mutex.Unlock()
	scenario.counts[stage]++
	if scenario.failStage == stage {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(writer, `{"error":{"code":400,"message":"scenario failure","status":"INVALID_ARGUMENT"}}`)
		return
	}
	text := ""
	candidate := map[string]any{}
	switch stage {
	case "preflight":
		text = "PARTICIPANTS AND SPEAKERS:\nJane Doe: Host"
		candidate["groundingMetadata"] = map[string]any{"webSearchQueries": []string{"test"}}
	case "chunk":
		text = `{"captions":[{"id":0,"start":"00:00:00.000","end":"00:00:00.500","text":"Original"}]}`
	case "audio":
		text = `{"contractVersion":"sparse-patch-v1","cues":[{"sourceIds":[0],"start":"00:00:00.000","end":"00:00:00.500","text":"Audio repaired"}]}`
	case "text":
		inputText := "Original"
		for _, content := range body.Contents {
			for _, part := range content.Parts {
				if strings.Contains(part.Text, "Audio repaired") {
					inputText = "Audio repaired"
				}
			}
		}
		payload, err := json.Marshal(map[string]any{"changes": []any{map[string]any{"id": 0, "text": "Proofread " + inputText}}})
		if err != nil {
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		text = string(payload)
	}
	candidate["content"] = map[string]any{"role": "model", "parts": []any{map[string]any{"text": text}}}
	payload, err := json.Marshal(map[string]any{"candidates": []any{candidate}})
	if err != nil {
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(writer, "data: %s\n\n", payload)
}

func (scenario *pipelineScenario) snapshot() map[string]int {
	scenario.mutex.Lock()
	defer scenario.mutex.Unlock()
	counts := map[string]int{}
	for stage, count := range scenario.counts {
		counts[stage] = count
	}
	return counts
}

func (scenario *pipelineScenario) fail(stage string) {
	scenario.mutex.Lock()
	defer scenario.mutex.Unlock()
	scenario.failStage = stage
}

func sourceVideo(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "source.mp4")
	command := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=3", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264", "-g", "10", "-c:a", "aac", "-shortest", path)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("create media fixture: %v: %s", err, output)
	}
	return path
}

func scenarioConfig(t *testing.T, video string, scenario *pipelineScenario) pipeline.Config {
	t.Helper()
	t.Chdir(t.TempDir())
	server := httptest.NewServer(http.HandlerFunc(scenario.serve))
	t.Cleanup(server.Close)
	return pipeline.Config{
		VideoPath: video, OutputPath: "output.vtt", APIKey: "test-key", BaseURL: server.URL,
		Model: "test-model", RefineModel: "refine-model", AudioRefineModel: "audio-model",
		ChunkDur: 1, Workers: 2, ThinkingLevel: "high", AudioRefine: true, RefineText: true,
	}
}

func TestRunPublishesSelectedRefinementAndCleansWork(t *testing.T) {
	video := sourceVideo(t)
	for _, test := range []struct {
		name  string
		audio bool
		text  bool
		want  string
	}{
		{"both", true, true, "Proofread Audio repaired"},
		{"audio only", true, false, "Audio repaired"},
		{"text only", false, true, "Proofread Original"},
		{"neither", false, false, "Original"},
	} {
		t.Run(test.name, func(t *testing.T) {
			scenario := &pipelineScenario{counts: map[string]int{}}
			config := scenarioConfig(t, video, scenario)
			config.AudioRefine, config.RefineText = test.audio, test.text
			if err := pipeline.Run(context.Background(), config); err != nil {
				t.Fatal(err)
			}
			result, err := vtt.Read(config.OutputPath)
			if err != nil || len(result.Cues) < 2 || result.Cues[0].Text != test.want {
				t.Fatalf("published subtitles = %#v, %v", result, err)
			}
			if result.Cues[0].Start != "00:00:00.000" || result.Cues[0].End != "00:00:00.500" {
				t.Fatalf("refinement changed timing: %#v", result.Cues[0])
			}
			counts := scenario.snapshot()
			wantAudio, wantText := 0, 0
			if test.audio {
				wantAudio = 1
			}
			if test.text {
				wantText = 1
			}
			if counts["preflight"] != 1 || counts["audio"] != wantAudio || counts["text"] != wantText {
				t.Fatalf("unexpected stages: %v", counts)
			}
			work, err := filepath.Glob(filepath.Join(pipeline.ChunkRoot, "*", "*"))
			if err != nil {
				t.Fatal(err)
			}
			for _, path := range work {
				if filepath.Base(path) != ".lock" {
					t.Errorf("successful run retained %s", path)
				}
			}
		})
	}
}

func TestRunFailurePreservesOutputAndResumesCachedStages(t *testing.T) {
	video := sourceVideo(t)
	for _, stage := range []string{"chunk", "audio", "text"} {
		t.Run(stage, func(t *testing.T) {
			scenario := &pipelineScenario{counts: map[string]int{}, failStage: stage}
			config := scenarioConfig(t, video, scenario)
			if err := os.WriteFile(config.OutputPath, []byte("previous output"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := pipeline.Run(context.Background(), config); err == nil {
				t.Fatal("failed stage was reported as successful")
			}
			output, err := os.ReadFile(config.OutputPath)
			if err != nil || string(output) != "previous output" {
				t.Fatalf("failed run changed previous output: %q, %v", output, err)
			}
			before := scenario.snapshot()
			filesBefore := resumableMedia(t)
			if len(filesBefore) < 3 {
				t.Fatalf("failed run lost media artifacts: %v", filesBefore)
			}
			// Fail again at text refinement to observe reused media before cleanup.
			scenario.fail("text")
			if err := pipeline.Run(context.Background(), config); err == nil {
				t.Fatal("text failure was reported as successful")
			}
			if after := resumableMedia(t); !reflect.DeepEqual(filesBefore, after) {
				t.Fatalf("retry regenerated cached media: before %v, after %v", filesBefore, after)
			}
			after := scenario.snapshot()
			if after["preflight"] != before["preflight"] {
				t.Fatal("retry repeated cached preflight")
			}
			if stage != "chunk" && after["chunk"] != before["chunk"] {
				t.Fatal("retry repeated valid chunk requests")
			}
			if stage == "text" && after["audio"] != before["audio"] {
				t.Fatal("retry repeated cached audio refinement")
			}
			scenario.fail("")
			if err := pipeline.Run(context.Background(), config); err != nil {
				t.Fatal(err)
			}
			final := scenario.snapshot()
			for _, cachedStage := range []string{"preflight", "chunk", "audio"} {
				if final[cachedStage] != after[cachedStage] {
					t.Errorf("final retry repeated %s", cachedStage)
				}
			}
			result, err := vtt.Read(config.OutputPath)
			if err != nil || len(result.Cues) < 2 || result.Cues[0].Text != "Proofread Audio repaired" {
				t.Fatalf("resumed output = %#v, %v", result, err)
			}
		})
	}
}

func resumableMedia(t *testing.T) map[string]int64 {
	t.Helper()
	files := map[string]int64{}
	for _, pattern := range []string{"chunk_*.mp4", "extracted_audio.ogg"} {
		paths, err := filepath.Glob(filepath.Join(pipeline.ChunkRoot, "*", pattern))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			files[path] = info.ModTime().UnixNano()
		}
	}
	return files
}
