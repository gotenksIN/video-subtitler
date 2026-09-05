package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

func buildCLI(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	binary := filepath.Join(root, "bin", "video-subtitler")
	build := exec.Command("go", "build", "-o", binary, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v: %s", err, output)
	}
	return root, binary
}

func cliEnvironment() []string {
	var environment []string
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "GEMINI_") {
			environment = append(environment, value)
		}
	}
	return environment
}

func TestCLIParsingContracts(t *testing.T) {
	_, binary := buildCLI(t)
	accepted := exec.Command(binary, "video.mp4", "--output", "out.vtt", "--context-url", "https://example.com", "--context-url=https://two.example", "--help")
	accepted.Env = cliEnvironment()
	if output, err := accepted.CombinedOutput(); err != nil {
		t.Fatalf("flags after input were rejected: %v: %s", err, output)
	}
	for _, test := range []struct {
		name    string
		args    []string
		message string
	}{
		{"thinking level", []string{"video.mp4", "--thinking-level", "extreme"}, "invalid choice"},
		{"workers", []string{"video.mp4", "--workers", "not-a-number"}, "invalid int value"},
		{"chunk duration", []string{"video.mp4", "--chunk-dur=bad"}, "invalid int value"},
		{"empty integer", []string{"video.mp4", "--workers="}, "invalid int value"},
	} {
		t.Run(test.name, func(t *testing.T) {
			command := exec.Command(binary, test.args...)
			command.Env = cliEnvironment()
			output, err := command.CombinedOutput()
			var exitError *exec.ExitError
			if !errors.As(err, &exitError) || exitError.ExitCode() != 2 || !strings.Contains(string(output), test.message) {
				t.Fatalf("syntax error result: %v: %s", err, output)
			}
		})
	}
}

func TestCLIRepositoryConfigurationPrecedence(t *testing.T) {
	root, binary := buildCLI(t)
	for _, test := range []struct {
		name        string
		fileModel   string
		environment bool
		flags       bool
		wantModel   string
		wantSource  string
	}{
		{"model default", "", false, false, "gemini-3.1-pro-preview", "file"},
		{"repository dotenv", "file-model", false, false, "file-model", "file"},
		{"environment overrides dotenv", "file-model", true, false, "env-model", "env"},
		{"flags override environment", "file-model", true, true, "flag-model", "flag"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				requests.Add(1)
				wantPath := "/" + test.wantSource + "/v1beta/models/" + test.wantModel + ":streamGenerateContent"
				if request.URL.Path != wantPath {
					t.Errorf("request path = %q, want %q", request.URL.Path, wantPath)
				}
				if request.Header.Get("x-goog-api-key") != "test-"+test.wantSource {
					t.Error("request used the wrong credential source")
				}
				var body struct {
					Tools []json.RawMessage `json:"tools"`
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
					writer.WriteHeader(http.StatusBadRequest)
					return
				}
				text := `{"changes":[{"id":0,"text":"Corrected"}]}`
				candidate := map[string]any{}
				if len(body.Tools) > 0 {
					text = "PARTICIPANTS AND SPEAKERS:\nJane Doe: Host"
					candidate["groundingMetadata"] = map[string]any{"webSearchQueries": []string{"test"}}
				}
				candidate["content"] = map[string]any{"role": "model", "parts": []any{map[string]any{"text": text}}}
				payload, err := json.Marshal(map[string]any{"candidates": []any{candidate}})
				if err != nil {
					t.Error(err)
					return
				}
				writer.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(writer, "data: %s\n\n", payload)
			}))
			defer server.Close()
			configuration := "GEMINI_API_KEY=test-file\nGEMINI_API_BASE=" + server.URL + "/file\n"
			if test.fileModel != "" {
				configuration += "GEMINI_REFINE_MODEL=" + test.fileModel + "\n"
			}
			if err := os.WriteFile(filepath.Join(root, ".env"), []byte(configuration), 0o600); err != nil {
				t.Fatal(err)
			}
			caller := t.TempDir()
			input := filepath.Join(caller, "input.vtt")
			output := filepath.Join(caller, "output.vtt")
			if err := os.WriteFile(input, []byte("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nOriginal\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			// A caller-local dotenv must not override repository configuration.
			if err := os.WriteFile(filepath.Join(caller, ".env"), []byte("GEMINI_REFINE_MODEL=wrong-model\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{input, "--refine-only", "--output", output}
			if test.flags {
				args = append(args, "--api-key", "test-flag", "--base-url", server.URL+"/flag", "--refine-model", "flag-model")
			}
			command := exec.Command(binary, args...)
			command.Dir = caller
			command.Env = cliEnvironment()
			if test.environment {
				command.Env = append(command.Env, "GEMINI_API_KEY=test-env", "GEMINI_API_BASE="+server.URL+"/env", "GEMINI_REFINE_MODEL=env-model")
			}
			if logs, err := command.CombinedOutput(); err != nil {
				t.Fatalf("CLI failed: %v: %s", err, logs)
			}
			result, err := vtt.Read(output)
			if err != nil || len(result.Cues) != 1 || result.Cues[0].Text != "Corrected" {
				t.Fatalf("output = %#v, %v", result, err)
			}
			if requests.Load() != 2 {
				t.Fatalf("requests = %d, want preflight and text refinement", requests.Load())
			}
		})
	}
}
