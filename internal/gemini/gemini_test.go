package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gotenksIN/video-subtitler/internal/core"
	"github.com/gotenksIN/video-subtitler/internal/media"
	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

type scriptedResponse struct {
	Text                   string
	Grounded               bool
	RetrievedURL           string
	URLRetrievalSuccessful bool
}

func newScriptedServer(t *testing.T, responses []scriptedResponse) (*httptest.Server, *[]string) {
	t.Helper()
	requests := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Error(err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		requests = append(requests, string(body))
		index := len(requests) - 1
		if index >= len(responses) {
			t.Errorf("unexpected Gemini request %d", index)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		response := responses[index]
		candidate := map[string]any{
			"content": map[string]any{
				"role":  "model",
				"parts": []map[string]any{{"text": response.Text}},
			},
		}
		if response.Grounded {
			candidate["groundingMetadata"] = map[string]any{
				"webSearchQueries": []string{"test query"},
				"groundingChunks": []map[string]any{{
					"web": map[string]any{"title": "Source", "uri": "https://source.example"},
				}},
			}
		}
		if response.RetrievedURL != "" {
			status := "URL_RETRIEVAL_STATUS_ERROR"
			if response.URLRetrievalSuccessful {
				status = "URL_RETRIEVAL_STATUS_SUCCESS"
			}
			candidate["urlContextMetadata"] = map[string]any{
				"urlMetadata": []map[string]any{{
					"retrievedUrl":       response.RetrievedURL,
					"urlRetrievalStatus": status,
				}},
			}
		}
		payload, err := json.Marshal(map[string]any{"candidates": []any{candidate}})
		if err != nil {
			t.Error(err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(writer, "data: %s\n\n", payload)
	}))
	return server, &requests
}

func TestChunkAdapterSendsContractAndIgnoresThoughtText(t *testing.T) {
	var requestPath string
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requestPath = request.URL.Path
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Error(err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		requestBody = string(body)
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"{\\\"captions\\\":[]}\"},{\"text\":\"not-json\",\"thought\":true}]}}]}\n\n")
	}))
	defer server.Close()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chunk_000.mp4"), []byte("video"), 0o644); err != nil {
		t.Fatal(err)
	}
	chunk := media.Chunk{Idx: 0, Name: "chunk_000.mp4", Duration: 10}
	ok := ProcessChunk(
		context.Background(),
		"test-key",
		server.URL+"/proxy/v1beta",
		dir,
		chunk,
		"test-model",
		"video/mp4",
		"high",
		"Title",
		[]string{"Jane Doe"},
	)
	if !ok {
		t.Fatal("chunk request failed")
	}
	if requestPath != "/proxy/v1beta/models/test-model:streamGenerateContent" {
		t.Fatalf("request path = %q", requestPath)
	}
	for _, required := range []string{
		`"inlineData"`,
		`"mimeType":"video/mp4"`,
		`"responseMimeType":"application/json"`,
		`"thinkingLevel":"HIGH"`,
		`"includeThoughts":true`,
		`"responseSchema"`,
	} {
		if !strings.Contains(requestBody, required) {
			t.Errorf("request does not contain %s: %s", required, requestBody)
		}
	}
	if strings.Contains(requestBody, "functionDeclarations") {
		t.Fatalf("request unexpectedly enables client functions: %s", requestBody)
	}
	if _, ok := LoadCachedCaptions(filepath.Join(dir, "subtitle_chunk_000.json"), 10); !ok {
		t.Fatal("published chunk cache is invalid")
	}
}

func TestPreflightRejectsIncompleteCache(t *testing.T) {
	for _, data := range []string{
		`null`,
		`{}`,
		`{"contract_version":"preflight-v1"}`,
		`{"contract_version":"preflight-v1","identity_context":"","terminology_context":"","grounded_names":null}`,
	} {
		dir := t.TempDir()
		path := filepath.Join(dir, PreflightFilename)
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, ok := LoadPreflight(dir); ok {
			t.Fatalf("incomplete cache was accepted: %s", data)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("invalid cache was not discarded: %v", err)
		}
	}
}

func TestAudioRefinementReusesCacheAndInvalidatesChangedAudio(t *testing.T) {
	for _, test := range []struct {
		name     string
		response string
	}{
		{"omitted deletions", `{"contractVersion":"sparse-patch-v1","cues":[]}`},
		{"empty deletions", `{"contractVersion":"sparse-patch-v1","deletedSourceIds":[],"cues":[]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			server, requests := newScriptedServer(t, []scriptedResponse{{Text: test.response}, {Text: test.response}})
			defer server.Close()
			dir := t.TempDir()
			script := filepath.Join(dir, "stitched.vtt")
			audio := filepath.Join(dir, "audio.ogg")
			output := filepath.Join(dir, "output.vtt")
			if err := os.WriteFile(script, []byte("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(audio, []byte("audio"), 0o644); err != nil {
				t.Fatal(err)
			}
			run := func() {
				t.Helper()
				if err := BoundaryAudioRefine(context.Background(), script, audio, dir, output, "test-key", server.URL, "test-model", "Title", 1, []float64{}); err != nil {
					t.Fatal(err)
				}
				result, err := vtt.Read(output)
				if err != nil || len(result.Cues) != 1 || result.Cues[0].Text != "Hello" {
					t.Fatalf("audio result = %#v, %v", result, err)
				}
			}
			run()
			if err := os.Remove(output); err != nil {
				t.Fatal(err)
			}
			run()
			if len(*requests) != 1 {
				t.Fatalf("unchanged inputs made %d requests, want 1", len(*requests))
			}
			if err := os.WriteFile(audio, []byte("changed audio"), 0o644); err != nil {
				t.Fatal(err)
			}
			run()
			if len(*requests) != 2 {
				t.Fatalf("changed audio made %d total requests, want 2", len(*requests))
			}
		})
	}
}

func TestPreflightAdaptersUseSearchURLContextAndDirectYouTube(t *testing.T) {
	ordinaryURL := "https://context.example/article"
	server, requests := newScriptedServer(t, []scriptedResponse{
		{
			Text:                   "PARTICIPANTS AND SPEAKERS:\nJane Doe: Host\nTOPIC TERMINOLOGY AND PROPER NOUNS:\nProgram: Title",
			Grounded:               true,
			RetrievedURL:           ordinaryURL,
			URLRetrievalSuccessful: true,
		},
		{Text: "Jane Doe: visible title at 00:00:01"},
	})
	defer server.Close()
	preflight, err := RunPreflight(
		context.Background(),
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"high",
		"Source",
		[]string{ordinaryURL, "https://youtu.be/video"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(preflight.GroundedNames) != 1 || preflight.GroundedNames[0] != "Jane Doe" {
		t.Fatalf("preflight names = %v", preflight.GroundedNames)
	}
	if preflight.YouTubeContext == nil || *preflight.YouTubeContext == "" {
		t.Fatal("direct YouTube context is empty")
	}
	if len(*requests) != 2 {
		t.Fatalf("requests = %d", len(*requests))
	}
	for _, required := range []string{`"googleSearch"`, `"urlContext"`, `"includeThoughts":true`} {
		if !strings.Contains((*requests)[0], required) {
			t.Errorf("research request lacks %s: %s", required, (*requests)[0])
		}
	}
	if !strings.Contains((*requests)[1], `"fileUri":"https://youtu.be/video"`) {
		t.Errorf("YouTube request lacks direct URI: %s", (*requests)[1])
	}
	if strings.Contains((*requests)[1], `"tools"`) {
		t.Errorf("YouTube request unexpectedly contains tools: %s", (*requests)[1])
	}
}

func TestPreflightAcceptsLeadingBlankCRLFSSE(t *testing.T) {
	thought, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{
				"role": "model",
				"parts": []any{map[string]any{
					"text":    "SHOULD NOT APPEAR",
					"thought": true,
				}},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	answer, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{
				"role": "model",
				"parts": []any{map[string]any{
					"text": "PARTICIPANTS AND SPEAKERS:\nJane Doe: Host\nTOPIC TERMINOLOGY AND PROPER NOUNS:\nProgram: Title",
				}},
			},
			"groundingMetadata": map[string]any{
				"webSearchQueries": []string{"test query"},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := append([]byte("\r\ndata: "), thought...)
	body = append(body, []byte("\r\n\r\n\r\ndata: ")...)
	body = append(body, answer...)
	body = append(body, []byte("\r\n\r\n")...)

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			t.Error("response writer does not support flushing")
			return
		}
		chunkSizes := []int{1, 2, 7, 3, 11}
		for offset, chunkIndex := 0, 0; offset < len(body); chunkIndex++ {
			size := chunkSizes[chunkIndex%len(chunkSizes)]
			end := min(offset+size, len(body))
			if _, err := writer.Write(body[offset:end]); err != nil {
				t.Error(err)
				return
			}
			flusher.Flush()
			offset = end
		}
	}))
	defer server.Close()

	preflight, err := RunPreflight(
		context.Background(),
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"high",
		"Source",
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if preflight.IdentityContext != "Jane Doe: Host" {
		t.Fatalf("identity context = %q", preflight.IdentityContext)
	}
	if strings.Contains(preflight.IdentityContext, "SHOULD NOT APPEAR") {
		t.Fatal("thought text was included in identity context")
	}
}

func TestPreflightSSEPreservesSDKRetries(t *testing.T) {
	answer, err := json.Marshal(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{
				"role": "model",
				"parts": []any{map[string]any{
					"text": "PARTICIPANTS AND SPEAKERS:\nJane Doe: Host",
				}},
			},
			"groundingMetadata": map[string]any{
				"webSearchQueries": []string{"test query"},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if requests.Add(1) == 1 {
			writer.Header().Set("Content-Type", "application/json")
			writer.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(writer, `{"error":{"code":500,"message":"temporary","status":"INTERNAL"}}`)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write(append(append([]byte("\r\ndata: "), answer...), []byte("\r\n\r\n")...))
	}))
	defer server.Close()

	preflight, err := RunPreflight(
		context.Background(),
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"high",
		"Source",
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatalf("requests = %d", requests.Load())
	}
	if preflight.IdentityContext != "Jane Doe: Host" {
		t.Fatalf("identity context = %q", preflight.IdentityContext)
	}
}

func TestPreflightSSEPreservesHTTPError(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(writer, `{"error":{"code":400,"message":"invalid request","status":"INVALID_ARGUMENT"}}`)
	}))
	defer server.Close()

	_, err := RunPreflight(
		context.Background(),
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"high",
		"Source",
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "invalid request") {
		t.Fatalf("HTTP error = %v", err)
	}
	if requests.Load() != 1 {
		t.Fatalf("requests = %d", requests.Load())
	}
}

func TestPreflightSSEPreservesCancellation(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.WriteHeader(http.StatusOK)
		writer.(http.Flusher).Flush()
		close(started)
		<-request.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		_, err := RunPreflight(
			ctx,
			"test-key",
			server.URL+"/v1beta",
			"test-model",
			"high",
			"Source",
			nil,
		)
		result <- err
	}()
	<-started
	cancel()
	err := <-result
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
}

func TestTextRefinementAdapterChangesTextOnly(t *testing.T) {
	server, requests := newScriptedServer(t, []scriptedResponse{{
		Text: `{"changes":[{"id":0,"text":"JANE DOE: Corrected"}]}`,
	}})
	defer server.Close()
	dir := t.TempDir()
	input := filepath.Join(dir, "input.vtt")
	output := filepath.Join(dir, "output.vtt")
	data := "WEBVTT\n\nSTYLE\n::cue { color: lime; }\n\n00:00:00.000 --> 00:00:01.000 align:start\nJane Doe: Original\n\n"
	if err := os.WriteFile(input, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	preflight := &core.PreflightContext{
		ContractVersion: "preflight-v1",
		GroundedNames:   []string{"Jane Doe"},
	}
	err := Refine(
		context.Background(),
		input,
		output,
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"high",
		"Title",
		nil,
		nil,
		preflight,
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := vtt.Read(output)
	if err != nil {
		t.Fatal(err)
	}
	if result.Cues[0].Start != "00:00:00.000" ||
		result.Cues[0].End != "00:00:01.000" ||
		result.Cues[0].Settings != "align:start" ||
		result.Cues[0].Text != "Jane Doe: Corrected" {
		t.Fatalf("refined cue = %#v", result.Cues[0])
	}
	if len(*requests) != 1 || !strings.Contains((*requests)[0], `"responseSchema"`) {
		t.Fatalf("refinement request = %v", *requests)
	}
	published, err := os.ReadFile(output)
	if err != nil || !strings.Contains(string(published), "STYLE\n::cue { color: lime; }") {
		t.Fatalf("refinement lost subtitle styles: %q, %v", published, err)
	}
	if strings.Contains((*requests)[0], `"tools"`) || strings.Contains((*requests)[0], "functionDeclarations") {
		t.Fatalf("refinement request enables tools: %s", (*requests)[0])
	}
}

func TestAudioRefinementAdapterUsesSparseJSONContract(t *testing.T) {
	server, requests := newScriptedServer(t, []scriptedResponse{{
		Text: `{"contractVersion":"sparse-patch-v1","deletedSourceIds":[],"cues":[]}`,
	}})
	defer server.Close()
	dir := t.TempDir()
	stitched := filepath.Join(dir, "stitched.vtt")
	audio := filepath.Join(dir, "audio.ogg")
	output := filepath.Join(dir, "output.vtt")
	if err := os.WriteFile(stitched, []byte("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n안녕\n\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(audio, []byte("ogg"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := BoundaryAudioRefine(
		context.Background(),
		stitched,
		audio,
		dir,
		output,
		"test-key",
		server.URL+"/v1beta",
		"test-model",
		"Title",
		1,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := vtt.Read(output)
	if err != nil || len(result.Cues) != 1 || result.Cues[0].Text != "안녕" {
		t.Fatalf("audio output = %#v, %v", result, err)
	}
	request := (*requests)[0]
	for _, required := range []string{
		`"mimeType":"audio/ogg"`,
		`"responseJsonSchema"`,
		`"maxOutputTokens":65536`,
		`"thinkingLevel":"HIGH"`,
	} {
		if !strings.Contains(request, required) {
			t.Errorf("audio request lacks %s: %s", required, request)
		}
	}
}
