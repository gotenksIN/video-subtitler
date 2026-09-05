package gemini

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/gotenksIN/video-subtitler/internal/core"
	"github.com/gotenksIN/video-subtitler/internal/media"
	"github.com/gotenksIN/video-subtitler/internal/storage"
	"github.com/gotenksIN/video-subtitler/internal/vtt"
	"google.golang.org/genai"
)

const (
	DefaultChunkModel       = "gemini-3.8-flash"
	DefaultRefineModel      = "gemini-3.1-pro-preview"
	DefaultAudioRefineModel = "gemini-3.8-flash"
	DefaultThinkingLevel    = "high"
	PreflightFilename       = "preflight_context.json"
)

var ThinkingLevels = []string{"minimal", "low", "medium", "high"}
var apiVersionSegment = regexp.MustCompile(`^v[0-9]+(alpha[0-9]*|beta[0-9]*)?$`)

type sseNormalizingTransport struct {
	transport http.RoundTripper
}

func (transport *sseNormalizingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := transport.transport.RoundTrip(request)
	if err != nil {
		return nil, err
	}
	mediaType, _, parseErr := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if parseErr != nil || mediaType != "text/event-stream" || response.StatusCode < 200 || response.StatusCode >= 300 {
		return response, nil
	}
	response.Body = newSSEBody(response.Body)
	response.ContentLength = -1
	response.Header.Del("Content-Length")
	return response, nil
}

type normalizedSSEBody struct {
	source      io.ReadCloser
	scanner     *bufio.Scanner
	pending     []byte
	finished    bool
	terminalErr error
}

func newSSEBody(source io.ReadCloser) *normalizedSSEBody {
	scanner := bufio.NewScanner(source)
	scanner.Buffer(make([]byte, 1024), 268435456)
	return &normalizedSSEBody{
		source:  source,
		scanner: scanner,
	}
}

func (body *normalizedSSEBody) Read(destination []byte) (int, error) {
	for len(body.pending) == 0 {
		if body.finished {
			if body.terminalErr != nil {
				err := body.terminalErr
				body.terminalErr = nil
				return 0, err
			}
			return 0, io.EOF
		}
		event, err := body.nextEvent()
		if len(event) > 0 {
			body.pending = append(event, '\n', '\n')
		}
		if err != nil {
			body.finished = true
			body.terminalErr = err
		}
	}
	read := copy(destination, body.pending)
	body.pending = body.pending[read:]
	return read, nil
}

func (body *normalizedSSEBody) nextEvent() ([]byte, error) {
	var event bytes.Buffer
	for body.scanner.Scan() {
		line := body.scanner.Bytes()
		if len(line) == 0 {
			if event.Len() == 0 {
				continue
			}
			return event.Bytes(), nil
		}
		if event.Len() > 0 {
			event.WriteByte('\n')
		}
		event.Write(line)
	}
	if err := body.scanner.Err(); err != nil {
		return nil, err
	}
	if event.Len() > 0 {
		return event.Bytes(), io.EOF
	}
	return nil, io.EOF
}

func (body *normalizedSSEBody) Close() error {
	return body.source.Close()
}

func ValidateThinking(model, level string) error {
	if level == "minimal" && !strings.Contains(strings.ToLower(model), "flash") {
		return fmt.Errorf("--thinking-level minimal is only supported by Flash models. Use low, medium, or high for this model")
	}
	for _, supportedLevel := range ThinkingLevels {
		if level == supportedLevel {
			return nil
		}
	}
	return fmt.Errorf("invalid thinking level %q", level)
}
func ptr[T any](value T) *T {
	return &value
}
func thinking(level string) *genai.ThinkingConfig {
	if level == "" {
		return nil
	}
	return &genai.ThinkingConfig{
		ThinkingLevel:   genai.ThinkingLevel(strings.ToUpper(level)),
		IncludeThoughts: true,
	}
}
func client(ctx context.Context, key, base string) (*genai.Client, error) {
	options := genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{}}
	if base != "" {
		options.BaseURL = base
		if parsedURL, err := url.Parse(base); err == nil {
			parts := strings.Split(strings.TrimRight(parsedURL.Path, "/"), "/")
			last := parts[len(parts)-1]
			if apiVersionSegment.MatchString(last) {
				options.APIVersion = last
				parsedURL.Path = strings.TrimSuffix(strings.TrimRight(parsedURL.Path, "/"), "/"+last)
				options.BaseURL = parsedURL.String()
			}
		}
	}
	return genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  key,
		Backend: genai.BackendGeminiAPI,
		HTTPClient: &http.Client{Transport: &sseNormalizingTransport{
			transport: http.DefaultTransport,
		}},
		HTTPOptions: options,
	})
}
func content(parts ...*genai.Part) *genai.Content {
	return genai.NewContentFromParts(parts, genai.RoleUser)
}
func schemaCaption() *genai.Schema {
	caption := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"id", "start", "end", "text"},
		Properties: map[string]*genai.Schema{
			"id":    {Type: genai.TypeInteger},
			"start": {Type: genai.TypeString},
			"end":   {Type: genai.TypeString},
			"text":  {Type: genai.TypeString},
		},
	}
	return &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"captions"},
		Properties: map[string]*genai.Schema{
			"captions": {Type: genai.TypeArray, Items: caption},
		},
	}
}
func schemaRefine() *genai.Schema {
	change := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"id", "text"},
		Properties: map[string]*genai.Schema{
			"id":   {Type: genai.TypeInteger},
			"text": {Type: genai.TypeString},
		},
	}
	return &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"changes"},
		Properties: map[string]*genai.Schema{
			"changes": {Type: genai.TypeArray, Items: change},
		},
	}
}
func configJSON(schema *genai.Schema, level string) *genai.GenerateContentConfig {
	return &genai.GenerateContentConfig{
		Temperature:      ptr(float32(0)),
		ResponseMIMEType: "application/json",
		ResponseSchema:   schema,
		ThinkingConfig:   thinking(level),
	}
}
func strictJSON(raw string, destination any) error {
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if decoder.More() {
		return fmt.Errorf("unexpected trailing JSON value")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return err
	}
	return nil
}

func decodeCaptions(raw []byte) ([]core.Caption, error) {
	if isJSONNull(raw) {
		return nil, fmt.Errorf("captions must not be null")
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, err
	}
	for _, item := range items {
		if err := requireFields(item, "id", "start", "end", "text"); err != nil {
			return nil, err
		}
	}
	var captions []core.Caption
	if err := strictJSON(string(raw), &captions); err != nil {
		return nil, err
	}
	return captions, nil
}

func decodeSubtitleResponse(raw string) (core.SubtitleResponse, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &fields); err != nil {
		return core.SubtitleResponse{}, err
	}
	if err := requireFields(fields, "captions"); err != nil {
		return core.SubtitleResponse{}, err
	}
	captions, err := decodeCaptions(fields["captions"])
	if err != nil {
		return core.SubtitleResponse{}, err
	}
	var response core.SubtitleResponse
	if err := strictJSON(raw, &response); err != nil {
		return core.SubtitleResponse{}, err
	}
	return core.SubtitleResponse{Captions: captions}, nil
}

func decodeRefinementResponse(raw string) (core.RefinementResponse, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &fields); err != nil {
		return core.RefinementResponse{}, err
	}
	if err := requireFields(fields, "changes"); err != nil {
		return core.RefinementResponse{}, err
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(fields["changes"], &items); err != nil {
		return core.RefinementResponse{}, err
	}
	for _, item := range items {
		if err := requireFields(item, "id", "text"); err != nil {
			return core.RefinementResponse{}, err
		}
	}
	var changes []core.RefinedCaption
	if err := strictJSON(string(fields["changes"]), &changes); err != nil {
		return core.RefinementResponse{}, err
	}
	var response core.RefinementResponse
	if err := strictJSON(raw, &response); err != nil {
		return core.RefinementResponse{}, err
	}
	return core.RefinementResponse{Changes: changes}, nil
}

func decodeAudioResponse(raw string) (core.AudioRefinementResponse, error) {
	var fields map[string]json.RawMessage
	if err := strictJSON(raw, &fields); err != nil {
		return core.AudioRefinementResponse{}, err
	}
	if err := requireFields(fields, "contractVersion", "cues"); err != nil {
		return core.AudioRefinementResponse{}, err
	}
	if deleted, exists := fields["deletedSourceIds"]; exists && isJSONNull(deleted) {
		return core.AudioRefinementResponse{}, fmt.Errorf("field %q must not be null", "deletedSourceIds")
	}
	var cueFields []map[string]json.RawMessage
	if err := json.Unmarshal(fields["cues"], &cueFields); err != nil {
		return core.AudioRefinementResponse{}, err
	}
	for _, cue := range cueFields {
		if err := requireFields(cue, "sourceIds", "start", "end", "text"); err != nil {
			return core.AudioRefinementResponse{}, err
		}
	}
	var response core.AudioRefinementResponse
	if err := strictJSON(raw, &response); err != nil {
		return response, err
	}
	if response.DeletedSourceIDs == nil {
		response.DeletedSourceIDs = []int{}
	}
	return response, nil
}

func requireFields(fields map[string]json.RawMessage, names ...string) error {
	for _, name := range names {
		value, exists := fields[name]
		if !exists || isJSONNull(value) {
			return fmt.Errorf("required field %q is missing", name)
		}
	}
	return nil
}

func isJSONNull(value json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(value), []byte("null"))
}
func stream(
	ctx context.Context,
	client *genai.Client,
	model string,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
) (string, []*genai.GenerateContentResponse, error) {
	var text strings.Builder
	var responses []*genai.GenerateContentResponse
	for response, err := range client.Models.GenerateContentStream(ctx, model, contents, config) {
		if err != nil {
			return "", responses, err
		}
		responses = append(responses, response)
		text.WriteString(response.Text())
	}
	return text.String(), responses, nil
}

func LoadCachedCaptions(path string, duration float64) ([]core.Caption, bool) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	captions, decodeErr := decodeCaptions(contents)
	if decodeErr != nil {
		_ = os.Remove(path)
		return nil, false
	}
	validated, err := core.ValidateCaptions(captions, duration)
	if err != nil {
		fmt.Printf("Ignoring invalid cached output %s: %v\n", path, err)
		_ = os.Remove(path)
		return nil, false
	}
	return validated, true
}
func ProcessChunk(
	ctx context.Context,
	key, base, dir string,
	chunk media.Chunk,
	model, mime, level, title string,
	names []string,
) bool {
	outputPath := filepath.Join(dir, fmt.Sprintf("subtitle_chunk_%03d.json", chunk.Idx))
	if _, ok := LoadCachedCaptions(outputPath, chunk.Duration); ok {
		fmt.Printf("Skipping %s - already processed.\n", chunk.Name)
		return true
	}
	data, err := os.ReadFile(filepath.Join(dir, chunk.Name))
	if err != nil {
		fmt.Printf("[Worker-%03d] ERROR processing %s: %v\n", chunk.Idx, chunk.Name, err)
		return false
	}
	if len(data) > 20*1024*1024 {
		fmt.Printf(
			"[Worker-%03d] Warning: %s is %.1f MB. Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail.\n",
			chunk.Idx,
			chunk.Name,
			float64(len(data))/1048576,
		)
	}
	for attempt := 0; attempt < 3; attempt++ {
		err = generateChunkAttempt(ctx, key, base, model, mime, level, title, names, data, chunk.Duration, outputPath)
		if err == nil {
			fmt.Printf("[Worker-%03d] Finished %s.\n", chunk.Idx, chunk.Name)
			return true
		}
		var validationError *responseValidationError
		if !errors.As(err, &validationError) {
			fmt.Printf("[Worker-%03d] ERROR processing %s: %v\n", chunk.Idx, chunk.Name, err)
			return false
		}
		if attempt < 2 {
			delay := time.Duration(1<<attempt) * time.Second
			fmt.Printf(
				"[Worker-%03d] Warning: Attempt %d failed for %s (%v); retrying in %s...\n",
				chunk.Idx,
				attempt+1,
				chunk.Name,
				err,
				delay,
			)
			select {
			case <-ctx.Done():
				return false
			case <-time.After(delay):
			}
		} else {
			fmt.Printf("[Worker-%03d] ERROR processing %s: %v\n", chunk.Idx, chunk.Name, err)
		}
	}
	return false
}

type responseValidationError struct {
	cause error
}

func (validationError *responseValidationError) Error() string {
	return validationError.cause.Error()
}

func (validationError *responseValidationError) Unwrap() error {
	return validationError.cause
}

func generateChunkAttempt(
	ctx context.Context,
	key, base, model, mime, level, title string,
	names []string,
	data []byte,
	duration float64,
	output string,
) error {
	client, err := client(ctx, key, base)
	if err != nil {
		return err
	}
	contents := []*genai.Content{content(
		genai.NewPartFromBytes(data, mime),
		genai.NewPartFromText(generationPrompt(duration, title, names)),
	)}
	raw, _, err := stream(ctx, client, model, contents, configJSON(schemaCaption(), level))
	if err != nil {
		return err
	}
	response, err := decodeSubtitleResponse(raw)
	if err != nil {
		return &responseValidationError{cause: err}
	}
	captions, err := core.ValidateCaptions(response.Captions, duration)
	if err != nil {
		return &responseValidationError{cause: err}
	}
	return storage.AtomicWriteJSON(output, captions)
}

func splitResearch(text string) (string, string) {
	var identityLines []string
	var terminologyLines []string
	target := &identityLines
	for _, line := range strings.Split(text, "\n") {
		heading := strings.TrimSuffix(strings.ToUpper(strings.TrimSpace(line)), ":")
		if heading == "PARTICIPANTS AND SPEAKERS" {
			target = &identityLines
		} else if heading == "TOPIC TERMINOLOGY AND PROPER NOUNS" {
			target = &terminologyLines
		} else {
			*target = append(*target, line)
		}
	}
	identity := strings.TrimSpace(strings.Join(identityLines, "\n"))
	terminology := strings.TrimSpace(strings.Join(terminologyLines, "\n"))
	return identity, terminology
}
func groundedNames(text string) []string {
	seen := map[string]bool{}
	out := []string{}
	non := map[string]bool{"evidence": true, "role": true, "aliases": true, "sources": true, "notes": true}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimLeft(line, " \t-*")
		colon := strings.Index(line, ":")
		if colon < 0 {
			continue
		}
		name := strings.TrimSpace(line[:colon])
		nameLength := len([]rune(name))
		if nameLength < 2 || nameLength > 31 {
			continue
		}
		if len(name) == 0 || name[0] < 'A' || name[0] > 'Z' {
			continue
		}
		key := core.CaseFold(name)
		if !non[key] && !seen[key] {
			seen[key] = true
			out = append(out, name)
		}
	}
	return out
}

type metadata struct {
	queries   []string
	sources   [][2]string
	retrieved map[string]genai.URLRetrievalStatus
}

func collectMetadata(responses []*genai.GenerateContentResponse) metadata {
	collected := metadata{retrieved: map[string]genai.URLRetrievalStatus{}}
	for _, response := range responses {
		for _, candidate := range response.Candidates {
			if candidate.GroundingMetadata != nil {
				grounding := candidate.GroundingMetadata
				collected.queries = append(collected.queries, grounding.WebSearchQueries...)
				for _, chunk := range grounding.GroundingChunks {
					if chunk.Web != nil && chunk.Web.URI != "" {
						collected.sources = append(
							collected.sources,
							[2]string{chunk.Web.Title, chunk.Web.URI},
						)
					}
				}
			}
			if candidate.URLContextMetadata != nil {
				for _, result := range candidate.URLContextMetadata.URLMetadata {
					collected.retrieved[result.RetrievedURL] = result.URLRetrievalStatus
				}
			}
		}
	}
	return collected
}
func urlIdentity(raw string) string {
	parsedURL, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	return strings.ToLower(parsedURL.Scheme) + "|" +
		strings.ToLower(parsedURL.Host) + "|" +
		strings.TrimRight(parsedURL.Path, "/") + "|" +
		parsedURL.RawQuery
}
func RunPreflight(
	ctx context.Context,
	key, base, model, level, title string,
	urls []string,
) (core.PreflightContext, error) {
	validURLs, err := core.ValidateContextURLs(urls)
	if err != nil {
		return core.PreflightContext{}, err
	}
	youTubeURLs, ordinaryURLs := core.ClassifyContextURLs(validURLs)
	geminiClient, err := client(ctx, key, base)
	if err != nil {
		return core.PreflightContext{}, err
	}
	tools := []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}}
	if len(ordinaryURLs) > 0 {
		tools = append(tools, &genai.Tool{URLContext: &genai.URLContext{}})
	}
	researchConfig := &genai.GenerateContentConfig{
		Temperature:    ptr(float32(0)),
		Tools:          tools,
		ThinkingConfig: thinking(level),
	}
	researchContents := []*genai.Content{
		content(genai.NewPartFromText(researchPrompt(title, ordinaryURLs, youTubeURLs))),
	}
	raw, responses, err := stream(ctx, geminiClient, model, researchContents, researchConfig)
	if err != nil {
		return core.PreflightContext{}, err
	}
	responseMetadata := collectMetadata(responses)
	if len(responseMetadata.queries) == 0 && len(responseMetadata.sources) == 0 {
		return core.PreflightContext{}, fmt.Errorf("the identity research response has no Google Search grounding; failing without publishing output")
	}
	retrievalStatuses := map[string]genai.URLRetrievalStatus{}
	for retrievedURL, status := range responseMetadata.retrieved {
		retrievalStatuses[urlIdentity(retrievedURL)] = status
	}
	for _, contextURL := range ordinaryURLs {
		status, ok := retrievalStatuses[urlIdentity(contextURL)]
		if !ok || status != genai.URLRetrievalStatusSuccess {
			return core.PreflightContext{}, fmt.Errorf("context URL %s was not retrieved successfully; failing without publishing output", contextURL)
		}
	}
	printGroundingMetadata(responseMetadata)
	identity, terms := splitResearch(raw)
	preflight := core.PreflightContext{
		ContractVersion:    "preflight-v1",
		IdentityContext:    identity,
		TerminologyContext: terms,
		GroundedNames:      groundedNames(identity),
	}
	if len(youTubeURLs) > 0 {
		parts := []*genai.Part{}
		for _, youTubeURL := range youTubeURLs {
			parts = append(parts, genai.NewPartFromURI(youTubeURL, "video/*"))
		}
		parts = append(parts, genai.NewPartFromText(youtubePrompt(title)))
		youtubeConfig := &genai.GenerateContentConfig{
			Temperature:    ptr(float32(0)),
			ThinkingConfig: thinking(level),
		}
		text, _, err := stream(
			ctx,
			geminiClient,
			model,
			[]*genai.Content{content(parts...)},
			youtubeConfig,
		)
		if err != nil {
			return preflight, err
		}
		if strings.TrimSpace(text) != "" {
			youTubeContext := strings.TrimSpace(text)
			preflight.YouTubeContext = &youTubeContext
		}
	}
	return preflight, nil
}

func printGroundingMetadata(value metadata) {
	if len(value.queries) > 0 {
		fmt.Println("Search queries:")
		for _, query := range SortedUnique(value.queries) {
			fmt.Printf("  - %s\n", query)
		}
	}
	if len(value.sources) > 0 {
		fmt.Println("Grounded sources:")
		seen := map[[2]string]bool{}
		for _, source := range value.sources {
			if seen[source] {
				continue
			}
			seen[source] = true
			title := source[0]
			if title == "" {
				title = "Untitled"
			}
			fmt.Printf("  - %s: %s\n", title, source[1])
		}
	}
}

func LoadPreflight(dir string) (core.PreflightContext, bool) {
	path := filepath.Join(dir, PreflightFilename)
	contents, err := os.ReadFile(path)
	if err != nil {
		return core.PreflightContext{}, false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(contents, &fields) != nil {
		_ = os.Remove(path)
		return core.PreflightContext{}, false
	}
	if err := requireFields(fields,
		"contract_version",
		"identity_context",
		"terminology_context",
		"grounded_names",
	); err != nil {
		_ = os.Remove(path)
		return core.PreflightContext{}, false
	}
	var preflight core.PreflightContext
	if strictJSON(string(contents), &preflight) != nil || preflight.ContractVersion != "preflight-v1" {
		_ = os.Remove(path)
		return core.PreflightContext{}, false
	}
	return preflight, true
}
func StorePreflight(dir string, preflight core.PreflightContext) error {
	return storage.AtomicWriteJSON(filepath.Join(dir, PreflightFilename), preflight)
}

func LoadScript(path string) ([]core.ScriptEntry, error) {
	inputFile, err := vtt.Read(path)
	if err != nil {
		return nil, err
	}
	entries := make([]core.ScriptEntry, len(inputFile.Cues))
	for index, cue := range inputFile.Cues {
		start, err := core.ParseTime(cue.Start)
		if err != nil {
			return nil, err
		}
		end, err := core.ParseTime(cue.End)
		if err != nil {
			return nil, err
		}
		entries[index] = core.ScriptEntry{
			ID:             index,
			Start:          core.MustFormatTime(start),
			End:            core.MustFormatTime(end),
			Text:           cue.Text,
			Classification: core.ClassifyCueText(cue.Text),
		}
	}
	return entries, nil
}
func Refine(
	ctx context.Context,
	input, output, key, base, model, level, title string,
	urls, names []string,
	preflight *core.PreflightContext,
) error {
	inputFile, err := vtt.Read(input)
	if err != nil {
		return err
	}
	var lines []string
	for index, cue := range inputFile.Cues {
		lines = append(lines, fmt.Sprintf("[%d] %s --> %s: %s", index, cue.Start, cue.End, cue.Text))
	}
	context := core.PreflightContext{}
	if preflight == nil {
		context, err = RunPreflight(ctx, key, base, model, level, title, urls)
		if err != nil {
			return err
		}
	} else {
		context = *preflight
	}
	geminiClient, err := client(ctx, key, base)
	if err != nil {
		return err
	}
	prompt := refinementPrompt(strings.Join(lines, "\n"), title, context)
	raw, _, err := stream(
		ctx,
		geminiClient,
		model,
		[]*genai.Content{content(genai.NewPartFromText(prompt))},
		configJSON(schemaRefine(), level),
	)
	if err != nil {
		return err
	}
	response, err := decodeRefinementResponse(raw)
	if err != nil {
		return fmt.Errorf("parsing or validating the model refinement response failed: %v\nRaw response:\n%s", err, raw)
	}
	if err = core.ValidateRefinementChanges(response.Changes, len(inputFile.Cues)); err != nil {
		return err
	}
	for _, change := range response.Changes {
		inputFile.Cues[change.ID].Text = change.Text
	}
	effectiveNames := append(append([]string{}, names...), context.GroundedNames...)
	inputFile.Cues = core.CanonicalizeSpeakerCasing(inputFile.Cues, effectiveNames)
	return inputFile.SaveAtomic(output)
}

func fileHash(path string) (string, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(contents)
	return hex.EncodeToString(hash[:]), nil
}
func audioIdentity(script, audio string, dur float64, bounds []float64, model string) (map[string]any, error) {
	scriptHash, err := fileHash(script)
	if err != nil {
		return nil, err
	}
	audioHash, err := fileHash(audio)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"script_sha256":     scriptHash,
		"audio_sha256":      audioHash,
		"audio_duration":    dur,
		"boundaries":        bounds,
		"model":             model,
		"thinking_level":    "high",
		"response_contract": "sparse-patch-v1",
		"mode":              "boundary",
	}, nil
}
func audioSchema() any {
	cue := map[string]any{
		"type":     "object",
		"required": []string{"sourceIds", "start", "end", "text"},
		"properties": map[string]any{
			"sourceIds": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "integer"},
			},
			"start": map[string]any{"type": "string"},
			"end":   map[string]any{"type": "string"},
			"text":  map[string]any{"type": "string"},
		},
	}
	return map[string]any{
		"type":     "object",
		"required": []string{"contractVersion", "cues"},
		"properties": map[string]any{
			"contractVersion": map[string]any{
				"type": "string",
				"enum": []string{"sparse-patch-v1"},
			},
			"deletedSourceIds": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "integer"},
			},
			"cues": map[string]any{
				"type":  "array",
				"items": cue,
			},
		},
	}
}
func BoundaryAudioRefine(
	ctx context.Context,
	stitched, audio, work, output, key, base, model, title string,
	duration float64,
	boundaries []float64,
) error {
	source, err := LoadScript(stitched)
	if err != nil {
		return err
	}
	identity, err := audioIdentity(stitched, audio, duration, boundaries, model)
	if err != nil {
		return err
	}
	responsePath := filepath.Join(work, "audio_refinement.json")
	metadataPath := filepath.Join(work, "audio_refinement.meta.json")

	response, cached := loadAudioCache(responsePath, metadataPath, identity)
	var validated []core.AudioRefinedCue
	if cached {
		validated, err = core.ValidateAudioRefinement(response, source, duration, boundaries)
		if err == nil {
			fmt.Println("Reusing cached boundary audio refinement response.")
		}
	}
	generated := !cached || err != nil
	if generated {
		discardAudioCache(responsePath, metadataPath)
		response, validated, err = requestAudioRefinement(
			ctx,
			key,
			base,
			model,
			title,
			audio,
			source,
			duration,
			boundaries,
		)
		if err != nil {
			return err
		}
	}

	candidate, err := publishAudioCandidate(work, validated)
	if err != nil {
		discardAudioCache(responsePath, metadataPath)
		return err
	}
	if generated {
		if err := storage.AtomicWriteJSON(responsePath, response); err != nil {
			return err
		}
		if err := storage.AtomicWriteJSON(metadataPath, map[string]any{"identity": identity}); err != nil {
			return err
		}
	}
	if output != candidate {
		file, err := vtt.Read(candidate)
		if err != nil {
			return err
		}
		if err := file.SaveAtomic(output); err != nil {
			return err
		}
	}
	fmt.Printf("Saved boundary audio-refined subtitles to %s\n", output)
	return nil
}

func loadAudioCache(responsePath, metadataPath string, identity map[string]any) (core.AudioRefinementResponse, bool) {
	var response core.AudioRefinementResponse
	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		return response, false
	}
	var metadata struct {
		Identity map[string]any `json:"identity"`
	}
	if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
		return response, false
	}
	storedIdentity, err := json.Marshal(metadata.Identity)
	if err != nil {
		return response, false
	}
	expectedIdentity, err := json.Marshal(identity)
	if err != nil || !bytes.Equal(storedIdentity, expectedIdentity) {
		return response, false
	}
	responseBytes, err := os.ReadFile(responsePath)
	if err != nil {
		return core.AudioRefinementResponse{}, false
	}
	response, err = decodeAudioResponse(string(responseBytes))
	if err != nil {
		return core.AudioRefinementResponse{}, false
	}
	return response, true
}

func discardAudioCache(paths ...string) {
	for _, path := range paths {
		_ = os.Remove(path)
	}
}

func requestAudioRefinement(
	ctx context.Context,
	key, base, model, title, audioPath string,
	source []core.ScriptEntry,
	duration float64,
	boundaries []float64,
) (core.AudioRefinementResponse, []core.AudioRefinedCue, error) {
	var response core.AudioRefinementResponse
	audioBytes, err := os.ReadFile(audioPath)
	if err != nil {
		return response, nil, err
	}
	client, err := client(ctx, key, base)
	if err != nil {
		return response, nil, err
	}
	contents := []*genai.Content{content(
		genai.NewPartFromBytes(audioBytes, media.AudioMIMEType),
		genai.NewPartFromText(audioPrompt(source, boundaries, duration, title)),
	)}
	config := &genai.GenerateContentConfig{
		Temperature:        ptr(float32(0)),
		ResponseMIMEType:   "application/json",
		ResponseJsonSchema: audioSchema(),
		ThinkingConfig:     thinking("high"),
		MaxOutputTokens:    65536,
	}
	raw, responses, err := stream(ctx, client, model, contents, config)
	if err != nil {
		return response, nil, err
	}
	for _, streamResponse := range responses {
		for _, candidate := range streamResponse.Candidates {
			if candidate.FinishReason == genai.FinishReasonMaxTokens {
				return response, nil, fmt.Errorf("audio refinement response exceeded the configured output budget (MAX_TOKENS)")
			}
		}
	}
	response, err = decodeAudioResponse(raw)
	if err != nil {
		return response, nil, fmt.Errorf(
			"parsing or validating the audio refinement response failed: %v\nRaw response:\n%s",
			err,
			raw,
		)
	}
	validated, err := core.ValidateAudioRefinement(response, source, duration, boundaries)
	if err != nil {
		return response, nil, fmt.Errorf(
			"boundary audio refinement response failed validation: %v\nRaw response:\n%s",
			err,
			raw,
		)
	}
	return response, validated, nil
}

func publishAudioCandidate(work string, validated []core.AudioRefinedCue) (string, error) {
	file := &vtt.File{}
	for _, cue := range validated {
		file.Cues = append(file.Cues, vtt.Cue{
			Start: cue.Start,
			End:   cue.End,
			Text:  cue.Text,
		})
	}
	candidate := filepath.Join(work, "audio_refined.vtt")
	temporary := filepath.Join(work, "audio_refined.vtt.tmp")
	_ = os.Remove(temporary)
	if err := file.SaveAtomic(temporary); err != nil {
		return "", err
	}
	defer os.Remove(temporary)
	serialized, err := vtt.Read(temporary)
	if err != nil {
		return "", err
	}
	if !equalCues(serialized.Cues, file.Cues) {
		return "", fmt.Errorf("serialized candidate does not match validated response")
	}
	if err := os.Rename(temporary, candidate); err != nil {
		return "", err
	}
	return candidate, nil
}

func equalCues(left, right []vtt.Cue) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Start != right[index].Start ||
			left[index].End != right[index].End ||
			left[index].Text != right[index].Text {
			return false
		}
	}
	return true
}

func SortedUnique(values []string) []string {
	seen := map[string]bool{}
	unique := []string{}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			unique = append(unique, value)
		}
	}
	return unique
}
