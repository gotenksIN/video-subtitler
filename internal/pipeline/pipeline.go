package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/gotenksIN/video-subtitler/internal/core"
	"github.com/gotenksIN/video-subtitler/internal/gemini"
	"github.com/gotenksIN/video-subtitler/internal/media"
	"github.com/gotenksIN/video-subtitler/internal/storage"
	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

const ChunkRoot = "temp_video_chunks"
const DefaultWorkers = 7

type Config struct {
	VideoPath        string
	OutputPath       string
	Model            string
	APIKey           string
	BaseURL          string
	RefineModel      string
	AudioRefineModel string
	ThinkingLevel    string
	ChunkDur         int
	Workers          int
	AudioRefine      bool
	RefineText       bool
	ContextURLs      []string
}

func (config Config) ChunkThinking() string {
	if config.ThinkingLevel == "" {
		return gemini.DefaultThinkingLevel
	}
	return config.ThinkingLevel
}
func BuildManifest(config Config) (map[string]any, string, error) {
	extension, mimeType, codec, err := media.ProbeVideoFormat(config.VideoPath)
	if err != nil {
		return nil, "", err
	}
	fingerprint, err := storage.Fingerprint(config.VideoPath)
	if err != nil {
		return nil, "", err
	}
	manifest := map[string]any{
		"video":                fingerprint,
		"chunk_dur":            config.ChunkDur,
		"format":               "stream-copy-v1",
		"mode":                 "generate",
		"model":                config.Model,
		"chunk_thinking_level": config.ChunkThinking(),
		"chunk_ext":            extension,
		"chunk_mime":           mimeType,
		"video_codec":          codec,
	}
	hash, err := storage.HashJSON(manifest)
	if err != nil {
		return nil, "", err
	}
	return manifest, filepath.Join(ChunkRoot, hash[:16]), nil
}

type Lock struct {
	file *os.File
}

func AcquireLock(dir string) (*Lock, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return acquireFileLock(filepath.Join(dir, ".lock"), dir)
}

func AcquireOutputLock(outputPath string) (*Lock, error) {
	absolutePath, err := filepath.Abs(outputPath)
	if err != nil {
		return nil, err
	}
	lockPath := filepath.Join(
		filepath.Dir(absolutePath),
		"."+filepath.Base(absolutePath)+".video-subtitler.lock",
	)
	return acquireFileLock(lockPath, absolutePath)
}

func acquireFileLock(path, resource string) (*Lock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		contents, readErr := os.ReadFile(path)
		closeErr := file.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		pid := strings.TrimSpace(string(contents))
		detail := ""
		if _, parseErr := strconv.Atoi(pid); parseErr == nil {
			detail = " (PID " + pid + ")"
		}
		return nil, fmt.Errorf("another run%s is already using %s", detail, resource)
	}
	if err := file.Truncate(0); err != nil {
		_ = file.Close()
		return nil, err
	}
	if _, err := file.Seek(0, 0); err != nil {
		_ = file.Close()
		return nil, err
	}
	if _, err := fmt.Fprint(file, os.Getpid()); err != nil {
		_ = file.Close()
		return nil, err
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return nil, err
	}
	return &Lock{file: file}, nil
}
func (lock *Lock) Close() {
	if lock != nil && lock.file != nil {
		_ = syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN)
		_ = lock.file.Close()
	}
}
func CleanCompleted(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.Name() == ".lock" {
			continue
		}
		if err = os.RemoveAll(filepath.Join(dir, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}
func Validate(config Config) ([]string, error) {
	contextURLs, err := core.ValidateContextURLs(config.ContextURLs)
	if err != nil {
		return nil, err
	}
	if config.ChunkDur <= 0 {
		return nil, fmt.Errorf("--chunk-dur must be greater than 0")
	}
	if config.Workers <= 0 {
		return nil, fmt.Errorf("--workers must be greater than 0")
	}
	if err = gemini.ValidateThinking(config.Model, config.ChunkThinking()); err != nil {
		return nil, err
	}
	if _, err = os.Stat(config.VideoPath); err != nil {
		return nil, fmt.Errorf("video file not found: %s", config.VideoPath)
	}
	videoPath, err := filepath.Abs(config.VideoPath)
	if err != nil {
		return nil, err
	}
	outputPath, err := filepath.Abs(config.OutputPath)
	if err != nil {
		return nil, err
	}
	if resolved, resolveErr := filepath.EvalSymlinks(videoPath); resolveErr == nil {
		videoPath = resolved
	}
	if resolved, resolveErr := filepath.EvalSymlinks(outputPath); resolveErr == nil {
		outputPath = resolved
	}
	if videoPath == outputPath {
		return nil, fmt.Errorf("--output must not resolve to the source video")
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("Gemini API key not configured; set GEMINI_API_KEY in .env or the environment, or pass --api-key")
	}
	return contextURLs, nil
}
func ProcessChunks(ctx context.Context, config Config, dir string, chunks []media.Chunk, mime, title string, names []string) []string {
	fmt.Printf("Processing %d chunks using %d workers...\n", len(chunks), config.Workers)
	jobs := make(chan media.Chunk)
	failed := make(chan string, len(chunks))
	var workers sync.WaitGroup
	for range config.Workers {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for chunk := range jobs {
				succeeded := gemini.ProcessChunk(
					ctx,
					config.APIKey,
					config.BaseURL,
					dir,
					chunk,
					config.Model,
					mime,
					config.ChunkThinking(),
					title,
					names,
				)
				if !succeeded {
					failed <- chunk.Name
				}
			}
		}()
	}
	for _, chunk := range chunks {
		jobs <- chunk
	}
	close(jobs)
	workers.Wait()
	close(failed)
	var failedNames []string
	for name := range failed {
		failedNames = append(failedNames, name)
	}
	sort.Strings(failedNames)
	return failedNames
}

type stitchedEntry struct {
	start      float64
	end        float64
	text       string
	chunkIndex int
}

func Stitch(dir, output string) error {
	fmt.Println("Stitching chunks into final VTT...")
	chunks := media.ListChunks(dir)
	chunksByIndex := map[int]media.Chunk{}
	var boundaryStarts []float64
	for index, chunk := range chunks {
		chunksByIndex[chunk.Idx] = chunk
		if index > 0 {
			boundaryStarts = append(boundaryStarts, chunk.Start)
		}
	}
	resultNames, err := subtitleResultNames(dir)
	if err != nil {
		return err
	}
	if len(resultNames) != len(chunks) {
		return fmt.Errorf("invalid subtitle results: expected %d chunks, found %d", len(chunks), len(resultNames))
	}
	var entries []stitchedEntry
	for _, name := range resultNames {
		chunkIndex, err := subtitleResultIndex(name)
		if err != nil {
			return err
		}
		chunk, ok := chunksByIndex[chunkIndex]
		if !ok {
			return fmt.Errorf("invalid subtitle results: unexpected chunk index %d", chunkIndex)
		}
		captions, err := readCaptionResult(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		for _, caption := range captions {
			start, err := core.ParseTime(caption.Start)
			if err != nil {
				return fmt.Errorf("invalid caption timing in %s: %w", name, err)
			}
			end, err := core.ParseTime(caption.End)
			if err != nil || end <= start {
				return fmt.Errorf("invalid caption timing in %s", name)
			}
			entries = append(entries, stitchedEntry{
				start:      chunk.Start + start,
				end:        chunk.Start + end,
				text:       caption.Text,
				chunkIndex: chunkIndex,
			})
		}
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].start < entries[j].start })
	entries = mergeVisualBoundaryEntries(entries, boundaryStarts)
	outputFile := &vtt.File{}
	for _, entry := range entries {
		outputFile.Cues = append(outputFile.Cues, vtt.Cue{
			Start: core.MustFormatTime(entry.start),
			End:   core.MustFormatTime(entry.end),
			Text:  entry.text,
		})
	}
	if err := outputFile.SaveAtomic(output); err != nil {
		return err
	}
	fmt.Printf("Successfully saved to %s with %d total captions.\n", output, len(outputFile.Cues))
	return nil
}

func subtitleResultNames(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, "subtitle_chunk_") && strings.HasSuffix(name, ".json") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names, nil
}

func subtitleResultIndex(name string) (int, error) {
	value := strings.TrimSuffix(strings.TrimPrefix(name, "subtitle_chunk_"), ".json")
	return strconv.Atoi(value)
}

func readCaptionResult(path string) ([]core.Caption, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var captions []core.Caption
	if err := json.Unmarshal(data, &captions); err != nil {
		return nil, fmt.Errorf("invalid subtitle result %s: %w", filepath.Base(path), err)
	}
	return captions, nil
}

func mergeVisualBoundaryEntries(entries []stitchedEntry, boundaries []float64) []stitchedEntry {
	for {
		merged := false
		for currentIndex, current := range entries {
			if current.chunkIndex >= len(boundaries) {
				continue
			}
			boundary := boundaries[current.chunkIndex]
			for followingIndex := currentIndex + 1; followingIndex < len(entries); followingIndex++ {
				following := entries[followingIndex]
				if !canMergeVisualEntries(current, following, boundary) {
					continue
				}
				entries[currentIndex] = stitchedEntry{
					start:      current.start,
					end:        following.end,
					text:       strings.TrimSpace(current.text),
					chunkIndex: following.chunkIndex,
				}
				entries = append(entries[:followingIndex], entries[followingIndex+1:]...)
				merged = true
				break
			}
			if merged {
				break
			}
		}
		if !merged {
			return entries
		}
	}
}

func canMergeVisualEntries(current, following stitchedEntry, boundary float64) bool {
	return following.chunkIndex == current.chunkIndex+1 &&
		core.ClassifyCueText(current.text) == "editorial" &&
		core.ClassifyCueText(following.text) == "editorial" &&
		strings.TrimSpace(current.text) == strings.TrimSpace(following.text) &&
		abs(boundary-current.end) <= 0.5 &&
		abs(following.start-boundary) <= 0.5
}
func abs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
func Run(ctx context.Context, config Config) error {
	contextURLs, err := Validate(config)
	if err != nil {
		return err
	}
	title := core.DeriveSourceTitle(config.VideoPath)
	manifest, workDir, err := BuildManifest(config)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(workDir, 0o755); err != nil {
		return err
	}
	workLock, err := AcquireLock(workDir)
	if err != nil {
		return err
	}
	defer workLock.Close()
	outputLock, err := AcquireOutputLock(config.OutputPath)
	if err != nil {
		return err
	}
	defer outputLock.Close()
	audioPath, audioDuration, err := prepareAudio(config, workDir)
	if err != nil {
		return err
	}
	preflight, err := preparePreflight(ctx, config, workDir, title, contextURLs)
	if err != nil {
		return err
	}
	if err = media.SplitVideo(config.VideoPath, workDir, config.ChunkDur, manifest); err != nil {
		return err
	}
	chunks := media.ListChunks(workDir)
	if len(chunks) == 0 {
		return fmt.Errorf("no video chunks were created")
	}
	failed := ProcessChunks(
		ctx,
		config,
		workDir,
		chunks,
		manifest["chunk_mime"].(string),
		title,
		preflight.GroundedNames,
	)
	if len(failed) > 0 {
		return fmt.Errorf(
			"failed to process %d chunk(s): %s; keeping %s so you can retry",
			len(failed),
			strings.Join(failed, ", "),
			workDir,
		)
	}
	stitchedPath := filepath.Join(workDir, "stitched.vtt")
	if err = Stitch(workDir, stitchedPath); err != nil {
		return err
	}
	currentPath := stitchedPath
	if config.AudioRefine {
		currentPath = filepath.Join(workDir, "audio_refined.vtt")
		boundaries := []float64{}
		for _, chunk := range chunks[1:] {
			boundaries = append(boundaries, chunk.Start)
		}
		if err = gemini.BoundaryAudioRefine(
			ctx,
			stitchedPath,
			audioPath,
			workDir,
			currentPath,
			config.APIKey,
			config.BaseURL,
			first(config.AudioRefineModel, gemini.DefaultAudioRefineModel),
			title,
			audioDuration,
			boundaries,
		); err != nil {
			return err
		}
	}
	if err = publishResult(ctx, config, currentPath, title, contextURLs, preflight); err != nil {
		return err
	}
	fmt.Printf("Cleaning up temporary directory: %s\n", workDir)
	return CleanCompleted(workDir)
}

func prepareAudio(config Config, workDir string) (string, float64, error) {
	if !config.AudioRefine {
		return "", 0, nil
	}
	if !media.HasAudio(config.VideoPath) {
		return "", 0, fmt.Errorf("failed to extract complete audio; the source may not contain an audio stream")
	}
	audioPath, duration, _, _, err := media.ExtractAudio(config.VideoPath, workDir)
	return audioPath, duration, err
}

func preparePreflight(
	ctx context.Context,
	config Config,
	workDir, title string,
	contextURLs []string,
) (core.PreflightContext, error) {
	if cached, ok := gemini.LoadPreflight(workDir); ok {
		return cached, nil
	}
	preflight, err := gemini.RunPreflight(
		ctx,
		config.APIKey,
		config.BaseURL,
		first(config.RefineModel, config.Model),
		"high",
		title,
		contextURLs,
	)
	if err != nil {
		return core.PreflightContext{}, err
	}
	if err := gemini.StorePreflight(workDir, preflight); err != nil {
		return core.PreflightContext{}, err
	}
	return preflight, nil
}

func publishResult(
	ctx context.Context,
	config Config,
	currentPath, title string,
	contextURLs []string,
	preflight core.PreflightContext,
) error {
	if !config.RefineText {
		outputFile, err := vtt.Read(currentPath)
		if err != nil {
			return err
		}
		outputFile.Cues = core.CanonicalizeSpeakerCasing(outputFile.Cues, nil)
		return outputFile.SaveAtomic(config.OutputPath)
	}
	outputDir := filepath.Dir(config.OutputPath)
	temporaryFile, err := os.CreateTemp(
		outputDir,
		"."+filepath.Base(config.OutputPath)+".*.staging.vtt",
	)
	if err != nil {
		return err
	}
	stagingPath := temporaryFile.Name()
	if err := temporaryFile.Close(); err != nil {
		_ = os.Remove(stagingPath)
		return err
	}
	defer os.Remove(stagingPath)
	if err := gemini.Refine(
		ctx,
		currentPath,
		stagingPath,
		config.APIKey,
		config.BaseURL,
		first(config.RefineModel, config.Model),
		"high",
		title,
		contextURLs,
		nil,
		&preflight,
	); err != nil {
		return err
	}
	return os.Rename(stagingPath, config.OutputPath)
}
func first(preferred, fallback string) string {
	if preferred != "" {
		return preferred
	}
	return fallback
}
