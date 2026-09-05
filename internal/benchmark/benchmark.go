package benchmark

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/gotenksIN/video-subtitler/internal/core"
	"github.com/gotenksIN/video-subtitler/internal/gemini"
	"github.com/gotenksIN/video-subtitler/internal/media"
	"github.com/gotenksIN/video-subtitler/internal/pipeline"
	"github.com/gotenksIN/video-subtitler/internal/storage"
	"github.com/gotenksIN/video-subtitler/internal/vtt"
)

type Case struct {
	Generation string
	Audio      string
	Refine     string
}

type Config struct {
	Video       string
	APIKey      string
	BaseURL     string
	Reference   string
	OutputDir   string
	Thinking    string
	Models      []string
	Cases       []Case
	ContextURLs []string
	ChunkDur    int
	Workers     int
}

func safeModel(model string) string {
	stem := regexp.MustCompile(`[^A-Za-z0-9_.-]+`).ReplaceAllString(model, "_")
	stem = strings.Trim(stem, ".")
	if stem == "" {
		stem = "model"
	}
	hash := sha256.Sum256([]byte(model))
	return stem + "-" + hex.EncodeToString(hash[:4])
}
func linkCopy(src, dst string) error {
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()
	info, err := source.Stat()
	if err != nil {
		return err
	}
	destination, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, info.Mode().Perm())
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(destination, source)
	closeErr := destination.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Chtimes(dst, info.ModTime(), info.ModTime())
}
func unique(values []string) []string {
	seen := map[string]bool{}
	result := []string{}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	return result
}

type runState struct {
	config            Config
	contextURLs       []string
	generationModels  []string
	workDir           string
	mimeType          string
	chunks            []media.Chunk
	boundaries        []float64
	audioPath         string
	audioDuration     float64
	sourceTitle       string
	generatedPaths    map[string]string
	generationSeconds map[string]float64
	audioPaths        map[string]string
	audioSeconds      map[string]float64
}

func Run(ctx context.Context, config Config) error {
	state, err := prepareRun(config)
	if err != nil {
		return err
	}
	workRoot := filepath.Join(state.config.OutputDir, "work")
	if err := os.MkdirAll(workRoot, 0o755); err != nil {
		return err
	}
	lock, err := pipeline.AcquireLock(workRoot)
	if err != nil {
		return err
	}
	defer lock.Close()

	if err := state.prepareMedia(); err != nil {
		return err
	}
	if err := state.runGenerationStage(ctx); err != nil {
		return err
	}
	if err := state.runAudioStage(ctx); err != nil {
		return err
	}
	return state.runRefinementStage(ctx)
}

func prepareRun(config Config) (*runState, error) {
	if config.OutputDir == "" {
		config.OutputDir = "benchmark_results"
	}
	if config.Thinking == "" {
		config.Thinking = "high"
	}
	if _, err := os.Stat(config.Video); err != nil {
		return nil, fmt.Errorf("source video not found: %s", config.Video)
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("Gemini API key not configured; set GEMINI_API_KEY or pass --api-key")
	}
	if config.ChunkDur <= 0 {
		return nil, fmt.Errorf("--chunk-dur must be greater than 0")
	}
	if config.Workers <= 0 {
		return nil, fmt.Errorf("--workers must be greater than 0")
	}
	if config.Reference != "" {
		if _, err := os.Stat(config.Reference); err != nil {
			return nil, fmt.Errorf("reference VTT not found: %s", config.Reference)
		}
	}
	contextURLs, err := core.ValidateContextURLs(config.ContextURLs)
	if err != nil {
		return nil, err
	}
	if len(config.Cases) == 0 && len(config.Models) == 0 {
		config.Cases = []Case{{
			Generation: gemini.DefaultChunkModel,
			Audio:      gemini.DefaultAudioRefineModel,
			Refine:     gemini.DefaultRefineModel,
		}}
	}
	generationModels := make([]string, 0, len(config.Cases)+len(config.Models))
	for _, benchmarkCase := range config.Cases {
		generationModels = append(generationModels, benchmarkCase.Generation)
	}
	generationModels = append(generationModels, config.Models...)
	generationModels = unique(generationModels)
	for _, model := range generationModels {
		if err := gemini.ValidateThinking(model, config.Thinking); err != nil {
			return nil, err
		}
	}
	return &runState{
		config:            config,
		contextURLs:       contextURLs,
		generationModels:  generationModels,
		generatedPaths:    map[string]string{},
		generationSeconds: map[string]float64{},
		audioPaths:        map[string]string{},
		audioSeconds:      map[string]float64{},
	}, nil
}

func (state *runState) prepareMedia() error {
	config := state.config
	extension, mimeType, codec, err := media.ProbeVideoFormat(config.Video)
	if err != nil {
		return err
	}
	fingerprint, err := storage.Fingerprint(config.Video)
	if err != nil {
		return err
	}
	manifest := map[string]any{
		"video":       fingerprint,
		"chunk_dur":   config.ChunkDur,
		"format":      "stream-copy-v1",
		"mode":        "benchmark",
		"chunk_ext":   extension,
		"chunk_mime":  mimeType,
		"video_codec": codec,
	}
	manifestHash, err := storage.HashJSON(manifest)
	if err != nil {
		return err
	}
	state.workDir = filepath.Join(config.OutputDir, "work", manifestHash[:16])
	if err := os.MkdirAll(state.workDir, 0o755); err != nil {
		return err
	}
	if err := media.SplitVideo(config.Video, state.workDir, config.ChunkDur, manifest); err != nil {
		return err
	}
	state.mimeType = mimeType
	state.chunks = media.ListChunks(state.workDir)
	if len(state.chunks) == 0 {
		return fmt.Errorf("no video chunks were created")
	}
	for _, chunk := range state.chunks[1:] {
		state.boundaries = append(state.boundaries, chunk.Start)
	}
	if len(config.Cases) > 0 {
		if !media.HasAudio(config.Video) {
			return fmt.Errorf("source video does not contain an audio stream for audio refinement")
		}
		state.audioPath, state.audioDuration, _, _, err = media.ExtractAudio(config.Video, state.workDir)
		if err != nil {
			return err
		}
	}
	state.sourceTitle = core.DeriveSourceTitle(config.Video)
	return nil
}

func (state *runState) runGenerationStage(ctx context.Context) error {
	for _, model := range state.generationModels {
		generationWork, err := state.prepareGenerationWork(model)
		if err != nil {
			return err
		}
		output := filepath.Join(state.config.OutputDir, safeModel(model)+".generated.vtt")
		pipelineConfig := pipeline.Config{
			APIKey:        state.config.APIKey,
			BaseURL:       state.config.BaseURL,
			Model:         model,
			Workers:       state.config.Workers,
			ThinkingLevel: state.config.Thinking,
		}
		started := time.Now()
		failed := pipeline.ProcessChunks(
			ctx,
			pipelineConfig,
			generationWork,
			state.chunks,
			state.mimeType,
			state.sourceTitle,
			nil,
		)
		if len(failed) > 0 {
			return fmt.Errorf("failed to process %d chunk(s): %s", len(failed), strings.Join(failed, ", "))
		}
		if err := pipeline.Stitch(generationWork, output); err != nil {
			return err
		}
		state.generatedPaths[model] = output
		state.generationSeconds[model] = time.Since(started).Seconds()
	}
	return nil
}

func (state *runState) prepareGenerationWork(model string) (string, error) {
	segments, err := os.ReadFile(filepath.Join(state.workDir, "segments.csv"))
	if err != nil {
		return "", err
	}
	chunkIdentity := make([]any, 0, len(state.chunks))
	for _, chunk := range state.chunks {
		info, err := os.Stat(filepath.Join(state.workDir, chunk.Name))
		if err != nil {
			return "", err
		}
		chunkIdentity = append(chunkIdentity, map[string]any{
			"name":     chunk.Name,
			"size":     info.Size(),
			"mtime_ns": info.ModTime().UnixNano(),
		})
	}
	identity := map[string]any{
		"model":          model,
		"thinking_level": state.config.Thinking,
		"split": map[string]any{
			"segments": string(segments),
			"chunks":   chunkIdentity,
		},
	}
	identityHash, err := storage.HashJSON(identity)
	if err != nil {
		return "", err
	}
	workDir := filepath.Join(state.workDir, "generation-"+identityHash[:16])
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return "", err
	}
	for _, chunk := range state.chunks {
		destination := filepath.Join(workDir, chunk.Name)
		if err := os.Remove(destination); err != nil && !os.IsNotExist(err) {
			return "", err
		}
		if err := linkCopy(filepath.Join(state.workDir, chunk.Name), destination); err != nil {
			return "", err
		}
	}
	segmentDestination := filepath.Join(workDir, "segments.csv")
	if err := os.Remove(segmentDestination); err != nil && !os.IsNotExist(err) {
		return "", err
	}
	if err := linkCopy(filepath.Join(state.workDir, "segments.csv"), segmentDestination); err != nil {
		return "", err
	}
	return workDir, nil
}

func audioPairKey(generationModel, audioModel string) string {
	return generationModel + "\x00" + audioModel
}

func (state *runState) runAudioStage(ctx context.Context) error {
	for _, benchmarkCase := range state.config.Cases {
		key := audioPairKey(benchmarkCase.Generation, benchmarkCase.Audio)
		if _, exists := state.audioPaths[key]; exists {
			continue
		}
		output := filepath.Join(
			state.config.OutputDir,
			safeModel(benchmarkCase.Generation)+"_audio_"+safeModel(benchmarkCase.Audio)+".vtt",
		)
		identity := map[string]any{
			"generation_model": benchmarkCase.Generation,
			"audio_model":      benchmarkCase.Audio,
		}
		identityHash, err := storage.HashJSON(identity)
		if err != nil {
			return err
		}
		audioWork := filepath.Join(state.workDir, "audio-"+identityHash[:16])
		if err := os.MkdirAll(audioWork, 0o755); err != nil {
			return err
		}
		started := time.Now()
		err = gemini.BoundaryAudioRefine(
			ctx,
			state.generatedPaths[benchmarkCase.Generation],
			state.audioPath,
			audioWork,
			output,
			state.config.APIKey,
			state.config.BaseURL,
			benchmarkCase.Audio,
			state.sourceTitle,
			state.audioDuration,
			state.boundaries,
		)
		if err != nil {
			return err
		}
		state.audioPaths[key] = output
		state.audioSeconds[key] = time.Since(started).Seconds()
	}
	return nil
}

func (state *runState) runRefinementStage(ctx context.Context) error {
	results := []map[string]any{}
	for _, benchmarkCase := range state.config.Cases {
		result, err := state.runCaseRefinement(ctx, benchmarkCase)
		if err != nil {
			return err
		}
		results = append(results, result)
		printJSON(result)
	}
	for _, model := range state.config.Models {
		result, err := state.chunkOnlyResult(model)
		if err != nil {
			return err
		}
		results = append(results, result)
		printJSON(result)
	}
	resultsPath := filepath.Join(state.config.OutputDir, "benchmark-results.json")
	if err := storage.AtomicWriteJSON(resultsPath, results); err != nil {
		return err
	}
	fmt.Printf("Saved benchmark results to %s\n", resultsPath)
	return nil
}

func (state *runState) runCaseRefinement(ctx context.Context, benchmarkCase Case) (map[string]any, error) {
	key := audioPairKey(benchmarkCase.Generation, benchmarkCase.Audio)
	output := filepath.Join(
		state.config.OutputDir,
		safeModel(benchmarkCase.Generation)+"_"+safeModel(benchmarkCase.Audio)+"_to_"+safeModel(benchmarkCase.Refine)+".final.vtt",
	)
	started := time.Now()
	err := gemini.Refine(
		ctx,
		state.audioPaths[key],
		output,
		state.config.APIKey,
		state.config.BaseURL,
		benchmarkCase.Refine,
		"high",
		state.sourceTitle,
		state.contextURLs,
		nil,
		nil,
	)
	if err != nil {
		return nil, err
	}
	refinementSeconds := time.Since(started).Seconds()
	result := map[string]any{
		"generation_model":     benchmarkCase.Generation,
		"audio_refine_model":   benchmarkCase.Audio,
		"refinement_model":     benchmarkCase.Refine,
		"generation_seconds":   state.generationSeconds[benchmarkCase.Generation],
		"audio_refine_seconds": state.audioSeconds[key],
		"refinement_seconds":   refinementSeconds,
		"total_seconds": state.generationSeconds[benchmarkCase.Generation] +
			state.audioSeconds[key] + refinementSeconds,
		"final_vtt": output,
	}
	if err := state.addComparison(result, output); err != nil {
		return nil, err
	}
	return result, nil
}

func (state *runState) chunkOnlyResult(model string) (map[string]any, error) {
	result := map[string]any{
		"generation_model":     model,
		"audio_refine_model":   nil,
		"refinement_model":     nil,
		"generation_seconds":   state.generationSeconds[model],
		"audio_refine_seconds": 0.0,
		"refinement_seconds":   0.0,
		"total_seconds":        state.generationSeconds[model],
		"generated_vtt":        state.generatedPaths[model],
	}
	if err := state.addComparison(result, state.generatedPaths[model]); err != nil {
		return nil, err
	}
	return result, nil
}

func (state *runState) addComparison(result map[string]any, output string) error {
	if state.config.Reference == "" {
		return nil
	}
	comparison, err := Compare(output, state.config.Reference)
	if err != nil {
		return err
	}
	result["comparison"] = comparison
	return nil
}

func printJSON(value any) {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return
	}
	fmt.Println(string(data))
}

var benchmarkSpeakerLabel = regexp.MustCompile(`^\s*[A-Z][\pL\pN_' -]{1,30}:\s*`)

var cueMarkup = regexp.MustCompile(`<[^>]*>`)

func normalize(text string) []string {
	text = cueMarkup.ReplaceAllString(text, "")
	text = benchmarkSpeakerLabel.ReplaceAllString(text, "")
	var normalized strings.Builder
	for _, character := range strings.ToLower(text) {
		if unicode.IsLetter(character) ||
			unicode.IsNumber(character) ||
			character == '_' ||
			unicode.IsSpace(character) {
			normalized.WriteRune(character)
		} else {
			normalized.WriteRune(' ')
		}
	}
	return strings.Fields(normalized.String())
}

type matchingBlock struct {
	leftStart  int
	rightStart int
	length     int
}

func sequenceRatio(left, right []string) float64 {
	rightIndexes := map[string][]int{}
	for index, word := range right {
		rightIndexes[word] = append(rightIndexes[word], index)
	}
	var blocks []matchingBlock
	queue := [][4]int{{0, len(left), 0, len(right)}}
	for len(queue) > 0 {
		rangeToSearch := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		leftLow, leftHigh := rangeToSearch[0], rangeToSearch[1]
		rightLow, rightHigh := rangeToSearch[2], rangeToSearch[3]
		best := findLongestMatch(left, rightIndexes, leftLow, leftHigh, rightLow, rightHigh)
		if best.length > 0 {
			blocks = append(blocks, best)
			if leftLow < best.leftStart && rightLow < best.rightStart {
				queue = append(queue, [4]int{leftLow, best.leftStart, rightLow, best.rightStart})
			}
			if best.leftStart+best.length < leftHigh && best.rightStart+best.length < rightHigh {
				queue = append(queue, [4]int{
					best.leftStart + best.length,
					leftHigh,
					best.rightStart + best.length,
					rightHigh,
				})
			}
		}
	}
	matches := 0
	for _, block := range blocks {
		matches += block.length
	}
	if len(left)+len(right) == 0 {
		return 1
	}
	return 2 * float64(matches) / float64(len(left)+len(right))
}

func findLongestMatch(
	left []string,
	rightIndexes map[string][]int,
	leftLow, leftHigh, rightLow, rightHigh int,
) matchingBlock {
	best := matchingBlock{leftStart: leftLow, rightStart: rightLow}
	previousLengths := map[int]int{}
	for leftIndex := leftLow; leftIndex < leftHigh; leftIndex++ {
		currentLengths := map[int]int{}
		for _, rightIndex := range rightIndexes[left[leftIndex]] {
			if rightIndex < rightLow {
				continue
			}
			if rightIndex >= rightHigh {
				break
			}
			length := previousLengths[rightIndex-1] + 1
			currentLengths[rightIndex] = length
			if length > best.length {
				best = matchingBlock{
					leftStart:  leftIndex - length + 1,
					rightStart: rightIndex - length + 1,
					length:     length,
				}
			}
		}
		previousLengths = currentLengths
	}
	return best
}
func Compare(outputPath, referencePath string) (map[string]any, error) {
	generated, err := vtt.Read(outputPath)
	if err != nil {
		return nil, err
	}
	reference, err := vtt.Read(referencePath)
	if err != nil {
		return nil, err
	}
	var generatedWords, referenceWords []string
	var generatedIntervals, referenceIntervals []core.Interval
	for _, cue := range generated.Cues {
		generatedWords = append(generatedWords, normalize(cue.Text)...)
		start, err := core.ParseTime(cue.Start)
		if err != nil {
			return nil, err
		}
		end, err := core.ParseTime(cue.End)
		if err != nil {
			return nil, err
		}
		generatedIntervals = append(generatedIntervals, core.Interval{Start: start, End: end})
	}
	for _, cue := range reference.Cues {
		referenceWords = append(referenceWords, normalize(cue.Text)...)
		start, err := core.ParseTime(cue.Start)
		if err != nil {
			return nil, err
		}
		end, err := core.ParseTime(cue.End)
		if err != nil {
			return nil, err
		}
		referenceIntervals = append(referenceIntervals, core.Interval{Start: start, End: end})
	}
	generatedIntervals = core.MergeIntervals(generatedIntervals)
	referenceIntervals = core.MergeIntervals(referenceIntervals)
	generatedSeconds := sum(generatedIntervals)
	referenceSeconds := sum(referenceIntervals)
	overlapSeconds := intersection(generatedIntervals, referenceIntervals)
	unionSeconds := referenceSeconds + generatedSeconds - overlapSeconds
	recall, precision, intersectionOverUnion := 0.0, 0.0, 0.0
	if referenceSeconds > 0 {
		recall = overlapSeconds / referenceSeconds
	}
	if generatedSeconds > 0 {
		precision = overlapSeconds / generatedSeconds
	}
	if unionSeconds > 0 {
		intersectionOverUnion = overlapSeconds / unionSeconds
	}
	return map[string]any{
		"reference_cues":           len(reference.Cues),
		"generated_cues":           len(generated.Cues),
		"text_similarity":          sequenceRatio(referenceWords, generatedWords),
		"reference_active_seconds": referenceSeconds,
		"generated_active_seconds": generatedSeconds,
		"temporal_overlap_seconds": overlapSeconds,
		"temporal_recall":          recall,
		"temporal_precision":       precision,
		"temporal_iou":             intersectionOverUnion,
	}, nil
}
func sum(intervals []core.Interval) float64 {
	total := 0.0
	for _, interval := range intervals {
		total += interval.End - interval.Start
	}
	return total
}
func intersection(left, right []core.Interval) float64 {
	total := 0.0
	leftIndex, rightIndex := 0, 0
	for leftIndex < len(left) && rightIndex < len(right) {
		start := left[leftIndex].Start
		if right[rightIndex].Start > start {
			start = right[rightIndex].Start
		}
		end := left[leftIndex].End
		if right[rightIndex].End < end {
			end = right[rightIndex].End
		}
		if end > start {
			total += end - start
		}
		if left[leftIndex].End < right[rightIndex].End {
			leftIndex++
		} else {
			rightIndex++
		}
	}
	return total
}
