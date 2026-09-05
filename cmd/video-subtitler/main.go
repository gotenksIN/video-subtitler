package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gotenksIN/video-subtitler/internal/benchmark"
	"github.com/gotenksIN/video-subtitler/internal/core"
	"github.com/gotenksIN/video-subtitler/internal/gemini"
	"github.com/gotenksIN/video-subtitler/internal/pipeline"
	"github.com/joho/godotenv"
)

func usage() {
	fmt.Print(`usage: video-subtitler [OPTIONS] VIDEO_OR_VTT

Generate English WebVTT subtitles from video using the Gemini API.

Options:
  -o, --output PATH             Output VTT (default output_subtitles.vtt)
  --api-key KEY                 Gemini API key override
  --base-url URL                Gemini-compatible proxy base URL
  --model MODEL                 Chunk model
  --audio-refine-model MODEL    Boundary audio model
  --refine-model MODEL          Global text model
  --chunk-dur SECONDS           Chunk duration (default 60)
  --workers COUNT               API workers (default 7)
  --thinking-level LEVEL        minimal, low, medium, or high
  --context-url URL             Repeatable grounding URL
  --disable-audio-refine        Disable boundary audio repair
  --disable-text-refine         Disable global text refinement
  --refine-only                 Refine an input VTT
  -h, --help                    Show help

Use "video-subtitler benchmark --help" for benchmark options.
`)
}
func benchmarkUsage() {
	fmt.Print(`usage: video-subtitler benchmark [OPTIONS] VIDEO

Options:
  --case GEN:AUDIO:REFINE       Repeatable full-pipeline case
  --model MODEL                 Repeatable chunk-only model
  --context-url URL             Repeatable grounding URL
  --reference-vtt PATH          Reference subtitle file
  --output-dir PATH             Output directory
  --api-key, --base-url, --chunk-dur, --workers, --thinking-level
`)
}

type parsed struct {
	values      map[string][]string
	bools       map[string]bool
	positionals []string
}

var valueFlags = map[string]bool{
	"-o":                   true,
	"--output":             true,
	"--api-key":            true,
	"--base-url":           true,
	"--model":              true,
	"--refine-model":       true,
	"--audio-refine-model": true,
	"--chunk-dur":          true,
	"--workers":            true,
	"--thinking-level":     true,
	"--context-url":        true,
	"--case":               true,
	"--reference-vtt":      true,
	"--output-dir":         true,
}

func parse(args []string, benchmarkMode bool) (parsed, error) {
	result := parsed{map[string][]string{}, map[string]bool{}, nil}
	for i := 0; i < len(args); i++ {
		argument := args[i]
		if argument == "--" {
			result.positionals = append(result.positionals, args[i+1:]...)
			break
		}
		if argument == "-h" || argument == "--help" || argument == "--disable-audio-refine" || argument == "--disable-text-refine" || argument == "--refine-only" {
			result.bools[argument] = true
			continue
		}
		if strings.HasPrefix(argument, "-o") && argument != "-o" {
			value := strings.TrimPrefix(argument, "-o")
			value = strings.TrimPrefix(value, "=")
			result.values["--output"] = append(result.values["--output"], value)
			continue
		}
		if strings.HasPrefix(argument, "--") && strings.Contains(argument, "=") {
			parts := strings.SplitN(argument, "=", 2)
			if !valueFlags[parts[0]] {
				return result, fmt.Errorf("unrecognized arguments: %s", argument)
			}
			if err := validateOptionValue(parts[0], parts[1]); err != nil {
				return result, err
			}
			result.values[parts[0]] = append(result.values[parts[0]], parts[1])
			continue
		}
		if valueFlags[argument] {
			if i+1 >= len(args) {
				return result, fmt.Errorf("argument %s: expected one argument", argument)
			}
			value := args[i+1]
			if isOption(value) {
				return result, fmt.Errorf("argument %s: expected one argument", argument)
			}
			i++
			if err := validateOptionValue(argument, value); err != nil {
				return result, err
			}
			option := argument
			if option == "-o" {
				option = "--output"
			}
			result.values[option] = append(result.values[option], value)
			continue
		}
		if strings.HasPrefix(argument, "-") {
			return result, fmt.Errorf("unrecognized arguments: %s", argument)
		}
		result.positionals = append(result.positionals, argument)
	}
	if err := validateModeOptions(result, benchmarkMode); err != nil {
		return result, err
	}
	return result, nil
}

func isOption(value string) bool {
	if valueFlags[value] || value == "-h" {
		return true
	}
	for _, option := range []string{
		"--help",
		"--disable-audio-refine",
		"--disable-text-refine",
		"--refine-only",
	} {
		if value == option {
			return true
		}
	}
	return strings.HasPrefix(value, "--") && strings.Contains(value, "=")
}

func validateModeOptions(arguments parsed, benchmarkMode bool) error {
	if benchmarkMode {
		for _, option := range []string{"--output", "-o", "--refine-model", "--audio-refine-model"} {
			if len(arguments.values[option]) > 0 {
				return fmt.Errorf("unrecognized arguments: %s", option)
			}
		}
		for _, option := range []string{"--disable-audio-refine", "--disable-text-refine", "--refine-only"} {
			if arguments.bools[option] {
				return fmt.Errorf("unrecognized arguments: %s", option)
			}
		}
		return nil
	}
	for _, option := range []string{"--case", "--reference-vtt", "--output-dir"} {
		if len(arguments.values[option]) > 0 {
			return fmt.Errorf("unrecognized arguments: %s", option)
		}
	}
	return nil
}

func validateOptionValue(option, value string) error {
	switch option {
	case "--chunk-dur", "--workers":
		if _, err := strconv.Atoi(value); err != nil {
			return fmt.Errorf("argument %s: invalid int value: %q", option, value)
		}
	case "--case":
		models := strings.Split(value, ":")
		if len(models) != 3 || models[0] == "" || models[1] == "" || models[2] == "" {
			return fmt.Errorf("invalid case %q; expected GEN:AUDIO:REFINE", value)
		}
	case "--thinking-level":
		for _, level := range gemini.ThinkingLevels {
			if value == level {
				return nil
			}
		}
		return fmt.Errorf("argument --thinking-level: invalid choice: %q", value)
	}
	return nil
}
func last(arguments parsed, key, defaultValue string) string {
	values := arguments.values[key]
	if len(values) == 0 && key == "--output" {
		values = arguments.values["-o"]
	}
	if len(values) == 0 {
		return defaultValue
	}
	return values[len(values)-1]
}
func number(arguments parsed, key string, defaultValue int) (int, error) {
	value := last(arguments, key, "")
	if value == "" {
		return defaultValue, nil
	}
	number, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("argument %s: invalid int value: %q", key, value)
	}
	return number, nil
}
func loadEnv() {
	executable, err := os.Executable()
	if err != nil {
		return
	}
	root := filepath.Dir(filepath.Dir(executable))
	values, err := godotenv.Read(filepath.Join(root, ".env"))
	if err != nil {
		return
	}
	for name, value := range values {
		if _, ok := os.LookupEnv(name); !ok {
			_ = os.Setenv(name, value)
		}
	}
}
func main() {
	loadEnv()
	args := os.Args[1:]
	benchmarkMode := len(args) > 0 && args[0] == "benchmark"
	if benchmarkMode {
		args = args[1:]
	}
	arguments, err := parse(args, benchmarkMode)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	if arguments.bools["-h"] || arguments.bools["--help"] {
		if benchmarkMode {
			benchmarkUsage()
		} else {
			usage()
		}
		return
	}
	if len(arguments.positionals) != 1 {
		if benchmarkMode {
			benchmarkUsage()
		} else {
			usage()
		}
		os.Exit(2)
	}
	ctx := context.Background()
	if benchmarkMode {
		err = runBenchmark(ctx, arguments)
	} else {
		err = runMain(ctx, arguments)
	}
	if err != nil {
		if benchmarkMode {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		} else {
			fmt.Printf("Error: %v\n", err)
		}
		os.Exit(1)
	}
}
func env(name, defaultValue string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return defaultValue
}
func runMain(ctx context.Context, arguments parsed) error {
	apiKey := last(arguments, "--api-key", env("GEMINI_API_KEY", ""))
	baseURL := last(arguments, "--base-url", env("GEMINI_API_BASE", ""))
	model := last(arguments, "--model", env("GEMINI_MODEL", gemini.DefaultChunkModel))
	refineModel := last(arguments, "--refine-model", env("GEMINI_REFINE_MODEL", gemini.DefaultRefineModel))
	audioRefineModel := last(arguments, "--audio-refine-model", env("GEMINI_AUDIO_REFINE_MODEL", gemini.DefaultAudioRefineModel))
	chunkDuration, err := number(arguments, "--chunk-dur", 60)
	if err != nil {
		return err
	}
	workers, err := number(arguments, "--workers", 7)
	if err != nil {
		return err
	}
	input := arguments.positionals[0]
	output := last(arguments, "--output", "output_subtitles.vtt")
	if arguments.bools["--refine-only"] {
		contextURLs, err := core.ValidateContextURLs(arguments.values["--context-url"])
		if err != nil {
			return err
		}
		if _, err = os.Stat(input); err != nil {
			return fmt.Errorf("input VTT file not found: %s", input)
		}
		if apiKey == "" {
			return fmt.Errorf("Gemini API key not configured; set GEMINI_API_KEY in .env or the environment, or pass --api-key")
		}
		outputLock, err := pipeline.AcquireOutputLock(output)
		if err != nil {
			return err
		}
		defer outputLock.Close()
		return gemini.Refine(
			ctx,
			input,
			output,
			apiKey,
			baseURL,
			firstNonempty(refineModel, model),
			"high",
			core.DeriveSourceTitle(input),
			contextURLs,
			nil,
			nil,
		)
	}
	return pipeline.Run(ctx, pipeline.Config{
		VideoPath:        input,
		OutputPath:       output,
		Model:            model,
		APIKey:           apiKey,
		BaseURL:          baseURL,
		RefineModel:      refineModel,
		AudioRefineModel: audioRefineModel,
		ChunkDur:         chunkDuration,
		Workers:          workers,
		ThinkingLevel:    last(arguments, "--thinking-level", ""),
		AudioRefine:      !arguments.bools["--disable-audio-refine"],
		RefineText:       !arguments.bools["--disable-text-refine"],
		ContextURLs:      arguments.values["--context-url"],
	})
}

func firstNonempty(first, fallback string) string {
	if first != "" {
		return first
	}
	return fallback
}
func runBenchmark(ctx context.Context, arguments parsed) error {
	chunkDuration, err := number(arguments, "--chunk-dur", 60)
	if err != nil {
		return err
	}
	workers, err := number(arguments, "--workers", 7)
	if err != nil {
		return err
	}
	cases := []benchmark.Case{}
	for _, raw := range arguments.values["--case"] {
		models := strings.Split(raw, ":")
		if len(models) != 3 || models[0] == "" || models[1] == "" || models[2] == "" {
			return fmt.Errorf("invalid case %q; expected GEN:AUDIO:REFINE", raw)
		}
		cases = append(cases, benchmark.Case{
			Generation: models[0],
			Audio:      models[1],
			Refine:     models[2],
		})
	}
	return benchmark.Run(ctx, benchmark.Config{
		Video:       arguments.positionals[0],
		APIKey:      last(arguments, "--api-key", env("GEMINI_API_KEY", "")),
		BaseURL:     last(arguments, "--base-url", env("GEMINI_API_BASE", "")),
		Models:      arguments.values["--model"],
		Cases:       cases,
		ContextURLs: arguments.values["--context-url"],
		Reference:   last(arguments, "--reference-vtt", ""),
		OutputDir:   last(arguments, "--output-dir", ""),
		ChunkDur:    chunkDuration,
		Workers:     workers,
		Thinking:    last(arguments, "--thinking-level", ""),
	})
}
