package core

import (
	"fmt"
	"math"
	"net/url"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/gotenksIN/video-subtitler/internal/vtt"
	"golang.org/x/text/cases"
)

const (
	AudioRefineResponseContract = "sparse-patch-v1"
	RepairWindowSeconds         = 10.0
)

type Caption struct {
	ID    int    `json:"id"`
	Start string `json:"start"`
	End   string `json:"end"`
	Text  string `json:"text"`
}
type SubtitleResponse struct {
	Captions []Caption `json:"captions"`
}
type RefinedCaption struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}
type RefinementResponse struct {
	Changes []RefinedCaption `json:"changes"`
}
type AudioRefinedCue struct {
	SourceIDs []int  `json:"sourceIds"`
	Start     string `json:"start"`
	End       string `json:"end"`
	Text      string `json:"text"`
}
type AudioRefinementResponse struct {
	ContractVersion  string            `json:"contractVersion"`
	DeletedSourceIDs []int             `json:"deletedSourceIds"`
	Cues             []AudioRefinedCue `json:"cues"`
}
type PreflightContext struct {
	ContractVersion    string   `json:"contract_version"`
	IdentityContext    string   `json:"identity_context"`
	TerminologyContext string   `json:"terminology_context"`
	YouTubeContext     *string  `json:"youtube_context"`
	GroundedNames      []string `json:"grounded_names"`
}
type ScriptEntry struct {
	ID             int    `json:"id"`
	Start          string `json:"start"`
	End            string `json:"end"`
	Text           string `json:"text"`
	Classification string `json:"classification"`
}

var languageTag = regexp.MustCompile(`^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$`)
var speakerLabel = regexp.MustCompile(`^([ \t]*)([A-Z][\pL\pN_' -]{1,30})(:[ \t]*)`)

func DeriveSourceTitle(path string) string {
	name := filepath.Base(path)
	for _, suffix := range []string{".vtt", ".srt", ".sub", ".sbv"} {
		if len(name) > len(suffix) && strings.HasSuffix(strings.ToLower(name), suffix) {
			name = name[:len(name)-len(suffix)]
			if dot := strings.LastIndexByte(name, '.'); dot >= 0 && languageTag.MatchString(name[dot+1:]) {
				name = name[:dot]
			}
			break
		}
	}
	for _, suffix := range []string{".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"} {
		if len(name) > len(suffix) && strings.HasSuffix(strings.ToLower(name), suffix) {
			name = name[:len(name)-len(suffix)]
			break
		}
	}
	return strings.TrimSpace(name)
}

func ParseTime(raw string) (float64, error) {
	normalized := strings.ReplaceAll(strings.TrimSpace(raw), ",", ".")
	if strings.HasPrefix(normalized, "-") {
		return 0, fmt.Errorf("negative timestamp: %s", raw)
	}
	parts := strings.Split(normalized, ":")
	if len(parts) < 1 || len(parts) > 3 {
		return 0, fmt.Errorf("invalid timestamp: %s", raw)
	}
	seconds, err := parseSeconds(parts[len(parts)-1])
	if err != nil {
		return 0, err
	}
	hours := 0
	minutes := 0
	if len(parts) == 2 {
		minutes, err = strconv.Atoi(parts[0])
	} else if len(parts) == 3 {
		hours, err = strconv.Atoi(parts[0])
		if err == nil {
			minutes, err = strconv.Atoi(parts[1])
		}
	}
	if err != nil {
		return 0, err
	}
	return float64(hours*3600+minutes*60) + seconds, nil
}

func parseSeconds(value string) (float64, error) {
	if strings.Count(value, ".") > 1 || value == "" {
		return 0, fmt.Errorf("invalid seconds %q", value)
	}
	parts := strings.SplitN(value, ".", 2)
	seconds, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, err
	}
	if len(parts) == 1 || parts[1] == "" {
		return float64(seconds), nil
	}
	fraction, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, err
	}
	divisor := math.Pow10(len(parts[1]))
	return float64(seconds) + float64(fraction)/divisor, nil
}

func FormatTime(seconds float64) (string, error) {
	if seconds < 0 {
		return "", fmt.Errorf("negative timestamp: %v", seconds)
	}
	milliseconds := int64(math.RoundToEven(seconds * 1000))
	hours := milliseconds / 3600000
	milliseconds %= 3600000
	minutes := milliseconds / 60000
	milliseconds %= 60000
	wholeSeconds := milliseconds / 1000
	milliseconds %= 1000
	return fmt.Sprintf(
		"%02d:%02d:%02d.%03d",
		hours,
		minutes,
		wholeSeconds,
		milliseconds,
	), nil
}

func MustFormatTime(value float64) string {
	formatted, err := FormatTime(value)
	if err != nil {
		panic(err)
	}
	return formatted
}

func ValidateCaptions(captions []Caption, duration float64) ([]Caption, error) {
	seen := map[int]bool{}
	duplicates := map[int]bool{}
	for _, caption := range captions {
		if seen[caption.ID] {
			duplicates[caption.ID] = true
		}
		seen[caption.ID] = true
	}
	if len(duplicates) > 0 {
		var ids []int
		for id := range duplicates {
			ids = append(ids, id)
		}
		sort.Ints(ids)
		return nil, fmt.Errorf("duplicate caption IDs: %v", ids)
	}
	validated := make([]Caption, 0, len(captions))
	for _, caption := range captions {
		start, err := ParseTime(caption.Start)
		if err != nil {
			return nil, err
		}
		end, err := ParseTime(caption.End)
		if err != nil {
			return nil, err
		}
		if start < 0 || end <= start {
			return nil, fmt.Errorf("invalid caption timing for id=%d: %s --> %s", caption.ID, caption.Start, caption.End)
		}
		if end > duration {
			if end-duration > .5 || duration <= start {
				return nil, fmt.Errorf("caption timing exceeds chunk duration for id=%d: %s --> %s", caption.ID, caption.Start, caption.End)
			}
			end = duration
		}
		canonicalStart := MustFormatTime(start)
		canonicalEnd := MustFormatTime(end)
		roundedStart, _ := ParseTime(canonicalStart)
		roundedEnd, _ := ParseTime(canonicalEnd)
		if roundedEnd <= roundedStart {
			return nil, fmt.Errorf("caption timing rounds to a non-positive interval for id=%d: %s --> %s", caption.ID, caption.Start, caption.End)
		}
		validated = append(validated, Caption{
			ID:    caption.ID,
			Start: canonicalStart,
			End:   canonicalEnd,
			Text:  caption.Text,
		})
	}
	sort.SliceStable(validated, func(leftIndex, rightIndex int) bool {
		leftStart, _ := ParseTime(validated[leftIndex].Start)
		rightStart, _ := ParseTime(validated[rightIndex].Start)
		if leftStart == rightStart {
			return validated[leftIndex].ID < validated[rightIndex].ID
		}
		return leftStart < rightStart
	})
	return validated, nil
}

func ValidateRefinementChanges(changes []RefinedCaption, count int) error {
	seen := map[int]bool{}
	for _, change := range changes {
		if change.ID < 0 || change.ID >= count {
			return fmt.Errorf("subtitle ID %d is out of range", change.ID)
		}
		if seen[change.ID] {
			return fmt.Errorf("subtitle ID %d is duplicated", change.ID)
		}
		if strings.TrimSpace(change.Text) == "" {
			return fmt.Errorf("subtitle ID %d has empty text", change.ID)
		}
		seen[change.ID] = true
	}
	return nil
}

func ValidateContextURLs(rawURLs []string) ([]string, error) {
	validated := []string{}
	seen := map[string]bool{}
	for _, rawURL := range rawURLs {
		value := strings.TrimSpace(rawURL)
		if strings.IndexFunc(value, unicode.IsSpace) >= 0 {
			return nil, fmt.Errorf("invalid --context-url %q: URL must not contain whitespace", value)
		}
		parsedURL, err := url.Parse(value)
		if err != nil || parsedURL.Hostname() == "" ||
			(strings.ToLower(parsedURL.Scheme) != "http" && strings.ToLower(parsedURL.Scheme) != "https") {
			return nil, fmt.Errorf("invalid --context-url %q: expected an absolute HTTP or HTTPS URL with a host", value)
		}
		if !seen[value] {
			validated = append(validated, value)
			seen[value] = true
		}
	}
	return validated, nil
}
func IsYouTubeURL(value string) bool {
	parsedURL, err := url.Parse(value)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsedURL.Hostname())
	if host == "youtu.be" {
		return len(strings.FieldsFunc(parsedURL.Path, func(character rune) bool {
			return character == '/'
		})) == 1
	}
	isYouTubeHost := host == "youtube.com" || host == "www.youtube.com" || host == "m.youtube.com"
	return isYouTubeHost && parsedURL.Path == "/watch" && parsedURL.Query().Get("v") != ""
}
func ClassifyContextURLs(values []string) (youTube, ordinary []string) {
	for _, value := range values {
		if IsYouTubeURL(value) {
			youTube = append(youTube, value)
		} else {
			ordinary = append(ordinary, value)
		}
	}
	return youTube, ordinary
}

func bracketParts(text string) (string, []string, bool) {
	var outside strings.Builder
	var fragments []string
	depth, start := 0, -1
	unmatched := false
	characters := []rune(text)
	for index, character := range characters {
		switch character {
		case '[':
			if depth == 0 {
				start = index
			}
			depth++
		case ']':
			if depth == 0 {
				unmatched = true
				outside.WriteRune(character)
			} else {
				depth--
				if depth == 0 {
					fragments = append(fragments, string(characters[start:index+1]))
					start = -1
				}
			}
		default:
			if depth == 0 {
				outside.WriteRune(character)
			}
		}
	}
	if depth > 0 {
		unmatched = true
		outside.WriteString(string(characters[start:]))
	}
	return outside.String(), fragments, unmatched
}
func ClassifyCueText(text string) string {
	outside, fragments, _ := bracketParts(text)
	if len(fragments) == 0 {
		return "dialogue"
	}
	if strings.IndexFunc(outside, func(character rune) bool {
		return character == '_' || unicode.IsLetter(character) || unicode.IsDigit(character)
	}) >= 0 {
		return "mixed"
	}
	return "editorial"
}
func VisualFragments(text string) []string {
	_, fragments, _ := bracketParts(text)
	return fragments
}

func HasUnmatchedBrackets(text string) bool {
	_, _, unmatched := bracketParts(text)
	return unmatched
}

func CaseFold(value string) string {
	return cases.Fold().String(value)
}
func CanonicalizeSpeakerCasing(cues []vtt.Cue, grounded []string) []vtt.Cue {
	groundedByFold := map[string]string{}
	for _, name := range grounded {
		if strings.TrimSpace(name) != "" {
			groundedByFold[CaseFold(name)] = name
		}
	}
	type labelCounts struct {
		order  []string
		counts map[string]int
	}
	groups := map[string]*labelCounts{}
	for _, cue := range cues {
		if ClassifyCueText(cue.Text) == "editorial" {
			continue
		}
		for _, line := range strings.Split(cue.Text, "\n") {
			match := speakerLabel.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			foldedLabel := CaseFold(match[2])
			if groups[foldedLabel] == nil {
				groups[foldedLabel] = &labelCounts{counts: map[string]int{}}
			}
			if groups[foldedLabel].counts[match[2]] == 0 {
				groups[foldedLabel].order = append(groups[foldedLabel].order, match[2])
			}
			groups[foldedLabel].counts[match[2]]++
		}
	}
	targets := map[string]string{}
	for foldedLabel, group := range groups {
		if groundedName := groundedByFold[foldedLabel]; groundedName != "" {
			targets[foldedLabel] = groundedName
			continue
		}
		for _, spelling := range group.order {
			if targets[foldedLabel] == "" || group.counts[spelling] > group.counts[targets[foldedLabel]] {
				targets[foldedLabel] = spelling
			}
		}
	}
	for cueIndex, cue := range cues {
		if ClassifyCueText(cue.Text) == "editorial" {
			continue
		}
		lines := strings.Split(cue.Text, "\n")
		for lineIndex, line := range lines {
			match := speakerLabel.FindStringSubmatchIndex(line)
			if match == nil {
				continue
			}
			label := line[match[4]:match[5]]
			if target := targets[CaseFold(label)]; target != "" {
				lines[lineIndex] = line[:match[4]] + target + line[match[5]:]
			}
		}
		cues[cueIndex].Text = strings.Join(lines, "\n")
	}
	return cues
}

type Interval struct {
	Start float64
	End   float64
}

func MergeIntervals(intervals []Interval) []Interval {
	sort.Slice(intervals, func(leftIndex, rightIndex int) bool {
		return intervals[leftIndex].Start < intervals[rightIndex].Start
	})
	merged := []Interval{}
	for _, interval := range intervals {
		if len(merged) > 0 && interval.Start <= merged[len(merged)-1].End {
			if interval.End > merged[len(merged)-1].End {
				merged[len(merged)-1].End = interval.End
			}
		} else {
			merged = append(merged, interval)
		}
	}
	return merged
}
func BuildRepairRegions(boundaries []float64, window float64) []Interval {
	regions := make([]Interval, len(boundaries))
	for index, boundary := range boundaries {
		regions[index] = Interval{boundary - window, boundary + window}
	}
	return MergeIntervals(regions)
}
func Intersects(interval Interval, regions []Interval) bool {
	for _, region := range regions {
		if interval.Start < region.End && interval.End > region.Start {
			return true
		}
	}
	return false
}
