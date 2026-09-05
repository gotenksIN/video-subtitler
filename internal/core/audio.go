package core

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

func canonicalTimestamp(value string) (string, error) {
	seconds, err := ParseTime(value)
	if err != nil {
		return "", fmt.Errorf("invalid timestamp %q: %v", value, err)
	}
	formatted := MustFormatTime(seconds)
	if strings.TrimSpace(value) != formatted {
		return "", fmt.Errorf("timestamp %q is not canonical HH:MM:SS.mmm format", value)
	}
	return formatted, nil
}

func containsDialogue(text string) bool {
	for _, line := range strings.Split(text, "\n") {
		outside, _, _ := bracketParts(line)
		if colon := strings.Index(outside, ":"); colon >= 0 {
			outside = outside[colon+1:]
		}
		if strings.IndexFunc(outside, unicode.IsLetter) >= 0 ||
			strings.IndexFunc(outside, unicode.IsDigit) >= 0 {
			return true
		}
	}
	return false
}

func fragmentCounts(texts []string) map[string]int {
	counts := map[string]int{}
	for _, text := range texts {
		for _, fragment := range VisualFragments(text) {
			counts[fragment]++
		}
	}
	return counts
}

func equalFragmentCounts(left, right map[string]int) bool {
	if len(left) != len(right) {
		return false
	}
	for fragment, count := range left {
		if right[fragment] != count {
			return false
		}
	}
	return true
}

func FilterAudioPatch(
	response AudioRefinementResponse,
	source []ScriptEntry,
	boundaries []float64,
) (AudioRefinementResponse, error) {
	regions := BuildRepairRegions(boundaries, RepairWindowSeconds)
	intervals := map[int]Interval{}
	for _, entry := range source {
		start, _ := ParseTime(entry.Start)
		end, _ := ParseTime(entry.End)
		intervals[entry.ID] = Interval{Start: start, End: end}
	}

	keptCues := []AudioRefinedCue{}
	for _, cue := range response.Cues {
		keep := true
		if len(cue.SourceIDs) == 0 {
			start, err := ParseTime(cue.Start)
			if err != nil {
				return response, err
			}
			end, err := ParseTime(cue.End)
			if err != nil {
				return response, err
			}
			keep = Intersects(Interval{Start: start, End: end}, regions)
		} else {
			hasUnknownSource := false
			for _, sourceID := range cue.SourceIDs {
				interval, known := intervals[sourceID]
				if !known {
					hasUnknownSource = true
				} else if !Intersects(interval, regions) {
					keep = false
				}
			}
			if hasUnknownSource {
				keep = true
			}
		}
		if keep {
			keptCues = append(keptCues, cue)
		}
	}

	keptDeleted := []int{}
	for _, sourceID := range response.DeletedSourceIDs {
		interval, known := intervals[sourceID]
		if !known || Intersects(interval, regions) {
			keptDeleted = append(keptDeleted, sourceID)
		}
	}
	response.Cues = keptCues
	response.DeletedSourceIDs = keptDeleted
	return response, nil
}

func ValidateAudioRefinement(
	response AudioRefinementResponse,
	source []ScriptEntry,
	duration float64,
	boundaries []float64,
) ([]AudioRefinedCue, error) {
	if response.ContractVersion != AudioRefineResponseContract {
		return nil, fmt.Errorf("invalid audio refinement contract %q", response.ContractVersion)
	}
	sourceByID, err := validateAudioSource(source)
	if err != nil {
		return nil, err
	}
	response, err = FilterAudioPatch(response, source, boundaries)
	if err != nil {
		return nil, err
	}
	if err := validateSparsePatchOrder(response.Cues); err != nil {
		return nil, err
	}
	response, deleted, err := reconstructAudioCandidate(response, source)
	if err != nil {
		return nil, err
	}
	output, references, descendants, err := normalizeAudioCues(response.Cues, duration)
	if err != nil {
		return nil, err
	}
	if err := validateAudioAccounting(source, sourceByID, references, descendants, deleted); err != nil {
		return nil, err
	}
	if err := validateAudioLineage(output, sourceByID, references); err != nil {
		return nil, err
	}
	if err := validateVisualIntegrity(output, source); err != nil {
		return nil, err
	}
	regions := BuildRepairRegions(boundaries, RepairWindowSeconds)
	if err := validateAudioAuthority(output, source, sourceByID, descendants, regions); err != nil {
		return nil, err
	}
	return output, nil
}

func validateAudioSource(source []ScriptEntry) (map[int]ScriptEntry, error) {
	sourceByID := map[int]ScriptEntry{}
	for _, entry := range source {
		if HasUnmatchedBrackets(entry.Text) {
			return nil, fmt.Errorf("audio refinement source cue contains unmatched brackets")
		}
		if _, exists := sourceByID[entry.ID]; exists {
			return nil, fmt.Errorf("audio refinement source IDs are not unique")
		}
		sourceByID[entry.ID] = entry
	}
	return sourceByID, nil
}

func validateSparsePatchOrder(cues []AudioRefinedCue) error {
	lastStart := -1.0
	lastSourceID := -1
	for _, cue := range cues {
		canonicalStart, err := canonicalTimestamp(cue.Start)
		if err != nil {
			return err
		}
		start, _ := ParseTime(canonicalStart)
		if start < lastStart {
			return fmt.Errorf("sparse audio refinement cues are not in script order")
		}
		lastStart = start
		for _, sourceID := range cue.SourceIDs {
			if sourceID < lastSourceID {
				return fmt.Errorf("sparse audio refinement cues reference source IDs out of script order")
			}
			lastSourceID = sourceID
		}
		if HasUnmatchedBrackets(cue.Text) {
			return fmt.Errorf("sparse audio refinement cue contains unmatched brackets")
		}
	}
	return nil
}

func reconstructAudioCandidate(
	response AudioRefinementResponse,
	source []ScriptEntry,
) (AudioRefinementResponse, map[int]bool, error) {
	referenced := map[int]bool{}
	for _, cue := range response.Cues {
		for _, sourceID := range cue.SourceIDs {
			referenced[sourceID] = true
		}
	}
	deleted := map[int]bool{}
	for _, sourceID := range response.DeletedSourceIDs {
		if deleted[sourceID] {
			return response, nil, fmt.Errorf("audio refinement deleted_source_ids contains duplicates")
		}
		deleted[sourceID] = true
	}
	for _, entry := range source {
		if referenced[entry.ID] || deleted[entry.ID] {
			continue
		}
		response.Cues = append(response.Cues, AudioRefinedCue{
			SourceIDs: []int{entry.ID},
			Start:     entry.Start,
			End:       entry.End,
			Text:      entry.Text,
		})
	}
	sort.SliceStable(response.Cues, func(left, right int) bool {
		leftStart, _ := ParseTime(response.Cues[left].Start)
		rightStart, _ := ParseTime(response.Cues[right].Start)
		if leftStart != rightStart {
			return leftStart < rightStart
		}
		return firstSourceID(response.Cues[left]) < firstSourceID(response.Cues[right])
	})
	return response, deleted, nil
}

func firstSourceID(cue AudioRefinedCue) int {
	if len(cue.SourceIDs) == 0 {
		return -1
	}
	return cue.SourceIDs[0]
}

func normalizeAudioCues(
	cues []AudioRefinedCue,
	duration float64,
) ([]AudioRefinedCue, map[int]int, map[int][]AudioRefinedCue, error) {
	durationMillis := int64(math.RoundToEven(duration * 1000))
	references := map[int]int{}
	descendants := map[int][]AudioRefinedCue{}
	output := make([]AudioRefinedCue, 0, len(cues))
	for _, cue := range cues {
		normalized, err := normalizeAudioCue(cue, durationMillis, references)
		if err != nil {
			return nil, nil, nil, err
		}
		output = append(output, normalized)
		for _, sourceID := range normalized.SourceIDs {
			descendants[sourceID] = append(descendants[sourceID], normalized)
		}
	}
	return output, references, descendants, nil
}

func normalizeAudioCue(
	cue AudioRefinedCue,
	durationMillis int64,
	references map[int]int,
) (AudioRefinedCue, error) {
	if strings.TrimSpace(cue.Text) == "" {
		return cue, fmt.Errorf("audio refinement output cues must contain text")
	}
	if len(cue.SourceIDs) == 0 {
		if len(VisualFragments(cue.Text)) > 0 {
			return cue, fmt.Errorf("recovered cues must not contain bracketed text")
		}
		if !containsDialogue(cue.Text) {
			return cue, fmt.Errorf("recovered cues must contain spoken dialogue")
		}
	}
	for index, sourceID := range cue.SourceIDs {
		if index > 0 && sourceID <= cue.SourceIDs[index-1] {
			return cue, fmt.Errorf(
				"audio refinement cue source_ids must be strictly increasing with no duplicates: %v",
				cue.SourceIDs,
			)
		}
		references[sourceID]++
	}
	start, err := canonicalTimestamp(cue.Start)
	if err != nil {
		return cue, err
	}
	end, err := canonicalTimestamp(cue.End)
	if err != nil {
		return cue, err
	}
	startSeconds, _ := ParseTime(start)
	endSeconds, _ := ParseTime(end)
	startMillis := int64(math.RoundToEven(startSeconds * 1000))
	endMillis := int64(math.RoundToEven(endSeconds * 1000))
	if startMillis < 0 {
		return cue, fmt.Errorf("audio refinement cue has a negative start: %s", start)
	}
	if endMillis > durationMillis {
		if endMillis-durationMillis > 500 {
			return cue, fmt.Errorf(
				"audio refinement cue %s --> %s exceeds the complete audio duration",
				start,
				end,
			)
		}
		endMillis = durationMillis
		end = MustFormatTime(float64(endMillis) / 1000)
	}
	if endMillis <= startMillis {
		return cue, fmt.Errorf(
			"audio refinement cue %s --> %s rounds to a non-positive interval",
			start,
			end,
		)
	}
	lines := []string{}
	for _, line := range strings.Split(strings.TrimSpace(cue.Text), "\n") {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, strings.TrimRightFunc(line, unicode.IsSpace))
		}
	}
	cue.Start = MustFormatTime(float64(startMillis) / 1000)
	cue.End = end
	cue.Text = strings.Join(lines, "\n")
	return cue, nil
}

func validateAudioAccounting(
	source []ScriptEntry,
	sourceByID map[int]ScriptEntry,
	references map[int]int,
	descendants map[int][]AudioRefinedCue,
	deleted map[int]bool,
) error {
	for sourceID := range references {
		if _, known := sourceByID[sourceID]; !known {
			return fmt.Errorf("audio refinement references unknown source IDs: [%d]", sourceID)
		}
		if deleted[sourceID] {
			return fmt.Errorf("audio refinement both references and deletes source IDs: [%d]", sourceID)
		}
	}
	for sourceID := range deleted {
		entry, known := sourceByID[sourceID]
		if !known {
			return fmt.Errorf("audio refinement deletes unknown source IDs: [%d]", sourceID)
		}
		if entry.Classification != "dialogue" {
			return fmt.Errorf("audio refinement deletes non-dialogue cue %d", sourceID)
		}
	}
	for _, entry := range source {
		if references[entry.ID] == 0 && !deleted[entry.ID] {
			return fmt.Errorf("audio refinement does not account for source IDs: [%d]", entry.ID)
		}
		if entry.Classification == "editorial" && !hasIdenticalSingleton(descendants[entry.ID], entry) {
			return fmt.Errorf(
				"pure editorial cue %d must be preserved with identical text and timing",
				entry.ID,
			)
		}
	}
	return nil
}

func hasIdenticalSingleton(cues []AudioRefinedCue, source ScriptEntry) bool {
	return len(cues) == 1 &&
		len(cues[0].SourceIDs) == 1 &&
		cues[0].SourceIDs[0] == source.ID &&
		cues[0].Text == source.Text &&
		cues[0].Start == source.Start &&
		cues[0].End == source.End
}

func validateAudioLineage(
	output []AudioRefinedCue,
	sourceByID map[int]ScriptEntry,
	references map[int]int,
) error {
	lastCueBySource := map[int]int{}
	for cueIndex, cue := range output {
		for _, sourceID := range cue.SourceIDs {
			if previous, exists := lastCueBySource[sourceID]; exists && previous != cueIndex-1 {
				return fmt.Errorf("source ID %d repeats in non-adjacent output cues", sourceID)
			}
			lastCueBySource[sourceID] = cueIndex
			if references[sourceID] > 1 && len(cue.SourceIDs) != 1 {
				return fmt.Errorf(
					"source ID %d must split into singleton cues; merges cannot overlap splits",
					sourceID,
				)
			}
		}
		if err := validateMergeContinuity(cue.SourceIDs, sourceByID); err != nil {
			return err
		}
	}
	return nil
}

func validateMergeContinuity(sourceIDs []int, sourceByID map[int]ScriptEntry) error {
	if len(sourceIDs) < 2 {
		return nil
	}
	included := map[int]bool{}
	for _, sourceID := range sourceIDs {
		included[sourceID] = true
	}
	for sourceID := sourceIDs[0] + 1; sourceID < sourceIDs[len(sourceIDs)-1]; sourceID++ {
		if !included[sourceID] && sourceByID[sourceID].Classification != "editorial" {
			return fmt.Errorf(
				"output cue %v merges source IDs that are not contiguous in script order",
				sourceIDs,
			)
		}
	}
	return nil
}

func validateVisualIntegrity(output []AudioRefinedCue, source []ScriptEntry) error {
	sourceByID := map[int]ScriptEntry{}
	var sourceVisualTexts []string
	for _, entry := range source {
		sourceByID[entry.ID] = entry
		if entry.Classification != "dialogue" {
			sourceVisualTexts = append(sourceVisualTexts, entry.Text)
		}
	}
	var outputTexts []string
	for _, cue := range output {
		outputTexts = append(outputTexts, cue.Text)
		allowed := map[string]bool{}
		for _, sourceID := range cue.SourceIDs {
			for _, fragment := range VisualFragments(sourceByID[sourceID].Text) {
				allowed[fragment] = true
			}
		}
		for _, fragment := range VisualFragments(cue.Text) {
			if len(cue.SourceIDs) > 0 && !allowed[fragment] {
				return fmt.Errorf(
					"cue %v contains bracketed fragments that do not belong to it",
					cue.SourceIDs,
				)
			}
		}
	}
	if !equalFragmentCounts(fragmentCounts(sourceVisualTexts), fragmentCounts(outputTexts)) {
		return fmt.Errorf("the complete output must preserve the exact bracketed fragment multiset")
	}
	for _, entry := range source {
		if entry.Classification != "mixed" {
			continue
		}
		var descendantTexts []string
		for _, cue := range output {
			for _, sourceID := range cue.SourceIDs {
				if sourceID == entry.ID {
					descendantTexts = append(descendantTexts, cue.Text)
					break
				}
			}
		}
		expected := fragmentCounts([]string{entry.Text})
		consumed := fragmentCounts(descendantTexts)
		for fragment, count := range expected {
			if consumed[fragment] < count {
				return fmt.Errorf(
					"bracketed fragments of mixed cue %d must be preserved exactly once",
					entry.ID,
				)
			}
		}
	}
	return nil
}

func validateAudioAuthority(
	output []AudioRefinedCue,
	source []ScriptEntry,
	sourceByID map[int]ScriptEntry,
	descendants map[int][]AudioRefinedCue,
	regions []Interval,
) error {
	for _, entry := range source {
		start, _ := ParseTime(entry.Start)
		end, _ := ParseTime(entry.End)
		if !Intersects(Interval{Start: start, End: end}, regions) &&
			!hasIdenticalSingleton(descendants[entry.ID], entry) {
			return fmt.Errorf(
				"cue %d lies outside every repair region and must remain identical",
				entry.ID,
			)
		}
	}
	seen := map[string]bool{}
	for _, cue := range output {
		key := cue.Start + "\x00" + cue.End + "\x00" + cue.Text
		if seen[key] {
			return fmt.Errorf("audio refinement produced duplicate output cues")
		}
		seen[key] = true
		if isIdenticalCue(cue, sourceByID) {
			continue
		}
		if !cueInsideAuthorityEnvelope(cue, sourceByID, regions) {
			if len(cue.SourceIDs) == 0 {
				return fmt.Errorf("recovered cues must stay inside a repair region")
			}
			return fmt.Errorf(
				"changed cues must stay inside one shared repair region plus the full extents of its referenced source cues",
			)
		}
	}
	return nil
}

func isIdenticalCue(cue AudioRefinedCue, sourceByID map[int]ScriptEntry) bool {
	if len(cue.SourceIDs) != 1 {
		return false
	}
	source := sourceByID[cue.SourceIDs[0]]
	return cue.Text == source.Text && cue.Start == source.Start && cue.End == source.End
}

func cueInsideAuthorityEnvelope(
	cue AudioRefinedCue,
	sourceByID map[int]ScriptEntry,
	regions []Interval,
) bool {
	start, _ := ParseTime(cue.Start)
	end, _ := ParseTime(cue.End)
	cueInterval := Interval{Start: start, End: end}
	if len(cue.SourceIDs) == 0 {
		for _, region := range regions {
			expanded := Interval{
				Start: region.Start - RepairWindowSeconds,
				End:   region.End + RepairWindowSeconds,
			}
			if intervalContains(expanded, cueInterval) {
				return true
			}
		}
		return false
	}
	for _, region := range regions {
		envelope := region
		shared := true
		for _, sourceID := range cue.SourceIDs {
			source := sourceByID[sourceID]
			sourceStart, _ := ParseTime(source.Start)
			sourceEnd, _ := ParseTime(source.End)
			sourceInterval := Interval{Start: sourceStart, End: sourceEnd}
			if !Intersects(sourceInterval, []Interval{region}) {
				shared = false
			}
			if sourceStart < envelope.Start {
				envelope.Start = sourceStart
			}
			if sourceEnd > envelope.End {
				envelope.End = sourceEnd
			}
		}
		if !shared {
			continue
		}
		envelope.Start -= RepairWindowSeconds
		envelope.End += RepairWindowSeconds
		if intervalContains(envelope, cueInterval) {
			return true
		}
	}
	return false
}

func intervalContains(container, value Interval) bool {
	return container.Start <= value.Start && value.End <= container.End
}
