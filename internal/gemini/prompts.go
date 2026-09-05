package gemini

import (
	"fmt"
	"strings"

	"github.com/gotenksIN/video-subtitler/internal/core"
)

func generationPrompt(duration float64, title string, names []string) string {
	context := ""
	if title != "" {
		context = fmt.Sprintf(`SOURCE CONTEXT

Source title: %s
Names in the source title are candidate identities only.
They do not prove which speaker said a specific line.

`, title)
	}
	if len(names) > 0 {
		context += `CANDIDATE SPEAKER IDENTITIES

- ` + strings.Join(names, "\n- ") + `

Use a candidate's canonical English name only when direct in-clip evidence establishes attribution in this clip.
Direct evidence includes a visible name banner, lower-third, title card, or spoken introduction.
Never assign identities from appearance alone.
Leave uncertain dialogue unlabeled.

`
	}
	return fmt.Sprintf(`You are an expert subtitle generator and translator.

Watch this %.3f-second video clip.

Generate accurate, natural English subtitles for dialogue and meaningful on-screen text throughout the entire clip.

%sTIMING

1. Use clip-relative timestamps from 00:00:00.000 through %s. Start and end spoken cues at exact audible syllables. Time on-screen text to visibility.
2. Preserve silent gaps. Do not stretch cues through silence, reactions, or scene changes. Sort cues and do not overlap them.
3. Avoid cues shorter than 500 milliseconds. Combine only adjacent speech by the same speaker when timing and meaning stay intact.

TRANSLATION

4. Translate all meaningful speech and on-screen text to natural English. Never return source-language transcription instead.
5. Preserve every meaningful question, answer, joke, reaction, product detail, qualification, name, brand, title, and recurring term.
Do not summarize, omit, infer missing dialogue, or invent facts.
6. Prefer faithful clear English over paraphrase.
Preserve precise cultural terms when English has no equivalent, but do not replace understandable English with unexplained romanization.
Transliterate uncertain proper nouns conservatively and preserve wordplay where possible.

SPEAKER LABELS

7. Use a person's name only when direct in-clip evidence establishes attribution. Never identify from appearance alone.
8. Otherwise use a stable descriptive role only when clear, or leave dialogue unlabeled. Never use generic numbered speakers.
9. Format labels as "Name: Dialogue". Put attributed turns on separate lines. Never label on-screen text.

ON-SCREEN TEXT

10. Include meaningful editorial text. Ignore decorative text, logos, watermarks, repeated UI, and irrelevant text.
11. Keep editorial text distinct from speech and wrap it in square brackets without mechanical prefixes. Do not combine unrelated speech and text.
12. Do not describe visible actions or sounds unless corresponding written text appears. Translate editorial idioms. Do not quote ordinary speech.

FORMATTING

13. Use sequential integer IDs starting at 0. Use at most 42 characters per line and two lines per cue without changing meaning.
14. Return only a valid JSON object matching the required schema with a "captions" array. No markdown or explanations.
`, duration, context, core.MustFormatTime(duration))
}

func researchPrompt(title string, ordinary, youtube []string) string {
	return fmt.Sprintf(`You research speaker identities and topic terminology for an English subtitle localization pass.

Use Google Search at least once and reputable evidence. Return concise plain text with exactly these sections:

PARTICIPANTS AND SPEAKERS:
Begin each entry with the canonical English public name or stable role followed by a colon, aliases, role, and cited evidence.

TOPIC TERMINOLOGY AND PROPER NOUNS:
Give canonical English spelling of recurring proper nouns, titles, organizations, products, and locations, one per line with a citation.

SOURCE TITLE
%s

CONTEXT URLS
%s

YOUTUBE VIDEO URL IDENTIFIERS (analyzed separately; do not open)
%s

Rank reputable web evidence above the source title.
Research establishes spelling and verified entities only.
It never establishes, invents, or changes dialogue, meaning, facts, or events.
State a stable role only when clear; otherwise state that the speaker stays unlabeled.
Return plain text without markdown.`, title, strings.Join(ordinary, "\n"), strings.Join(youtube, "\n"))
}

func youtubePrompt(title string) string {
	return fmt.Sprintf(`Watch the attached public YouTube video.
Return concise plain text listing each participant's official English name and role.
Include timestamped direct speaker-identification evidence from labels, title cards, or spoken introductions.
Establish identity and spelling only.
Never infer or change dialogue, meaning, facts, or events.
Source title: %s.
Return plain text only.`, title)
}

func refinementPrompt(script, title string, context core.PreflightContext) string {
	return fmt.Sprintf(`You are a conservative subtitle proofreader applying a minimal text-only patch.

The complete subtitle script is authoritative.
You cannot hear or see the source.
Assume every cue needs no change.
Return a change only for an objective, necessary correction supported by the script.

SOURCE TITLE: %s
GROUNDED IDENTITY CONTEXT:
%s
GROUNDED TERMINOLOGY CONTEXT:
%s
DIRECT VIDEO IDENTITY ANALYSIS:
%s

Research ranks below explicit script introductions and title cards.
It establishes identity and canonical proper-name spelling only.
It never establishes dialogue, facts, meaning, or events.

Correct only existing label spelling, casing, identity, inconsistent proper nouns, objective grammar, spelling, punctuation, OCR errors, incomprehensible literal idioms, explicit pronoun errors, or clear script-supported continuity errors.
Leave uncertain, subjective, stylistic, grammatical, intelligible, or faithful lines unchanged.

Preserve every line's meaning, words, order, tone, register, repetition, qualifications, and structure by default.
Do not paraphrase, summarize, embellish, smooth, replace synonyms, add claims or explanations, remove fragments, or change text only for line length.
Never duplicate adjacent content.
Never add dialogue, facts, relationships, jokes, or events.

Do not merge, split, reorder, add, or remove cues.
Do not alter IDs or timestamps.
Preserve canonical names and terminology.
Do not alter romanization unless script evidence requires it.
Preserve cultural terms, footnote markers, and meaningful vocalizations.

Never add a speaker label to an unlabeled line, propagate labels, or infer attribution from neighbors, timing, title, research, or analysis.
Correct only an existing label.
Preserve label presence, placement, line breaks, turn boundaries, and spoken names.
Never label editorial text.

Treat every complete outer [bracketed fragment] as protected visual content.
Mixed cues must retain both dialogue and every bracketed fragment in the same cue.
Edit outside brackets unless correcting an objective visual-text error.
Editorial cues preserve every fragment and bracket.
Dialogue cues gain no brackets.
Remove a mechanical "On-screen text:" prefix only while preserving its text.

Return JSON with a "changes" list containing only changed existing numeric IDs and complete corrected text. Return no unchanged cues, timestamps, markdown, or explanation.

SCRIPT
%s`, title, context.IdentityContext, context.TerminologyContext, stringValue(context.YouTubeContext), script)
}
func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func audioPrompt(source []core.ScriptEntry, boundaries []float64, duration float64, title string) string {
	var boundaryLines []string
	var scriptLines []string
	for _, boundary := range boundaries {
		boundaryLines = append(boundaryLines, "- "+core.MustFormatTime(boundary))
	}
	for _, entry := range source {
		scriptLines = append(scriptLines, fmt.Sprintf(
			"[%d] %s --> %s [%s]: %s",
			entry.ID,
			entry.Start,
			entry.End,
			entry.Classification,
			entry.Text,
		))
	}
	return fmt.Sprintf(`You are an expert audio subtitle repair editor. Listen to the complete audio and compare it with the script.

SOURCE TITLE: %s
COMPLETE AUDIO DURATION: %.3f seconds
ACTUAL SEGMENT BOUNDARIES:
%s

SCRIPT:
%s

MODE: boundary-limited.
Repair authority is limited to connected regions ten seconds before through ten seconds after each boundary.
Every cue outside stays byte-identical in text and timing.
Referenced cues must share a repair region.
A changed cue stays within that region plus full source extents and ten seconds on each side.
Recovered cues stay within a region plus ten seconds.

Fix only audio-established boundary faults: missing, duplicate, hallucinated, split, merged, mistranslated, or mistimed dialogue.
Preserve silent gaps and audible syllable timing.
Preserve every bracketed visual fragment exactly within its lineage.
Pure editorial cues stay byte-identical.
Mixed cues preserve each fragment exactly once.
Recovered cues contain spoken dialogue, no brackets, and no guessed identity.
Preserve labels and turn structure.
Do not describe sounds, music, or actions.
Every cue has text.

Return only one sparse JSON patch.
contractVersion is "sparse-patch-v1".
cues contains changed, replacement, split, merged, or recovered cues in script order; omitted cues remain unchanged.
deletedSourceIds deletes only false dialogue.
A rewrite has one sourceIds value.
A merge has increasing IDs and may skip only editorial cues.
A split repeats one singleton ID in adjacent cues.
Recovered dialogue has empty sourceIds.
Changed IDs are referenced or deleted, never both.
Use canonical source-relative HH:MM:SS.mmm timestamps.`, title, duration, strings.Join(boundaryLines, "\n"), strings.Join(scriptLines, "\n"))
}
