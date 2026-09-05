package vtt

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Cue struct {
	Identifier string
	Start      string
	End        string
	Settings   string
	Text       string
}

type File struct {
	Header string
	Styles []string
	Cues   []Cue
}

func Read(path string) (*File, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	normalized := strings.ReplaceAll(string(contents), "\r\n", "\n")
	normalized = strings.TrimPrefix(strings.ReplaceAll(normalized, "\r", "\n"), "\ufeff")
	lines := strings.Split(normalized, "\n")
	header := strings.TrimSpace(lines[0])
	if header != "WEBVTT" && !strings.HasPrefix(header, "WEBVTT ") && !strings.HasPrefix(header, "WEBVTT\t") {
		return nil, fmt.Errorf("invalid WebVTT header")
	}
	result := &File{Header: header}
	for i := 1; i < len(lines); {
		for i < len(lines) && strings.TrimSpace(lines[i]) == "" {
			i++
		}
		if i >= len(lines) {
			break
		}
		if strings.HasPrefix(lines[i], "NOTE") || lines[i] == "STYLE" || lines[i] == "REGION" {
			start := i
			for i < len(lines) && strings.TrimSpace(lines[i]) != "" {
				i++
			}
			if lines[start] == "STYLE" {
				result.Styles = append(result.Styles, strings.Join(lines[start:i], "\n"))
			}
			continue
		}
		identifier := ""
		timing := lines[i]
		if !strings.Contains(timing, "-->") {
			identifier = timing
			i++
			if i >= len(lines) {
				return nil, fmt.Errorf("cue %q has no timing", identifier)
			}
			timing = lines[i]
		}
		parts := strings.SplitN(timing, "-->", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid cue timing %q", timing)
		}
		start := strings.TrimSpace(parts[0])
		right := strings.Fields(strings.TrimSpace(parts[1]))
		if len(right) == 0 {
			return nil, fmt.Errorf("invalid cue timing %q", timing)
		}
		end := right[0]
		settings := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(parts[1]), end))
		i++
		var text []string
		for i < len(lines) && strings.TrimSpace(lines[i]) != "" {
			text = append(text, lines[i])
			i++
		}
		if len(text) == 0 {
			continue
		}
		result.Cues = append(result.Cues, Cue{
			Identifier: identifier,
			Start:      start,
			End:        end,
			Settings:   settings,
			Text:       strings.Join(text, "\n"),
		})
	}
	return result, nil
}

func (file *File) Bytes() []byte {
	var output strings.Builder
	output.WriteString("WEBVTT\n\n")
	for _, style := range file.Styles {
		output.WriteString(style)
		output.WriteString("\n\n")
	}
	for _, cue := range file.Cues {
		if cue.Identifier != "" {
			output.WriteString(cue.Identifier)
			output.WriteByte('\n')
		}
		fmt.Fprintf(&output, "%s --> %s", cue.Start, cue.End)
		if cue.Settings != "" {
			output.WriteByte(' ')
			output.WriteString(cue.Settings)
		}
		output.WriteByte('\n')
		output.WriteString(cue.Text)
		output.WriteString("\n\n")
	}
	return []byte(output.String())
}

func (file *File) SaveAtomic(path string) error {
	return WriteAtomic(path, file.Bytes())
}

func WriteAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*.tmp.vtt")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	w := bufio.NewWriter(tmp)
	if _, err = w.Write(data); err == nil {
		err = w.Flush()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}
