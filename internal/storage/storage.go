package storage

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
)

const ManifestName = "manifest.json"

func AtomicWriteJSON(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err = os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func Fingerprint(path string) (map[string]any, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if resolved, resolveErr := filepath.EvalSymlinks(abs); resolveErr == nil {
		abs = resolved
	}
	return map[string]any{"path": abs, "size": st.Size(), "mtime_ns": st.ModTime().UnixNano()}, nil
}

func HashJSON(value any) (string, error) {
	serialized, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(serialized)
	return hex.EncodeToString(hash[:]), nil
}
