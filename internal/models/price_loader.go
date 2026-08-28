package models

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
)

const (
	// MaxFileSizeBytes is the maximum size of a model prices file (100MB)
	MaxFileSizeBytes = 100 * 1024 * 1024
)

// LoadModelPrices loads model prices from a link (file:// or http(s)://)
func LoadModelPrices(link string) (map[string]*ModelPrice, error) {
	if link == "" {
		return nil, fmt.Errorf("empty link")
	}

	data, err := LoadModelPriceBytes(link)
	if err != nil {
		return nil, err
	}

	// Parse JSON
	var rawPrices map[string]*ModelPrice
	if err := json.Unmarshal(data, &rawPrices); err != nil {
		return nil, fmt.Errorf("failed to parse model prices JSON: %w", err)
	}

	return normalizeModelPrices(rawPrices), nil
}

func LoadModelPriceBytes(link string) ([]byte, error) {
	if link == "" {
		return nil, fmt.Errorf("empty link")
	}

	var data []byte
	var err error

	// Parse the link to determine source type
	switch {
	case strings.HasPrefix(link, "file://"):
		// File source
		filePath := strings.TrimPrefix(link, "file://")
		data, err = loadFromFile(filePath)
	case strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://"):
		// HTTP source
		data, err = loadFromHTTP(link)
	case !strings.Contains(link, "://"):
		// Treat as file path without file:// prefix
		data, err = loadFromFile(link)
	default:
		return nil, fmt.Errorf("unsupported link format: %s", link)
	}

	if err != nil {
		return nil, err
	}
	return data, nil
}

func normalizeModelPrices(rawPrices map[string]*ModelPrice) map[string]*ModelPrice {
	// Normalize model names (convert keys to normalized format)
	// Also store raw lowercase keys so that provider-prefixed names like
	// "google/gemini-3-flash-preview-highlimits" survive normalisation and
	// can be matched in ModelPriceRegistry.GetPriceAny's first pass.
	normalizedPrices := make(map[string]*ModelPrice)
	normalizedSources := make(map[string]string) // normalized name -> original full name (for collision detection)
	for fullName, price := range rawPrices {
		if price == nil {
			continue
		}
		if price.LiteLLMProvider == "" {
			if slash := strings.IndexByte(fullName, '/'); slash > 0 {
				price.LiteLLMProvider = fullName[:slash]
			} else {
				price.LiteLLMProvider = inferProviderFromModelName(fullName)
			}
		}
		normalized := NormalizeModelName(fullName)

		// Store the raw lowercase key alongside the normalised one so that
		// Update/LoadPrices can put both into the registry, enabling the
		// two-pass lookup in GetPriceAny. Done unconditionally, before the
		// collision check below, so this entry's own distinct price is never
		// lost even when it loses the shared normalized key — and so the
		// outcome doesn't depend on which entry Go's randomized map
		// iteration happens to visit first (see the collision branch below).
		raw := strings.ToLower(strings.TrimSpace(fullName))
		if raw != normalized {
			normalizedPrices[raw] = price
		}

		if existingFullName, exists := normalizedSources[normalized]; exists {
			existingIsBare := !strings.Contains(existingFullName, "/")
			newIsBare := !strings.Contains(fullName, "/")

			if existingIsBare && !newIsBare {
				// Existing entry is a bare model name (e.g. "gpt-4") and
				// the new entry has a provider prefix (e.g. "openai/gpt-4").
				// Keep the bare entry as owner of the shared normalized key
				// — it is more specific in the two-pass lookup — regardless
				// of iteration order. The prefixed entry's own raw key was
				// already stored above, so it isn't lost.
				continue
			}

			slog.Warn("normalized model name collision: entry will be overwritten",
				"normalized_name", normalized,
				"existing_entry", existingFullName,
				"new_entry", fullName,
			)
		}
		normalizedSources[normalized] = fullName
		normalizedPrices[normalized] = price
	}

	return normalizedPrices
}

// loadFromFile reads model prices from a file
func loadFromFile(filePath string) ([]byte, error) {
	// Validate path to prevent directory traversal attacks
	if hasPathTraversal(filePath) {
		return nil, fmt.Errorf("path contains traversal segments: %s", filePath)
	}

	cleanPath := filepath.Clean(filePath)

	// Check file size first
	stat, err := os.Stat(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	if stat.Size() > MaxFileSizeBytes {
		return nil, fmt.Errorf("model prices file exceeds 100MB: %d bytes", stat.Size())
	}

	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return data, nil
}

// hasPathTraversal checks whether a path contains explicit ".." traversal segments.
func hasPathTraversal(path string) bool {
	for _, part := range strings.Split(filepath.ToSlash(path), "/") {
		if part == ".." {
			return true
		}
	}
	return false
}

// loadFromHTTP fetches model prices from HTTP(S) endpoint
func loadFromHTTP(link string) ([]byte, error) {
	// Validate URL format
	parsedURL, err := url.Parse(link)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme: %s (must be http or https)", parsedURL.Scheme)
	}

	// Create HTTP client with timeout
	client := httputil.NewHTTPClient(nil)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, link, nil)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch from URL: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d %s", resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	// Check Content-Length header if present
	if resp.ContentLength > MaxFileSizeBytes {
		return nil, fmt.Errorf("model prices file exceeds 100MB: %d bytes", resp.ContentLength)
	}

	// Read body with size limit
	limitedReader := io.LimitReader(resp.Body, MaxFileSizeBytes+1)
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if int64(len(data)) > MaxFileSizeBytes {
		return nil, fmt.Errorf("model prices file exceeds 100MB: %d bytes", len(data))
	}

	return data, nil
}

// NormalizeModelName extracts the model name from various formats
// Examples:
//   - "openai/gpt-4-turbo" -> "gpt-4-turbo"
//   - "anthropic.claude/claude-3-opus" -> "claude-3-opus"
//   - "vertex/gemini-1.5-pro" -> "gemini-1.5-pro"
//   - "claude-sonnet" -> "claude-sonnet"
//
// Versions are preserved (gpt-4-turbo stays gpt-4-turbo)
func NormalizeModelName(fullName string) string {
	// Trim whitespace
	fullName = strings.TrimSpace(fullName)

	// Split by '/' to extract the model name part
	parts := strings.Split(fullName, "/")
	modelName := parts[len(parts)-1]

	// Convert to lowercase for case-insensitive matching
	return strings.ToLower(modelName)
}

// modelNameProviderPrefixes maps well-known bare model-name prefixes to their
// provider tag, for price files that key entries by plain model name (e.g.
// "gemini-2.5-pro") rather than "provider/model" (e.g. "vertex_ai/gemini-2.5-pro")
// and don't set litellm_provider explicitly. Only unambiguous, first-party
// naming conventions are listed here to avoid misclassifying unrelated models.
var modelNameProviderPrefixes = []struct {
	prefix   string
	provider string
}{
	{"gemini-", "gemini"},
	{"vertex-", "vertex"},
	{"google-", "google"},
}

// inferProviderFromModelName is a last-resort fallback for price entries that
// have neither an explicit litellm_provider nor a "provider/model" key
// convention. It only matches a known, unambiguous model-name prefix and
// returns "" (no guess) otherwise.
func inferProviderFromModelName(fullName string) string {
	name := strings.ToLower(strings.TrimSpace(fullName))
	for _, p := range modelNameProviderPrefixes {
		if strings.HasPrefix(name, p.prefix) {
			return p.provider
		}
	}
	return ""
}
