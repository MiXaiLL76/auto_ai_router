package models

import (
	"encoding/json"
	"fmt"
	"io"
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

	var data []byte
	var err error

	// Parse the link to determine source type
	if strings.HasPrefix(link, "file://") {
		// File source
		filePath := strings.TrimPrefix(link, "file://")
		data, err = loadFromFile(filePath)
	} else if strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") {
		// HTTP source
		data, err = loadFromHTTP(link)
	} else if !strings.Contains(link, "://") {
		// Treat as file path without file:// prefix
		data, err = loadFromFile(link)
	} else {
		return nil, fmt.Errorf("unsupported link format: %s", link)
	}

	if err != nil {
		return nil, err
	}

	// Parse JSON
	var rawPrices map[string]*ModelPrice
	if err := json.Unmarshal(data, &rawPrices); err != nil {
		return nil, fmt.Errorf("failed to parse model prices JSON: %w", err)
	}

	// Normalize model names (convert keys to normalized format)
	normalizedPrices := make(map[string]*ModelPrice)
	for fullName, price := range rawPrices {
		normalized := NormalizeModelName(fullName)
		normalizedPrices[normalized] = price
	}

	return normalizedPrices, nil
}

// loadFromFile reads model prices from a file
func loadFromFile(filePath string) ([]byte, error) {
	// Validate path to prevent directory traversal attacks
	cleanPath := filepath.Clean(filePath)
	if cleanPath != filePath {
		// Path was cleaned (had .. or other suspicious patterns)
		// This is a security check - relative paths are OK, but directory traversal is not
		return nil, fmt.Errorf("path contains invalid patterns: %s", filePath)
	}

	// Check file size first
	stat, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	if stat.Size() > MaxFileSizeBytes {
		return nil, fmt.Errorf("model prices file exceeds 100MB: %d bytes", stat.Size())
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return data, nil
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

	resp, err := client.Get(link)
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

	// Split by '.' to remove namespace prefixes (e.g., "anthropic.claude" -> take last part)
	// But keep the model name intact
	if len(parts) == 1 && strings.Contains(fullName, ".") {
		// Format like "anthropic.claude/model" was not found, try splitting by dot
		dotParts := strings.Split(fullName, ".")
		modelName = dotParts[len(dotParts)-1]
	}

	// Convert to lowercase for case-insensitive matching
	return strings.ToLower(modelName)
}
