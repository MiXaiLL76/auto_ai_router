package httputil

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

const (
	defaultTimeout        = 5 * time.Second
	maxResponseSizeBytes  = 10 * 1024 * 1024 // 10MB limit for proxy responses
	minProxyFetchInterval = 100 * time.Millisecond
)

type proxyFetchLimiter struct {
	mu   sync.Mutex
	last map[string]time.Time
}

func newProxyFetchLimiter() *proxyFetchLimiter {
	return &proxyFetchLimiter{
		last: make(map[string]time.Time),
	}
}

func (l *proxyFetchLimiter) wait(ctx context.Context, key string, minInterval time.Duration) error {
	if minInterval <= 0 {
		return nil
	}

	l.mu.Lock()
	now := time.Now()
	last := l.last[key]
	waitFor := minInterval - now.Sub(last)
	if waitFor <= 0 {
		l.last[key] = now
		l.mu.Unlock()
		return nil
	}
	l.last[key] = now.Add(waitFor)
	l.mu.Unlock()

	timer := time.NewTimer(waitFor)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

var proxyFetchRateLimiter = newProxyFetchLimiter()

// FetchFromProxy makes an HTTP GET request to a proxy credential
// and returns the response body. Handles timeouts, auth headers, and error logging.
// Note: caller should provide ctx with timeout if defaultTimeout is insufficient
func FetchFromProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	path string,
	logger *slog.Logger,
) ([]byte, error) {
	// Create context with timeout if not already set
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultTimeout)
		defer cancel()
	}

	if err := proxyFetchRateLimiter.wait(ctx, cred.Name, minProxyFetchInterval); err != nil {
		logger.Debug("Proxy fetch rate limited",
			"credential", cred.Name,
			"path", path,
			"error", err,
		)
		return nil, fmt.Errorf("proxy fetch rate limited: %w", err)
	}

	// Build URL
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	url := baseURL + path

	// Create request
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		logger.Error("Failed to create request",
			"credential", cred.Name,
			"url", url,
			"error", err,
		)
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add Authorization header if api_key is set
	if cred.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cred.APIKey)
	}

	// Send request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		logger.Error("Failed to fetch from proxy",
			"credential", cred.Name,
			"url", url,
			"error", err,
		)
		return nil, fmt.Errorf("failed to fetch: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			logger.Debug("Failed to close response body", "error", closeErr)
		}
	}()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponseSizeBytes))
		preview := safeStringPreview(body, 200)
		logger.Error("Proxy returned non-200 status",
			"credential", cred.Name,
			"status", resp.StatusCode,
			"response_preview", preview,
		)
		return nil, fmt.Errorf("proxy returned status %d", resp.StatusCode)
	}

	// Read body with size limit
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSizeBytes))
	if err != nil {
		logger.Error("Failed to read response body",
			"credential", cred.Name,
			"error", err,
		)
		return nil, fmt.Errorf("failed to read body: %w", err)
	}

	return body, nil
}

// FetchJSONFromProxy fetches JSON from a proxy and unmarshals it
func FetchJSONFromProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	path string,
	logger *slog.Logger,
	v any,
) error {
	body, err := FetchFromProxy(ctx, cred, path, logger)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(body, v); err != nil {
		logger.Error("Failed to parse JSON response",
			"credential", cred.Name,
			"error", err,
		)
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	return nil
}

// safeStringPreview safely converts bytes to string, handling non-UTF-8 data
// Returns a safe preview of the data, replacing invalid UTF-8 sequences
func safeStringPreview(data []byte, maxLen int) string {
	if len(data) == 0 {
		return ""
	}

	if len(data) > maxLen {
		data = data[:maxLen]
	}

	// Use fmt.Sprintf with %q to safely escape invalid UTF-8 sequences
	// Then remove the surrounding quotes
	escaped := fmt.Sprintf("%q", data)
	if len(escaped) > 2 {
		return escaped[1 : len(escaped)-1]
	}
	return escaped
}
