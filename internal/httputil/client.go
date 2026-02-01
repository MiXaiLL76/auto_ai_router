package httputil

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

const defaultTimeout = 5 * time.Second

// FetchFromProxy makes an HTTP GET request to a proxy credential
// and returns the response body. Handles timeouts, auth headers, and error logging.
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
		body, _ := io.ReadAll(resp.Body)
		logger.Error("Proxy returned non-200 status",
			"credential", cred.Name,
			"status", resp.StatusCode,
			"response_preview", string(body[:min(len(body), 200)]),
		)
		return nil, fmt.Errorf("proxy returned status %d", resp.StatusCode)
	}

	// Read body
	body, err := io.ReadAll(resp.Body)
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
	v interface{},
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
