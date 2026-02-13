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
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
)

const (
	defaultTimeout             = 5 * time.Second
	maxResponseSizeBytes       = 10 * 1024 * 1024 // 10MB limit for proxy responses
	minProxyFetchInterval      = 100 * time.Millisecond
	defaultMaxIdleConns        = 100
	defaultMaxIdleConnsPerHost = 10
	defaultIdleConnTimeout     = 90 * time.Second
)

// HTTPClientConfig holds configuration for HTTP client creation
type HTTPClientConfig struct {
	Timeout             time.Duration
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
}

// DefaultHTTPClientConfig returns HTTP client configuration with sensible defaults
// Used for consistent HTTP client configuration across the application
func DefaultHTTPClientConfig() *HTTPClientConfig {
	return &HTTPClientConfig{
		Timeout:             defaultTimeout,
		MaxIdleConns:        defaultMaxIdleConns,
		MaxIdleConnsPerHost: defaultMaxIdleConnsPerHost,
		IdleConnTimeout:     defaultIdleConnTimeout,
	}
}

// NewHTTPClient creates a new HTTP client with the given configuration
// This centralized factory ensures consistent HTTP client behavior throughout the application
func NewHTTPClient(cfg *HTTPClientConfig) *http.Client {
	if cfg == nil {
		cfg = DefaultHTTPClientConfig()
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	maxIdleConns := cfg.MaxIdleConns
	if maxIdleConns == 0 {
		maxIdleConns = defaultMaxIdleConns
	}

	maxIdleConnsPerHost := cfg.MaxIdleConnsPerHost
	if maxIdleConnsPerHost == 0 {
		maxIdleConnsPerHost = defaultMaxIdleConnsPerHost
	}

	idleConnTimeout := cfg.IdleConnTimeout
	if idleConnTimeout == 0 {
		idleConnTimeout = defaultIdleConnTimeout
	}

	return &http.Client{
		// No global timeout — streaming responses can run for minutes.
		// ResponseHeaderTimeout on Transport protects the connect + header phase.
		Timeout: 0,
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment, // Support HTTP_PROXY, HTTPS_PROXY, NO_PROXY
			ResponseHeaderTimeout: timeout,                   // Timeout for connect + response headers only
			MaxIdleConns:          maxIdleConns,
			MaxIdleConnsPerHost:   maxIdleConnsPerHost,
			IdleConnTimeout:       idleConnTimeout,
			DisableKeepAlives:     false,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// proxyFetchRateLimiter enforces minimum interval between proxy credential fetches
// to prevent overwhelming proxy servers with frequent requests.
// Uses TimeBasedRateLimiter from the ratelimit package for interval-based rate limiting.
var proxyFetchRateLimiter = ratelimit.NewTimeBasedRateLimiter()

// proxyHTTPClient is the shared HTTP client for proxy fetch operations
// Uses the centralized configuration for consistent behavior
var proxyHTTPClient = NewHTTPClient(&HTTPClientConfig{
	Timeout:             defaultTimeout,
	MaxIdleConns:        defaultMaxIdleConns,
	MaxIdleConnsPerHost: defaultMaxIdleConnsPerHost,
	IdleConnTimeout:     defaultIdleConnTimeout,
})

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

	if err := proxyFetchRateLimiter.Wait(ctx, cred.Name, minProxyFetchInterval); err != nil {
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

	// Send request using centralized HTTP client
	resp, err := proxyHTTPClient.Do(req)
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
