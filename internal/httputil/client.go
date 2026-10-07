// Package httputil provides shared HTTP client construction and helpers.
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
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

const (
	defaultTimeout             = 5 * time.Second
	maxResponseSizeBytes       = 10 * 1024 * 1024 // 10MB limit for proxy responses
	minProxyFetchInterval      = 100 * time.Millisecond
	defaultMaxIdleConns        = 100
	defaultMaxIdleConnsPerHost = 10
	defaultIdleConnTimeout     = 90 * time.Second

	// A pooled HTTP/2 connection whose peer stopped responding (egress proxy ate
	// the RST, remote end went dark) looks identical to a merely-idle one — Go's
	// Transport has no way to tell them apart and will keep handing it to new
	// requests until ResponseHeaderTimeout gives up on whatever request lands on
	// it next, which can take minutes. These two defaults wire up Go's built-in
	// HTTP/2 keepalive (http.HTTP2Config, Go 1.24+): after
	// defaultHTTP2IdlePingTimeout with no frames on a connection, a PING is sent;
	// if it isn't acked within defaultHTTP2PingTimeout, the connection is closed
	// and the next request dials a fresh one. A live-but-slow non-streaming call
	// (no frames while the provider is still generating) is unaffected — the PING
	// rides alongside it, and once acked, the request keeps waiting normally.
	defaultHTTP2IdlePingTimeout = 15 * time.Second
	defaultHTTP2PingTimeout     = 10 * time.Second
)

// ProxyStatusError reports a non-success response from a proxied endpoint.
type ProxyStatusError struct {
	StatusCode int
}

func (e *ProxyStatusError) Error() string {
	return fmt.Sprintf("proxy returned status %d", e.StatusCode)
}

// HTTPClientConfig holds configuration for HTTP client creation
type HTTPClientConfig struct {
	Timeout             time.Duration
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
	// HTTP2IdlePingTimeout and HTTP2PingTimeout configure HTTP/2 keepalive pings
	// (http.HTTP2Config.SendPingTimeout / .PingTimeout) so a pooled connection that
	// has gone silent — not idle, just unresponsive — gets probed and, if it
	// doesn't answer, closed instead of sitting in the pool until
	// ResponseHeaderTimeout gives up on it. Zero falls back to the package
	// defaults; a negative duration (e.g. -1s) on EITHER field explicitly
	// disables HTTP/2 ping liveness checking entirely — plain "-1" has no unit
	// and fails time.ParseDuration, so this is spelled out as a duration, not
	// the bare -1 sentinel request_timeout uses elsewhere in config.go.
	//
	// The disable path always goes through SendPingTimeout (Go's own
	// "0 == no health check is performed"), never through PingTimeout: Go's
	// vendored x/net/http2 (net/http/internal/http2/config.go,
	// setConfigDefaults) unconditionally runs
	// setDefault(&conf.PingTimeout, 1, math.MaxInt64, 15*time.Second), so any
	// PingTimeout below 1ns — including a literal 0 — silently becomes 15s
	// again deep inside net/http, with no way to make it a true "off" on its
	// own. Setting only HTTP2PingTimeout negative therefore still disables
	// the whole check (via SendPingTimeout), not because PingTimeout itself
	// went to 0.
	HTTP2IdlePingTimeout time.Duration
	HTTP2PingTimeout     time.Duration
}

// DefaultHTTPClientConfig returns HTTP client configuration with sensible defaults
// Used for consistent HTTP client configuration across the application
func DefaultHTTPClientConfig() *HTTPClientConfig {
	return &HTTPClientConfig{
		Timeout:              defaultTimeout,
		MaxIdleConns:         defaultMaxIdleConns,
		MaxIdleConnsPerHost:  defaultMaxIdleConnsPerHost,
		IdleConnTimeout:      defaultIdleConnTimeout,
		HTTP2IdlePingTimeout: defaultHTTP2IdlePingTimeout,
		HTTP2PingTimeout:     defaultHTTP2PingTimeout,
	}
}

// newTransport builds the *http.Transport for NewHTTPClient. Split out so the
// HTTP/2 ping wiring can be asserted directly in tests without unwrapping the
// otelhttp.Transport that NewHTTPClient returns it inside.
func newTransport(cfg *HTTPClientConfig) *http.Transport {
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

	// A negative value on either field is an explicit opt-out, resolved
	// before per-field defaulting: see HTTPClientConfig's doc comment for why
	// disabling always goes through SendPingTimeout (the only field whose
	// zero actually means "off" at the Go level) regardless of which one of
	// the two the caller set negative.
	disablePing := cfg.HTTP2IdlePingTimeout < 0 || cfg.HTTP2PingTimeout < 0

	http2IdlePingTimeout := cfg.HTTP2IdlePingTimeout
	switch {
	case disablePing:
		http2IdlePingTimeout = 0
	case http2IdlePingTimeout == 0:
		http2IdlePingTimeout = defaultHTTP2IdlePingTimeout
	}

	// PingTimeout has no real "off" to preserve here (see the doc comment),
	// so it's always resolved to a usable positive value — including when
	// disablePing is true, where it's simply unused: once SendPingTimeout is
	// 0, Go never sends a ping to wait on an ack for in the first place.
	http2PingTimeout := cfg.HTTP2PingTimeout
	if http2PingTimeout <= 0 {
		http2PingTimeout = defaultHTTP2PingTimeout
	}

	return &http.Transport{
		Proxy:                 proxyFromRequest,
		TLSHandshakeTimeout:   timeout, // Timeout for TLS handshake phase
		ResponseHeaderTimeout: timeout, // Timeout for connect + response headers only
		MaxIdleConns:          maxIdleConns,
		MaxIdleConnsPerHost:   maxIdleConnsPerHost,
		IdleConnTimeout:       idleConnTimeout,
		DisableKeepAlives:     false,
		// Detects a connection that has gone silent (no HTTP/2 frames at all,
		// including on in-flight streams) and closes it if it doesn't answer a
		// PING, instead of leaving a dead connection in the pool for up to
		// ResponseHeaderTimeout on every request that happens to land on it.
		HTTP2: &http.HTTP2Config{
			SendPingTimeout: http2IdlePingTimeout,
			PingTimeout:     http2PingTimeout,
			// errType is one of a small fixed set of lowercase_with_underscores
			// stdlib-internal reason strings (see net/http.HTTP2Config.CountError),
			// safe as a label with no cardinality risk. CountError fires for every
			// client-side HTTP/2 transport error Go recognizes — frame read errors
			// and plain EOF (read_frame_*), GOAWAY (recv_goaway_*) and RST_STREAM
			// (recv_rststream_*) from the peer, all of which are routine and
			// unrelated to our ping config — not just our own ping timeouts. The
			// one reason this specific ping config can trigger is
			// conn_close_lost_ping: the PING went unacked and the connection was
			// torn down. See monitoring.HTTP2TransportErrorsTotal's doc comment —
			// triage by filtering reason="conn_close_lost_ping", not the total.
			CountError: func(errType string) {
				monitoring.HTTP2TransportErrorsTotal.WithLabelValues(errType).Inc()
			},
		},
	}
}

// NewHTTPClient creates a new HTTP client with the given configuration
// This centralized factory ensures consistent HTTP client behavior throughout the application
func NewHTTPClient(cfg *HTTPClientConfig) *http.Client {
	transport := newTransport(cfg)

	return &http.Client{
		// No global timeout — streaming responses can run for minutes.
		// ResponseHeaderTimeout on Transport protects the connect + header phase.
		Timeout: 0,
		// otelhttp creates client spans and injects traceparent into outgoing
		// requests (providers and chained routers). It uses the global
		// TracerProvider, which is a no-op unless OTEL is enabled in config.
		Transport: otelhttp.NewTransport(transport),
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
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

// proxyFetchTimeout is the timeout applied to proxy health/models fetch requests.
// Override at startup via SetProxyFetchTimeout.
var proxyFetchTimeout = defaultTimeout

// SetProxyFetchTimeout reconfigures the shared proxy HTTP client with a new timeout.
// Must be called before the first proxy fetch (typically right after config load).
func SetProxyFetchTimeout(d time.Duration) {
	if d <= 0 {
		return
	}
	proxyFetchTimeout = d
	proxyHTTPClient = NewHTTPClient(&HTTPClientConfig{
		Timeout:             d,
		MaxIdleConns:        defaultMaxIdleConns,
		MaxIdleConnsPerHost: defaultMaxIdleConnsPerHost,
		IdleConnTimeout:     defaultIdleConnTimeout,
	})
}

// FetchFromProxy makes an HTTP GET request to a proxy credential
// and returns the response body. Handles timeouts, auth headers, and error logging.
// Note: caller should provide ctx with timeout if defaultTimeout is insufficient
func FetchFromProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	path string,
	logger *slog.Logger,
) ([]byte, error) {
	body, _, err := FetchResponseFromProxy(ctx, cred, path, logger)
	return body, err
}

// FetchResponseFromProxy makes an HTTP GET request to a proxy credential and
// returns the response body together with response headers. It is useful for
// callers that need AIR-specific response metadata while preserving the
// FetchFromProxy body-only API for existing call sites.
func FetchResponseFromProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	path string,
	logger *slog.Logger,
) ([]byte, http.Header, error) {
	// Create context with timeout if not already set
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, proxyFetchTimeout)
		defer cancel()
	}

	if err := proxyFetchRateLimiter.Wait(ctx, cred.Name, minProxyFetchInterval); err != nil {
		logger.Debug("Proxy fetch rate limited",
			"credential", cred.Name,
			"path", path,
			"error", err,
		)
		return nil, nil, fmt.Errorf("proxy fetch rate limited: %w", err)
	}

	// Build URL
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	url := baseURL + path

	// Create request
	req, err := http.NewRequestWithContext(WithProxyURL(ctx, cred.ProxyURL), "GET", url, nil)
	if err != nil {
		logger.Error("Failed to create request",
			"credential", cred.Name,
			"url", url,
			"error", err,
		)
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
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
		return nil, nil, fmt.Errorf("failed to fetch: %w", err)
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
		return nil, resp.Header.Clone(), &ProxyStatusError{StatusCode: resp.StatusCode}
	}

	// Read body with size limit
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSizeBytes))
	if err != nil {
		logger.Error("Failed to read response body",
			"credential", cred.Name,
			"error", err,
		)
		return nil, resp.Header.Clone(), fmt.Errorf("failed to read body: %w", err)
	}

	return body, resp.Header.Clone(), nil
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
