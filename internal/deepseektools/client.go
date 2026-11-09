// Package deepseektools executes DeepSeek's own Search and Fetch tools on
// behalf of a model that cannot run them itself.
//
// DeepSeek exposes two REST endpoints outside the chat protocol:
//
//	POST {base_url}/v1/tools/search  {"query": ..., "count": ..., ...}
//	POST {base_url}/v1/tools/fetch  {"url": ...}
//
// deepseek-flash (DeepSeek-V4.1-Flash-1017) does not implement the built-in
// web_search tool at all -- a request carrying it is rejected -- so the
// router has to run the search itself, hand the result to the model as a
// tool result, and let the model continue.
//
// The client authenticates with the credential's own API key and builds every
// header from scratch: the caller's inbound Authorization never reaches
// DeepSeek.
package deepseektools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
)

// Logger is the optional logger of a Client: it records every Search and
// Fetch call -- the path, the status and how long it took -- so the calls
// the router makes on the model's behalf are visible in the router's own
// logs. A Client without one stays silent.
type Logger interface {
	DebugContext(ctx context.Context, msg string, args ...any)
	WarnContext(ctx context.Context, msg string, args ...any)
}

const (
	// SearchPath is DeepSeek's search endpoint, relative to the credential's
	// base URL.
	SearchPath = "/v1/tools/search"
	// FetchPath is DeepSeek's page-fetch endpoint, relative to the
	// credential's base URL.
	FetchPath = "/v1/tools/fetch"
)

// endpointPath returns the URL of a tool endpoint for a base URL.
//
// The endpoints live under /v1, but a credential's base URL may already end
// with it (https://api.deepseek.com/v1) or not
// (https://api.deepseek.com): appending the path verbatim would ask for
// /v1/v1/tools/search on the second kind.
func endpointPath(baseURL, path string) string {
	base := strings.TrimSuffix(baseURL, "/")
	if strings.HasSuffix(base, "/v1") {
		return base + strings.TrimPrefix(path, "/v1")
	}
	return base + path
}

// ClientConfig configures a Client.
type ClientConfig struct {
	// BaseURL is the credential's base URL (https://api.deepseek.com or
	// .../v1).
	BaseURL string
	// APIKey is the credential's key, sent as a bearer token.
	APIKey string
	// Timeout bounds one Search/Fetch call.
	Timeout time.Duration
	// ProxyURL, when set, routes the router's own outbound connection
	// through an upstream HTTP/HTTPS proxy.
	ProxyURL string
	// MaxRetries is how many times a call is retried after a 429 or a 5xx.
	MaxRetries int
	// Logger, when set, records every call this client makes.
	Logger Logger
}

// Client calls DeepSeek's Search and Fetch endpoints.
type Client struct {
	baseURL    string
	apiKey     string
	http       *http.Client
	MaxRetries int
	logger     Logger
}

// NewClient builds a Client over the shared HTTP transport (connection
// pooling, HTTP/2 keepalive) so this feature does not hand-roll its own.
func NewClient(cfg ClientConfig) *Client {
	httpCfg := httputil.DefaultHTTPClientConfig()
	httpCfg.Timeout = cfg.Timeout
	client := httputil.NewHTTPClient(httpCfg)
	if cfg.ProxyURL != "" {
		client = httputil.WithClientProxy(client, cfg.ProxyURL)
	}
	return &Client{
		baseURL:    strings.TrimSuffix(cfg.BaseURL, "/"),
		apiKey:     cfg.APIKey,
		http:       client,
		MaxRetries: cfg.MaxRetries,
		logger:     cfg.Logger,
	}
}

// SearchRequest is DeepSeek's /v1/tools/search request body. Only Query is
// required; the rest are the documented optional fields.
//
// Unknown fields are ignored by DeepSeek, so the model-supplied arguments
// are mapped onto exactly these.
type SearchRequest struct {
	// Query is the search query: 1 to 128 Unicode characters, not all
	// whitespace. Required.
	Query string `json:"query"`
	// Count is the number of results to return, 1 to 50. Zero means the
	// provider's default (10).
	Count int `json:"count,omitempty"`
	// OriQuestion is the user's original question, which helps the provider
	// understand the query. Optional.
	OriQuestion string `json:"ori_question,omitempty"`
	// Localization is an optional IP address used to localize results.
	Localization string `json:"localization,omitempty"`
	// NeedContent asks the provider to return page content in each result's
	// content field. It does NOT create a separate Fetch charge.
	NeedContent bool `json:"need_content,omitempty"`
}

// SearchResult is one item of DeepSeek's search results array.
type SearchResult struct {
	RerankScore float64 `json:"rerank_score"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Snippet     string  `json:"snippet"`
	Summary     string  `json:"summary"`
	Content     string  `json:"content"`
	SiteName    *string `json:"site_name"`
	SiteIcon    *string `json:"site_icon"`
	Date        *string `json:"date"`
	Authority   int     `json:"authority"`
}

// SearchResponse is DeepSeek's /v1/tools/search response body.
type SearchResponse struct {
	Results []SearchResult `json:"results"`
}

// FetchRequest is DeepSeek's /v1/tools/fetch request body.
type FetchRequest struct {
	// URL is the page to fetch: at most 8192 characters, an HTTP or HTTPS
	// URL with a host, no embedded credentials. Required.
	URL string `json:"url"`
}

// FetchResponse is DeepSeek's /v1/tools/fetch response body.
type FetchResponse struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// ErrInvalidBody is returned when a request body is not a JSON object.
var ErrInvalidBody = fmt.Errorf("deepseektools: request body must be a JSON object")

// do posts body to path and returns the status code and the raw response
// body. It never returns an error for a non-2xx response: that is a valid
// provider answer the caller has to interpret, not a client error.
func (c *Client) do(ctx context.Context, path string, body any) (int, []byte, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return 0, nil, fmt.Errorf("deepseektools: encoding request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpointPath(c.baseURL, path), bytes.NewReader(encoded))
	if err != nil {
		return 0, nil, fmt.Errorf("deepseektools: building upstream request: %w", err)
	}
	// Built from scratch: the caller's own Authorization/X-Forwarded-For/etc.
	// never reach DeepSeek.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	started := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		c.logError(ctx, "DeepSeek tool call failed", path, 0, time.Since(started), err)
		return 0, nil, fmt.Errorf("deepseektools: upstream request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.logError(ctx, "DeepSeek tool call failed", path, resp.StatusCode, time.Since(started), err)
		return resp.StatusCode, nil, fmt.Errorf("deepseektools: reading upstream response: %w", err)
	}
	c.logCall(ctx, path, resp.StatusCode, time.Since(started), len(respBody))
	return resp.StatusCode, respBody, nil
}

// logCall records a finished call: the endpoint, the status, how long it
// took and how much came back.
func (c *Client) logCall(ctx context.Context, path string, status int, took time.Duration, size int) {
	if c.logger == nil {
		return
	}
	c.logger.DebugContext(ctx, "DeepSeek tool call",
		"path", path, "status", status, "duration_ms", took.Milliseconds(), "response_bytes", size)
}

// logError records a call that did not finish.
func (c *Client) logError(ctx context.Context, msg, path string, status int, took time.Duration, err error) {
	if c.logger == nil {
		return
	}
	c.logger.WarnContext(ctx, msg,
		"path", path, "status", status, "duration_ms", took.Milliseconds(), "error", err)
}

// Search runs a search and returns the parsed response.
//
// It returns the provider's own status code and body for a non-2xx answer
// (the caller relays the provider's error format verbatim), so only a
// transport failure is an error here.
//
// A 429 or a 5xx is retried with a delay, as the provider's guide asks: those
// are transient. Other 4xx answers are returned as they are -- retrying them
// without changing the request would only waste the account's quota.
func (c *Client) Search(ctx context.Context, req SearchRequest) (int, *SearchResponse, []byte, error) {
	return c.searchWithRetry(ctx, req, c.MaxRetries)
}

// searchWithRetry runs a search, retrying a transient failure.
func (c *Client) searchWithRetry(ctx context.Context, req SearchRequest, retries int) (int, *SearchResponse, []byte, error) {
	var status int
	var body []byte
	var err error
	for attempt := 0; ; attempt++ {
		status, body, err = c.do(ctx, SearchPath, req)
		if err != nil {
			return 0, nil, nil, err
		}
		if !isRetryableStatus(status) || attempt >= retries {
			break
		}
		if waitErr := SleepWithContext(ctx, retryDelay(status, attempt)); waitErr != nil {
			return status, nil, body, waitErr
		}
	}
	if status < 200 || status >= 300 {
		return status, nil, body, nil
	}
	var parsed SearchResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return status, nil, body, fmt.Errorf("deepseektools: decoding search response: %w", err)
	}
	return status, &parsed, body, nil
}

// Fetch fetches a page and returns the parsed response.
//
// As in Search, a non-2xx answer is returned to the caller as the provider's
// status and body, not as an error, and a transient failure is retried.
func (c *Client) Fetch(ctx context.Context, req FetchRequest) (int, *FetchResponse, []byte, error) {
	return c.fetchWithRetry(ctx, req, c.MaxRetries)
}

// fetchWithRetry fetches a page, retrying a transient failure.
func (c *Client) fetchWithRetry(ctx context.Context, req FetchRequest, retries int) (int, *FetchResponse, []byte, error) {
	var status int
	var body []byte
	var err error
	for attempt := 0; ; attempt++ {
		status, body, err = c.do(ctx, FetchPath, req)
		if err != nil {
			return 0, nil, nil, err
		}
		if !isRetryableStatus(status) || attempt >= retries {
			break
		}
		if waitErr := SleepWithContext(ctx, retryDelay(status, attempt)); waitErr != nil {
			return status, nil, body, waitErr
		}
	}
	if status < 200 || status >= 300 {
		return status, nil, body, nil
	}
	var parsed FetchResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return status, nil, body, fmt.Errorf("deepseektools: decoding fetch response: %w", err)
	}
	return status, &parsed, body, nil
}

// isRetryableStatus reports whether a status is worth retrying: the provider
// asks for a retry after a 429 (its QPS limit) and after a 5xx.
func isRetryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// retryDelay is how long to wait before the next attempt: the provider's
// Retry-After when it gave one, otherwise a growing delay.
func retryDelay(status int, attempt int) time.Duration {
	if status == http.StatusTooManyRequests {
		return time.Duration(attempt+1) * 500 * time.Millisecond
	}
	return time.Duration(attempt+1) * time.Second
}

// SleepWithContext waits for the given duration, or until the context is
// done. It returns the context's error when the wait was cut short.
func SleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
