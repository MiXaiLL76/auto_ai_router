// Package moonshottools executes Moonshot's own Search, Search Pro and Fetch
// tools on behalf of a model that cannot run them itself.
//
// Moonshot exposes three REST endpoints outside the chat protocol:
//
//	POST {base_url}/v1/tools/search      {"text_query": ..., ...}
//	POST {base_url}/v1/tools/search_pro  {"text_query": ..., "sites": [...], ...}
//	POST {base_url}/v1/tools/fetch       {"url": ...}
//
// The router's built-in $web_search was switched off on 2026-10-20: its
// replacement is these separate REST APIs, which are not reachable by simply
// passing a parameter to the model. So the router takes them over: the
// client's search request becomes function tools the model can call, the
// router runs the REST calls itself, and the results go back to the model as
// ordinary tool results.
//
// The client authenticates with the credential's own API key and builds every
// header from scratch: the caller's inbound Authorization never reaches
// Moonshot.
package moonshottools

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

// Logger is the optional logger of a Client: it records every Search, Search
// Pro and Fetch call -- the path, the status and how long it took -- so the
// calls the router makes on the model's behalf are visible in the router's own
// logs. A Client without one stays silent.
type Logger interface {
	DebugContext(ctx context.Context, msg string, args ...any)
	WarnContext(ctx context.Context, msg string, args ...any)
}

const (
	// SearchPath is Moonshot's Basic search endpoint, relative to the
	// credential's base URL.
	SearchPath = "/v1/tools/search"
	// SearchProPath is Moonshot's Pro search endpoint, relative to the
	// credential's base URL.
	SearchProPath = "/v1/tools/search_pro"
	// FetchPath is Moonshot's page-fetch endpoint, relative to the
	// credential's base URL.
	FetchPath = "/v1/tools/fetch"
)

// endpointPath returns the URL of a tool endpoint for a base URL.
//
// The endpoints live under /v1, but a credential's base URL may already end
// with it (https://api.moonshot.ai/v1) or not (https://api.moonshot.ai):
// appending the path verbatim would ask for /v1/v1/tools/search on the
// second kind.
func endpointPath(baseURL, path string) string {
	base := strings.TrimSuffix(baseURL, "/")
	if strings.HasSuffix(base, "/v1") {
		return base + strings.TrimPrefix(path, "/v1")
	}
	return base + path
}

// ClientConfig configures a Client.
type ClientConfig struct {
	// BaseURL is the credential's base URL (https://api.moonshot.ai or
	// .../v1).
	BaseURL string
	// APIKey is the credential's key, sent as a bearer token.
	APIKey string
	// Timeout bounds one Search/Search Pro/Fetch call.
	Timeout time.Duration
	// ProxyURL, when set, routes the router's own outbound connection
	// through an upstream HTTP/HTTPS proxy.
	ProxyURL string
	// MaxRetries is how many times a call is retried after a 429 or a 5xx.
	MaxRetries int
	// Logger, when set, records every call this client makes.
	Logger Logger
}

// Client calls Moonshot's Search, Search Pro and Fetch endpoints.
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

// SearchRequest is Moonshot's Basic search request body. Only TextQuery is
// required; the rest are the documented optional fields.
//
// Unknown fields are ignored by Moonshot, so the model-supplied arguments
// are mapped onto exactly these.
type SearchRequest struct {
	// TextQuery is the search query. Required.
	TextQuery string `json:"text_query"`
	// Limit is the number of results to return. Zero means the provider's
	// default.
	Limit int `json:"limit,omitempty"`
	// TimeoutSeconds bounds the provider's own search. Zero means the
	// provider's default.
	TimeoutSeconds int `json:"timeout_seconds,omitempty"`
	// IncludeContent asks the provider to return page content in each
	// result's content field. It does NOT create a separate Fetch charge.
	IncludeContent bool `json:"include_content,omitempty"`
}

// SearchProRequest is Moonshot's Pro search request body.
//
// The Pro search differs from the Basic one in what it can narrow by: the
// sites to search in and the time window to search. TextQuery is still the
// query itself.
type SearchProRequest struct {
	// TextQuery is the search query. Required.
	TextQuery string `json:"text_query"`
	// Limit is the number of results to return. Zero means the provider's
	// default.
	Limit int `json:"limit,omitempty"`
	// TimeoutSeconds bounds the provider's own search. Zero means the
	// provider's default.
	TimeoutSeconds int `json:"timeout_seconds,omitempty"`
	// IncludeContent asks the provider to return page content in each
	// result's content field.
	IncludeContent bool `json:"include_content,omitempty"`
	// Sites narrows the search to these sites. Empty means every site.
	Sites []string `json:"sites,omitempty"`
	// TimeWindow narrows the search to this window. Empty means every
	// window.
	TimeWindow string `json:"time_window,omitempty"`
}

// SearchResult is one item of a search response's results array.
type SearchResult struct {
	Title    string  `json:"title"`
	URL      string  `json:"url"`
	Snippet  string  `json:"snippet"`
	Summary  string  `json:"summary"`
	Content  string  `json:"content"`
	SiteName *string `json:"site_name"`
	Date     *string `json:"date"`
}

// SearchResponse is a search response body.
type SearchResponse struct {
	// Results are the pages the provider found.
	Results []SearchResult `json:"search_results"`
	// Message is the provider's own note, when it answered with one instead
	// of results.
	Message string `json:"message"`
}

// FetchRequest is Moonshot's /v1/tools/fetch request body.
type FetchRequest struct {
	// URL is the page to fetch. Required.
	URL string `json:"url"`
}

// FetchResponse is Moonshot's /v1/tools/fetch response body.
type FetchResponse struct {
	Title string `json:"title"`
	// Markdown is the page's content, as markdown.
	Markdown string `json:"markdown"`
	// Error is the provider's own error code, when it refused the fetch
	// (for instance "markdown_not_found").
	Error string `json:"error"`
}

// do posts body to path and returns the status code and the raw response
// body. It never returns an error for a non-2xx response: that is a valid
// provider answer the caller has to interpret, not a client error.
func (c *Client) do(ctx context.Context, path string, body any) (int, []byte, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return 0, nil, fmt.Errorf("moonshottools: encoding request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpointPath(c.baseURL, path), bytes.NewReader(encoded))
	if err != nil {
		return 0, nil, fmt.Errorf("moonshottools: building upstream request: %w", err)
	}
	// Built from scratch: the caller's own Authorization/X-Forwarded-For/etc.
	// never reach Moonshot.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	started := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		c.logError(ctx, "Moonshot tool call failed", path, 0, time.Since(started), err)
		return 0, nil, fmt.Errorf("moonshottools: upstream request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.logError(ctx, "Moonshot tool call failed", path, resp.StatusCode, time.Since(started), err)
		return resp.StatusCode, nil, fmt.Errorf("moonshottools: reading upstream response: %w", err)
	}
	c.logCall(ctx, path, resp.StatusCode, time.Since(started), len(respBody))
	return resp.StatusCode, respBody, nil
}

// Search runs a Basic search and returns the parsed response.
//
// A non-2xx status is returned as such with the body, not as an error: the
// caller decides what a failed search means (it is never billable).
func (c *Client) Search(ctx context.Context, req SearchRequest) (int, SearchResponse, []byte, error) {
	status, raw, err := c.do(ctx, SearchPath, req)
	if err != nil {
		return status, SearchResponse{}, raw, err
	}
	var response SearchResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return status, SearchResponse{}, raw, fmt.Errorf("moonshottools: parsing search response: %w", err)
	}
	return status, response, raw, nil
}

// SearchPro runs a Pro search and returns the parsed response.
func (c *Client) SearchPro(ctx context.Context, req SearchProRequest) (int, SearchResponse, []byte, error) {
	status, raw, err := c.do(ctx, SearchProPath, req)
	if err != nil {
		return status, SearchResponse{}, raw, err
	}
	var response SearchResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return status, SearchResponse{}, raw, fmt.Errorf("moonshottools: parsing search_pro response: %w", err)
	}
	return status, response, raw, nil
}

// Fetch fetches a page and returns the parsed response.
func (c *Client) Fetch(ctx context.Context, req FetchRequest) (int, FetchResponse, []byte, error) {
	status, raw, err := c.do(ctx, FetchPath, req)
	if err != nil {
		return status, FetchResponse{}, raw, err
	}
	var response FetchResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return status, FetchResponse{}, raw, fmt.Errorf("moonshottools: parsing fetch response: %w", err)
	}
	return status, response, raw, nil
}

// logCall records a finished call: the endpoint, the status, how long it
// took and how big the answer was.
func (c *Client) logCall(ctx context.Context, path string, status int, took time.Duration, size int) {
	if c.logger == nil {
		return
	}
	c.logger.DebugContext(ctx, "Moonshot tool call",
		"path", path,
		"status", status,
		"duration_ms", took.Milliseconds(),
		"response_bytes", size,
	)
}

// logError records a call that never produced a response.
func (c *Client) logError(ctx context.Context, msg, path string, status int, took time.Duration, err error) {
	if c.logger == nil {
		return
	}
	c.logger.WarnContext(ctx, msg,
		"path", path,
		"status", status,
		"duration_ms", took.Milliseconds(),
		"error", err,
	)
}
