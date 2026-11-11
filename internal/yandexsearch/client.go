// Package yandexsearch implements a thin, authenticated passthrough from the
// router's /v1/search/yandex endpoint to Yandex Cloud's Search API
// (POST {base_url}/v2/web/search). The router holds the Yandex API key and
// Folder ID; callers never see or supply them, and their own inbound
// Authorization/identifying headers are never forwarded upstream.
package yandexsearch

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

const searchPath = "/v2/web/search"

const folderIDField = "folderId"

// ClientConfig configures a Client.
type ClientConfig struct {
	BaseURL  string
	APIKey   string
	FolderID string
	Timeout  time.Duration
	// ProxyURL, when set, routes the router's own outbound connection to
	// Yandex through an upstream HTTP/HTTPS proxy — optional, has nothing
	// to do with client auth or folderId.
	ProxyURL string
}

// Client forwards web-search requests to Yandex Cloud's Search API.
type Client struct {
	baseURL  string
	apiKey   string
	folderID string
	http     *http.Client
}

// NewClient builds a Client over the shared HTTP transport (connection
// pooling, HTTP/2 keepalive) so this feature doesn't hand-roll its own.
func NewClient(cfg ClientConfig) *Client {
	httpCfg := httputil.DefaultHTTPClientConfig()
	httpCfg.Timeout = cfg.Timeout
	client := httputil.NewHTTPClient(httpCfg)
	if cfg.ProxyURL != "" {
		client = httputil.WithClientProxy(client, cfg.ProxyURL)
	}
	return &Client{
		baseURL:  strings.TrimSuffix(cfg.BaseURL, "/"),
		apiKey:   cfg.APIKey,
		folderID: cfg.FolderID,
		http:     client,
	}
}

// ErrInvalidBody is returned when the client's request body is not a JSON
// object.
var ErrInvalidBody = fmt.Errorf("yandexsearch: request body must be a JSON object")

// Search forwards body (the client's own request payload, verbatim except
// for folderId) to Yandex and returns Yandex's status code and response
// body for the caller to relay. It never returns an error for a non-2xx
// Yandex response — that's a valid passthrough result, not a client error.
func (c *Client) Search(ctx context.Context, body []byte) (status int, respBody []byte, err error) {
	outBody, err := withFolderID(body, c.folderID)
	if err != nil {
		return 0, nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+searchPath, bytes.NewReader(outBody))
	if err != nil {
		return 0, nil, fmt.Errorf("yandexsearch: building upstream request: %w", err)
	}
	// Built from scratch: never copies headers from the inbound client
	// request, so the client's own Authorization/X-Forwarded-For/etc. never
	// reach Yandex.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Api-Key "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("yandexsearch: upstream request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err = io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("yandexsearch: reading upstream response: %w", err)
	}
	return resp.StatusCode, respBody, nil
}

// withFolderID parses body just enough to set/overwrite the top-level
// "folderId" field, leaving every other field's raw bytes untouched, then
// re-marshals. Yandex's folderId is a server-held value the client never
// supplies (or, if it tries to, is overridden).
func withFolderID(body []byte, folderID string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, ErrInvalidBody
	}
	if fields == nil {
		fields = map[string]json.RawMessage{}
	}
	encodedFolderID, err := json.Marshal(folderID)
	if err != nil {
		return nil, fmt.Errorf("yandexsearch: encoding folder_id: %w", err)
	}
	fields[folderIDField] = encodedFolderID
	return json.Marshal(fields)
}
