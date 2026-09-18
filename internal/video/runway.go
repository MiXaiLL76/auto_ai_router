package video

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type RunwayConfig struct {
	BaseURL    string
	APIVersion string
	APIKey     string
	Models     map[string]string
	HTTPClient *http.Client
}

type RunwayClient struct {
	baseURL string
	version string
	key     string
	models  map[string]string
	http    *http.Client
}

func NewRunwayClient(cfg RunwayConfig) (*RunwayClient, error) {
	u, err := url.Parse(strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/"))
	if err != nil || !secureEndpoint(u) {
		return nil, ErrInvalid
	}
	if cfg.APIKey == "" {
		return nil, ErrInvalid
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = "2024-11-06"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	} else {
		client := *cfg.HTTPClient
		if client.Timeout <= 0 || client.Timeout > 30*time.Second {
			client.Timeout = 30 * time.Second
		}
		cfg.HTTPClient = &client
	}
	if len(cfg.Models) == 0 {
		cfg.Models = map[string]string{"runway/gen4.5": "gen4.5", "runway/gen4_turbo": "gen4_turbo"}
	}
	models := make(map[string]string, len(cfg.Models))
	for public, provider := range cfg.Models {
		if strings.TrimSpace(public) == "" || strings.TrimSpace(provider) == "" {
			return nil, ErrInvalid
		}
		models[public] = provider
	}
	return &RunwayClient{baseURL: u.String(), version: cfg.APIVersion, key: cfg.APIKey, models: models, http: cfg.HTTPClient}, nil
}

func (c *RunwayClient) Submit(ctx context.Context, j *Job, promptImage string) (string, error) {
	if c == nil || j == nil {
		return "", ErrInvalid
	}
	model := c.models[j.Request.Model]
	if model == "" {
		return "", ErrUnsupportedModel
	}
	ratio := runwayRatio(j.Request.AspectRatio)
	if ratio == "" {
		ratio = runwayRatioFromSize(j.Request.Size)
	}
	payload := map[string]any{"model": model, "promptText": j.Request.Prompt, "ratio": ratio, "duration": j.Request.DurationSeconds, "watermark": false}
	if j.Request.Seed != nil {
		payload["seed"] = *j.Request.Seed
	}
	path := "/v1/text_to_video"
	if promptImage != "" {
		path = "/v1/image_to_video"
		payload["promptImage"] = promptImage
	}
	var out map[string]any
	if err := c.do(ctx, http.MethodPost, path, payload, j.ID, &out); err != nil {
		return "", err
	}
	id := findText(out, "id", "taskId", "task_id")
	if id == "" {
		return "", fmt.Errorf("runway response has no task id")
	}
	return id, nil
}

func runwayRatio(value string) string {
	switch strings.TrimSpace(value) {
	case "16:9":
		return "1280:720"
	case "9:16":
		return "720:1280"
	case "1:1":
		return "960:960"
	case "4:3":
		return "1104:832"
	case "3:4":
		return "832:1104"
	default:
		return strings.TrimSpace(value)
	}
}

func runwayRatioFromSize(value string) string {
	switch strings.TrimSpace(value) {
	case "720p", "1080p", "1280x720", "1920x1080", "1792x1024":
		return "1280:720"
	case "720x1280", "1080x1920", "1024x1792":
		return "720:1280"
	default:
		return ""
	}
}

func (c *RunwayClient) Poll(ctx context.Context, id string) (ProviderResult, error) {
	if id == "" {
		return ProviderResult{}, ErrInvalid
	}
	var out map[string]any
	if err := c.do(ctx, http.MethodGet, "/v1/tasks/"+url.PathEscape(id), nil, "", &out); err != nil {
		return ProviderResult{}, err
	}
	state := strings.ToUpper(findText(out, "status", "state"))
	r := ProviderResult{ID: id}
	switch state {
	case "PENDING", "THROTTLED":
		r.Status = StatusQueued
	case "RUNNING":
		r.Status = StatusInProgress
	case "SUCCEEDED":
		r.Status = StatusCompleted
		r.ResultURL = findText(out, "output", "url", "video_url")
		if r.ResultURL == "" {
			return ProviderResult{}, fmt.Errorf("runway success has no output")
		}
	case "FAILED", "SAFETY", "INTERNAL":
		r.Status = StatusFailed
		r.ErrorCode = "provider_failed"
		r.ErrorMessage = findText(out, "failure", "message")
	case "CANCELED", "CANCELLED":
		r.Status = StatusCancelled
	default:
		return ProviderResult{}, fmt.Errorf("unknown runway state %q", state)
	}
	return r, nil
}

func (c *RunwayClient) Cancel(ctx context.Context, id string) error {
	if id == "" {
		return ErrInvalid
	}
	return c.do(ctx, http.MethodDelete, "/v1/tasks/"+url.PathEscape(id), nil, "", nil)
}

func (c *RunwayClient) do(ctx context.Context, method, path string, body any, idem string, out any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader) //nolint:gosec // base URL and paths are validated configuration
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("X-Runway-Version", c.version)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if idem != "" {
		req.Header.Set("Idempotency-Key", idem)
	}
	resp, err := c.http.Do(req) //nolint:gosec // request target is the validated Runway base URL
	if err != nil {
		return fmt.Errorf("%w: %v", ErrProviderUnavailable, err)
	}
	defer func() { _ = resp.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if method == http.MethodDelete && resp.StatusCode == http.StatusNotFound {
			return nil
		}
		return fmt.Errorf("runway status %d", resp.StatusCode)
	}
	if len(raw) == 0 || out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("invalid runway response: %w", err)
	}
	return nil
}

func findText(v any, keys ...string) string {
	switch value := v.(type) {
	case map[string]any:
		for _, k := range keys {
			if x, ok := value[k]; ok {
				switch t := x.(type) {
				case string:
					if t != "" {
						return t
					}
				case []any:
					for _, item := range t {
						if s, ok := item.(string); ok && s != "" {
							return s
						}
					}
				}
			}
		}
		for _, x := range value {
			if s := findText(x, keys...); s != "" {
				return s
			}
		}
	case []any:
		for _, x := range value {
			if s := findText(x, keys...); s != "" {
				return s
			}
		}
	}
	return ""
}
