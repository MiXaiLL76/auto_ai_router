package proxy

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_SosanaLiveAcceptance(t *testing.T) {
	if os.Getenv("SOSANA_ACCEPTANCE") != "1" {
		t.Skip("SOSANA_ACCEPTANCE=1 not set, skipping paid Sosana live acceptance test")
	}

	apiKey := os.Getenv("SOSANA_API_KEY")
	require.NotEmpty(t, apiKey, "SOSANA_API_KEY is required for Sosana live acceptance test")

	baseURL := os.Getenv("SOSANA_BASE_URL")
	if baseURL == "" {
		baseURL = "https://sosana.art"
	}
	model := os.Getenv("SOSANA_MODEL")
	if model == "" {
		model = "banana-2-1k-compliant"
	}
	prompt := os.Getenv("SOSANA_PROMPT")
	if prompt == "" {
		prompt = "a small blue cube on a white background"
	}

	prx := NewTestProxyBuilder().
		WithSingleCredential("sosana-live", config.ProviderTypeSosana, baseURL, apiKey).
		WithRequestTimeout(2 * time.Minute).
		Build()

	body, err := json.Marshal(map[string]any{
		"model":  model,
		"prompt": prompt,
		"size":   "1024x1024",
		"n":      1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "response body: %s", w.Body.String())
	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 1)
	require.Empty(t, resp.Data[0].URL)
	require.NotEmpty(t, resp.Data[0].B64JSON)
	_, err = base64.StdEncoding.DecodeString(resp.Data[0].B64JSON)
	require.NoError(t, err)
}
