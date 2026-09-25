package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_LiteLLMCompatKeepsResponsesEndpointForConvertedRequests(t *testing.T) {
	const usage = `{"prompt_tokens":37,"completion_tokens":59,"total_tokens":96,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":57}}`

	for _, tt := range []struct {
		name   string
		stream bool
	}{
		{name: "non-stream", stream: false},
		{name: "stream", stream: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/v1/chat/completions", r.URL.Path)
				body, _ := io.ReadAll(r.Body)
				assert.Contains(t, string(body), `"messages"`)
				if tt.stream {
					w.Header().Set("Content-Type", "text/event-stream")
					for _, chunk := range []string{
						`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}`,
						`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":` + usage + `}`,
					} {
						_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
					}
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","created":1,"model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":` + usage + `}`))
			}))
			defer upstream.Close()

			passthroughResponses := false
			builder := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "deepseek-direct",
					Type:    config.ProviderTypeOpenAI,
					BaseURL: upstream.URL + "/v1",
					APIKey:  "upstream-key",
					RPM:     100,
					TPM:     -1,
				}).
				WithMasterKey("master-key")
			builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
				{Name: "deepseek-flash", Credential: "deepseek-direct", PassthroughResponses: &passthroughResponses},
			})
			builder.config.ModelManager.LoadModelsFromConfig(builder.config.Credentials)
			prx := builder.Build()
			prx.LiteLLMDB = &stubLiteLLMManager{}
			registry := pricing.NewModelPriceRegistry()
			registry.Update(map[string]*pricing.ModelPrice{
				"deepseek-flash": {InputCostPerToken: 1, OutputCostPerToken: 2},
			})
			prx.priceRegistry = registry
			prx.responseCompat = compatlitellm.New()

			requestBody := `{"model":"deepseek-flash","input":"hello","max_output_tokens":200}`
			if tt.stream {
				requestBody = `{"model":"deepseek-flash","input":"hello","max_output_tokens":200,"stream":true}`
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(requestBody))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			if tt.stream {
				assert.Contains(t, w.Body.String(), `"type":"response.created"`)
				assert.Contains(t, w.Body.String(), `"type":"response.output_text.delta"`)
				assert.Contains(t, w.Body.String(), `"type":"response.completed"`)
				assert.NotContains(t, w.Body.String(), `"object":"chat.completion.chunk"`)
				return
			}
			var response struct {
				Object string `json:"object"`
				Output []struct {
					Content []struct {
						Text string `json:"text"`
					} `json:"content"`
				} `json:"output"`
			}
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response), w.Body.String())
			assert.Equal(t, "response", response.Object)
			require.Len(t, response.Output, 1)
			require.Len(t, response.Output[0].Content, 1)
			assert.Equal(t, "ok", response.Output[0].Content[0].Text)
		})
	}
}
