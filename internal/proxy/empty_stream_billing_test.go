package proxy

import (
	"encoding/json"
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

// TestProxyRequest_EmptyMessagesStreamBillsNoOutput is the regression guard for the
// overcharge described in issue #1538: a provider answering 200 with a stream that
// carries no content delta and usage.output_tokens=0 used to be logged with
// completion_tokens equal to the whole request (prompt + cache), because
// finalizeStreamingLog fell back to the stream's request-shaped total_tokens for a
// missing completion count. Output is priced ~50x a cache read, so those rows became
// the most expensive kind of row in air.logs while nothing was actually generated.
func TestProxyRequest_EmptyMessagesStreamBillsNoOutput(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/messages", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		// A well-formed stream that never produces a token: message_start reports
		// the input, message_delta reports output_tokens 0, message_stop carries no
		// usage at all (so lastChunk has none — the provider's figures are only in
		// what was merged into logCtx.TokenUsage along the way).
		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_empty","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1000,"output_tokens":1,"cache_read_input_tokens":100,"cache_creation_input_tokens":50}}}

`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}

`,
			`event: message_stop
data: {"type":"message_stop"}

`,
		}
		for _, e := range events {
			_, _ = w.Write([]byte(e))
		}
	}))
	defer upstream.Close()

	cred := config.CredentialConfig{
		Name: "anthropic", Type: config.ProviderTypeAnthropic,
		BaseURL: upstream.URL, APIKey: "upstream-key", RPM: 100, TPM: 10000,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(cred).
		WithMasterKey("master-key").
		Build()

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	prx.responseCompat = compatlitellm.New()

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"claude-empty": {
			InputCostPerToken:           1,
			OutputCostPerToken:          2,
			CacheReadInputTokenCost:     0.5,
			CacheCreationInputTokenCost: 3,
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-empty",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]

	// The request is still billed for what it consumed: 1000 uncached + 100 cache
	// read + 50 cache write.
	assert.Equal(t, 1150, entry.PromptTokens)
	// ...but nothing was generated, so nothing may be billed as output.
	assert.Zero(t, entry.CompletionTokens,
		"an empty answer must not be billed as if it produced the whole prompt")
	assert.Equal(t, 1150, entry.TotalTokens)
	// input(1000)*1 + cache_read(100)*0.5 + cache_write(50)*3
	assert.Equal(t, float64(1200), entry.Spend)
}

// TestProxyRequest_EmptyChatStreamBillsNoOutput is the Chat Completions twin of the test
// above: the same signature (completion_tokens == prompt_tokens) also shows up on
// /v1/chat/completions rows, so the fix has to hold there too. The usage chunk is
// deliberately not the last data chunk — which is how those rows arise: lastChunk-based
// usage extraction sees a usage-less trailing chunk, and the provider's request-shaped
// total_tokens was the only number left to bill as output.
func TestProxyRequest_EmptyChatStreamBillsNoOutput(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-empty\",\"model\":\"test-empty\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":0,\"total_tokens\":900}}\n\n" +
				"data: {\"id\":\"chat-empty\",\"model\":\"test-empty\",\"choices\":[]}\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "openai", Type: config.ProviderTypeOpenAI,
		BaseURL: upstream.URL, APIKey: "upstream-key", RPM: 100,
	}).WithMasterKey("master-key").Build()
	prx.responseCompat = compatlitellm.New()

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"test-empty": {InputCostPerToken: 1, OutputCostPerToken: 2},
	})
	prx.priceRegistry = registry

	body, err := json.Marshal(map[string]any{
		"model": "test-empty", "max_tokens": 32, "stream": true,
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
	})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 900, entry.PromptTokens)
	assert.Zero(t, entry.CompletionTokens,
		"an empty chat answer must not be billed as if it produced the whole prompt")
	assert.Equal(t, float64(900), entry.Spend)
}
