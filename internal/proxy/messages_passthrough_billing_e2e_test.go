package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_MessagesPassthroughStillLogsUsageAndCost is an end-to-end guard that the
// /v1/messages native-passthrough path (added so a real Anthropic Messages API request no
// longer round-trips through Chat Completions shape) did not lose token/cost accounting.
// The response side (ResponseTo->ChatToMessages, usage extraction) was never touched by the
// passthrough change — this proves it by driving a full ProxyRequest call against a real
// Anthropic-shaped upstream response and asserting the logged spend entry.
func TestProxyRequest_MessagesPassthroughStillLogsUsageAndCost(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/messages", r.URL.Path)
		require.Equal(t, "upstream-key", r.Header.Get("x-api-key"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_e2e","type":"message","role":"assistant","model":"claude-opus-4-7",
			"content":[{"type":"text","text":"hi there"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":100,"output_tokens":40,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}
		}`))
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

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"claude-e2e": {
			InputCostPerToken:           1,
			OutputCostPerToken:          2,
			CacheReadInputTokenCost:     0.5,
			CacheCreationInputTokenCost: 3,
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-e2e",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	// Client still gets an Anthropic Messages-shaped response (ChatToMessages ran).
	var clientBody map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &clientBody))
	assert.Equal(t, "message", clientBody["type"])
	usage, ok := clientBody["usage"].(map[string]interface{})
	require.True(t, ok, "client response must carry usage")
	assert.NotZero(t, usage["input_tokens"])
	assert.NotZero(t, usage["output_tokens"])

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, "success", entry.Status)
	// input_tokens(100) is exclusive of cache per Anthropic's usage shape; the inclusive
	// prompt_tokens CalculateTokenCosts expects adds the cache figures back: 100+10+5=115.
	assert.Equal(t, 115, entry.PromptTokens)
	assert.Equal(t, 40, entry.CompletionTokens)
	assert.NotZero(t, entry.Spend, "passthrough request must still be billed, not silently free")
}

// TestProxyRequest_MessagesPassthroughStreamingStillLogsUsageAndCost is the streaming
// counterpart — this is the shape real Claude Code traffic actually uses (stream:true) and
// exercises handleMessagesAPIStreaming's separate StreamTo->TransformChatStreamToMessages
// pipeline instead of the non-streaming ResponseTo->ChatToMessages one.
func TestProxyRequest_MessagesPassthroughStreamingStillLogsUsageAndCost(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/messages", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_e2e","type":"message","role":"assistant","model":"claude-opus-4-7","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}}}

`,
			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi there"}}

`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":0}

`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":40}}

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

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"claude-e2e": {
			InputCostPerToken:           1,
			OutputCostPerToken:          2,
			CacheReadInputTokenCost:     0.5,
			CacheCreationInputTokenCost: 3,
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-e2e",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "message_start")
	assert.Contains(t, w.Body.String(), "hi there")

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, "success", entry.Status)
	assert.Equal(t, 115, entry.PromptTokens)
	assert.Equal(t, 40, entry.CompletionTokens)
	assert.NotZero(t, entry.Spend, "streaming passthrough request must still be billed")
}
