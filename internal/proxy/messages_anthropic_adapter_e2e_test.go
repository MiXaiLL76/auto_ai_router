package proxy

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// anthropicWebSearchStream: one server web search (query streamed as
// input_json_delta of server_tool_use) and a 1h cache write.
const anthropicWebSearchStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_ws","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":50,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":50}}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_01A","name":"web_search","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"tallest building oslo\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_01A","content":[{"type":"web_search_result","url":"https://example.com","title":"t","encrypted_content":"x"}]}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Oslo's tallest building is the Radisson Blu Plaza."}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":40,"server_tool_use":{"web_search_requests":1}}}

event: message_stop
data: {"type":"message_stop"}

`

func newAnthropicTest(t *testing.T, contentType, response string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	upstream := serveUpstream(t, "/v1/messages", contentType, response)
	t.Cleanup(upstream.Close)
	dbStub := &stubLiteLLMManager{}
	return newBillingTestProxy(t, config.ProviderTypeAnthropic, upstream.URL, dbStub), dbStub
}

// A /v1/messages stream is billed from the re-emitted Messages events, so the
// search counter and the TTL split must survive Anthropic -> Chat -> Messages.
func TestProxyRequest_MessagesStreamKeepsServerSearchAndCacheTTL(t *testing.T) {
	prx, dbStub := newAnthropicTest(t, "text/event-stream", anthropicWebSearchStream)

	w := postAsMaster(t, prx, "/v1/messages", `{"model":"claude-test","max_tokens":100,"stream":true,
		"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":1}],
		"messages":[{"role":"user","content":"tallest building in Oslo?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	body := w.Body.String()
	assert.Contains(t, body, "Radisson Blu Plaza")
	assert.NotContains(t, body, `"tool_use"`, "a server search must not reach the client as a client tool call")
	assert.NotContains(t, body, "tallest building oslo")
	assert.Contains(t, compactJSONFragments(t, body), `"server_tool_use":{"web_search_requests":1}`)
	assert.Contains(t, compactJSONFragments(t, body), `"cache_creation":{"ephemeral_1h_input_tokens":50}`)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 150, entry.PromptTokens)
	assert.Equal(t, 40, entry.CompletionTokens)
	metadata := decodeMetadata(t, entry.Metadata)
	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(50*5), costBreakdown["cache_creation_cost"])
	assert.Equal(t, float64(7), costBreakdown["web_search_cost"])
	assert.Equal(t, float64(100+50*5+40*2+7), entry.Spend)
}

// The same stream through /v1/chat/completions: no tool_calls from the search.
func TestProxyRequest_ChatStreamDoesNotSurfaceServerSearchAsToolCall(t *testing.T) {
	prx, dbStub := newAnthropicTest(t, "text/event-stream", anthropicWebSearchStream)

	w := postAsMaster(t, prx, "/v1/chat/completions", `{"model":"claude-test","stream":true,
		"stream_options":{"include_usage":true},"tools":[{"type":"web_search"}],
		"messages":[{"role":"user","content":"tallest building in Oslo?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	body := w.Body.String()
	assert.Contains(t, body, "Radisson Blu Plaza")
	assert.NotContains(t, body, "tool_calls")
	assert.NotContains(t, body, "tallest building oslo")

	require.Len(t, dbStub.loggedEntries, 1)
	metadata := decodeMetadata(t, dbStub.loggedEntries[0].Metadata)
	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(7), costBreakdown["web_search_cost"])
}

// Clients echo the content back; an empty text block gets a 400 from Anthropic.
func TestProxyRequest_MessagesToolUseTurnHasNoEmptyTextBlock(t *testing.T) {
	prx, _ := newAnthropicTest(t, "application/json", `{
		"id":"msg_tool","type":"message","role":"assistant","model":"claude-opus-5-5",
		"content":[{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{"city":"Paris"}}],
		"stop_reason":"tool_use",
		"usage":{"input_tokens":100,"output_tokens":20}
	}`)

	w := postAsMaster(t, prx, "/v1/messages", `{"model":"claude-test","max_tokens":100,
		"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
		"messages":[{"role":"user","content":"weather in Paris?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var response struct {
		StopReason string                   `json:"stop_reason"`
		Content    []map[string]interface{} `json:"content"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, "tool_use", response.StopReason)
	require.Len(t, response.Content, 1)
	assert.Equal(t, "tool_use", response.Content[0]["type"])
	assert.Equal(t, "toolu_01A", response.Content[0]["id"])
}
