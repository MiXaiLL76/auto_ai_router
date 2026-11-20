package proxy

import (
	"context"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// referenceSSEDataPayloads is how the passthrough Responses handlers used to
// read a whole stream: the data: payloads of its complete lines.
func referenceSSEDataPayloads(stream string) []string {
	lastNL := strings.LastIndex(stream, "\n")
	if lastNL < 0 {
		return nil
	}
	var payloads []string
	for _, line := range strings.Split(stream[:lastNL], "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		if payload := strings.TrimSpace(strings.TrimPrefix(line, "data:")); payload != "" && payload != "[DONE]" {
			payloads = append(payloads, payload)
		}
	}
	return payloads
}

func TestResponsesStreamLines_MatchWholeStreamSplit(t *testing.T) {
	long := strings.Repeat("x", 20_000)
	streams := []string{
		"",
		"data: unterminated",
		"\n\n\n",
		"data: {\"a\":1}\n\ndata: [DONE]\n\n",
		"data:{\"no\":\"space\"}\n\ndata:   \n\n",
		"event: e\r\ndata: {\"a\":1}\r\n\r\n",
		"data: x\r\r\n: comment\ndata: \t{\"b\":2} \r\nid: 3\ndata: tail without newline",
		"data: " + long + "\n\ndata: " + long + "\r\n\r\ndata: unterminated " + long,
	}
	rng := rand.New(rand.NewSource(1))
	for _, stream := range streams {
		want := referenceSSEDataPayloads(stream)
		sizes := []int{0, 1, 2, 3, 7, 8192} // 0: the whole stream in one chunk
		for i := 0; i < 20; i++ {
			sizes = append(sizes, -1) // random chunk sizes
		}
		for _, size := range sizes {
			lines := newResponsesStreamLines()
			var got []string
			data := []byte(stream)
			for len(data) > 0 {
				n := len(data)
				switch {
				case size > 0:
					n = min(size, len(data))
				case size < 0:
					n = min(1+rng.Intn(64), len(data))
				}
				lines.Observe(data[:n], func(line []byte) {
					if payload := responsesStreamDataPayload(line); payload != nil {
						got = append(got, string(payload))
					}
				})
				data = data[n:]
			}
			assert.Equal(t, want, got, "stream %.40q chunk size %d", stream, size)
		}
	}
}

func TestParseResponsesTerminalEvent_OnlyTerminalTypes(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    bool
	}{
		{"completed", `{"type":"response.completed","response":{"status":"completed"}}`, true},
		{"incomplete", `{"type":"response.incomplete","response":{"status":"incomplete"}}`, true},
		{"type after response", `{"response":{"status":"completed"},"type":"response.completed"}`, true},
		{"delta quoting the type", `{"type":"response.output_text.delta","delta":"the \"response.completed\" event"}`, false},
		{"created", `{"type":"response.created","response":{"status":"in_progress"}}`, false},
		{"failed", `{"type":"response.failed","response":{"status":"failed"}}`, false},
		{"truncated JSON", `{"type":"response.completed","response":{"usage":{"input_tokens":5`, false},
		{"type not a string", `{"type":7,"note":"response.completed"}`, false},
		{"response not an object", `{"type":"response.completed","response":"x"}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := parseResponsesTerminalEvent([]byte(tt.payload))
			assert.Equal(t, tt.want, ok)
		})
	}
}

// TestResponsesTerminalEvent_MatchesPayloadExtractors pins the single decode
// of a terminal event to what finalizeStreamingLog used to compute by
// decoding the payload again, and its typed usage to the usage of the whole
// response decode the handlers used to run on every event.
func TestResponsesTerminalEvent_MatchesPayloadExtractors(t *testing.T) {
	const fullUsage = `{"input_tokens":7168,"output_tokens":717,"total_tokens":8185,` +
		`"input_tokens_details":{"cached_tokens":1024,"cached_audio_tokens":24,"cache_creation_tokens":100,` +
		`"cache_creation_token_details":{"ephemeral_5m_input_tokens":60,"ephemeral_1h_input_tokens":40},` +
		`"audio_tokens":50,"image_tokens":300,"video_tokens":20,"cache_type":"ephemeral"},` +
		`"output_tokens_details":{"reasoning_tokens":300,"audio_tokens":10,"image_tokens":5,"cached_tokens":3,` +
		`"accepted_prediction_tokens":4,"rejected_prediction_tokens":2},` +
		`"server_tool_use":{"web_search_requests":2},` +
		`"server_side_tool_usage_details":{"web_search_calls":2,"x_search_calls":1}}`
	webItems := `{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"web_search_call","id":"ws_2","status":"failed"},` +
		`{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"x"}]}`
	payloads := map[string]string{
		"full usage":            `{"type":"response.completed","response":{"id":"r","status":"completed","output":[],"usage":` + fullUsage + `}}`,
		"incomplete":            `{"type":"response.incomplete","response":{"status":"incomplete","usage":{"input_tokens":3,"output_tokens":1}}}`,
		"cache write only":      `{"type":"response.completed","response":{"usage":{"input_tokens":500,"output_tokens":20,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":77}}}}`,
		"web search items":      `{"type":"response.completed","response":{"output":[` + webItems + `],"usage":{"input_tokens":100,"output_tokens":10}}}`,
		"xai zero web search":   `{"type":"response.completed","response":{"output":[` + webItems + `],"usage":{"input_tokens":100,"output_tokens":10,"server_side_tool_usage_details":{"web_search_calls":0}}}}`,
		"top-level web items":   `{"type":"response.completed","output":[` + webItems + `],"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"usage null":            `{"type":"response.completed","response":{"output":[],"usage":null}}`,
		"usage absent":          `{"type":"response.completed","response":{"output":[]}}`,
		"response null":         `{"type":"response.completed","response":null}`,
		"top-level usage only":  `{"type":"response.completed","usage":{"input_tokens":21,"output_tokens":4},"response":{"output":[]}}`,
		"top-level chat usage":  `{"type":"response.completed","usage":{"prompt_tokens":11,"completion_tokens":6},"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"top-level chat zeros":  `{"type":"response.completed","usage":{"prompt_tokens":0,"completion_tokens":0},"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"usage type mismatch":   `{"type":"response.completed","response":{"usage":{"input_tokens":"10","output_tokens":5}}}`,
		"top usage mismatch":    `{"type":"response.completed","usage":"x","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"top output mismatch":   `{"type":"response.completed","output":"x","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"explicit zero usage":   `{"type":"response.completed","response":{"usage":{"input_tokens":0,"output_tokens":0}}}`,
		"top-level error":       `{"type":"response.completed","error":{"message":"boom"},"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"top-level error null":  `{"type":"response.completed","error":null,"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		"audio cached in usage": `{"type":"response.completed","response":{"usage":{"input_tokens":90,"output_tokens":9,"input_tokens_details":{"cached_tokens":30,"cached_audio_tokens":10,"audio_tokens":40}}}}`,
	}
	for name, payload := range payloads {
		t.Run(name, func(t *testing.T) {
			require.NotContains(t, payload, "data:", "the payload extractor treats data: as SSE framing")
			event, ok := parseResponsesTerminalEvent([]byte(payload))
			require.True(t, ok)
			for _, alreadyExcludesCachedAudio := range []bool{false, true} {
				extractor := &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: alreadyExcludesCachedAudio}
				parsed := event.finalChunkParse([]byte(payload), alreadyExcludesCachedAudio)
				assert.Equal(t, extractor.ExtractUsage([]byte(payload)), parsed.usage, "usage, cached audio excluded=%v", alreadyExcludesCachedAudio)
				assert.Equal(t, extractStreamErrorEvent([]byte(payload)), parsed.errorEvent)
			}

			var whole struct {
				Response responses.Response `json:"response"`
			}
			if json.Unmarshal([]byte(payload), &whole) == nil {
				assert.Equal(t, whole.Response.Usage, event.typedUsage())
			}
		})
	}
}

// passthroughResponsesStreamProxy builds a proxy whose OpenAI credential
// answers /v1/responses with sse, as a codex passthrough upstream does.
func passthroughResponsesStreamProxy(t *testing.T, sse string, price *pricing.ModelPrice) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name: "openai-terminal", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "upstream-key",
			RPM: 100, TPM: 1_000_000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-terminal", price)
	return prx, dbStub
}

func postResponsesStream(t *testing.T, prx *Proxy, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// TestProxyRequest_ResponsesStreamDataColonInAnswerKeepsUsageDetails: the
// payload extractor finalizeStreamingLog used to run on the terminal event
// took a payload holding "data:" (here a data URL in the answer) for SSE
// framing and found no usage, so the details only it reads (cache writes,
// input image tokens) went unbilled.
func TestProxyRequest_ResponsesStreamDataColonInAnswerKeepsUsageDetails(t *testing.T) {
	sse := "event: response.completed\ndata: " +
		`{"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-terminal",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"see data:image/png;base64,AAAA"}]}],` +
		`"usage":{"input_tokens":500,"output_tokens":20,"total_tokens":520,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":77,"image_tokens":90}}}}` +
		"\n\ndata: [DONE]\n\n"
	prx, dbStub := passthroughResponsesStreamProxy(t, sse, &pricing.ModelPrice{
		InputCostPerToken: 1, OutputCostPerToken: 2, InputCostPerImageToken: 3, CacheCreationInputTokenCost: 1.5,
	})

	w := postResponsesStream(t, prx, `{"model":"gpt-terminal","input":"hi","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 500, entry.PromptTokens)
	// (500-77-90) text x1 + 77 cache writes x1.5 + 90 image tokens x3 + 20 output x2
	assert.InDelta(t, 333+115.5+270+40, entry.Spend, 1e-9)
}

// TestProxyRequest_ResponsesStreamTerminalEventFeedsRateLimiterWithoutTypedResponse:
// a terminal event whose response does not decode into the typed Response
// (metadata values must be strings) used to be skipped whole, so the rate
// limiter consumed a local estimate instead of the provider's total_tokens.
func TestProxyRequest_ResponsesStreamTerminalEventFeedsRateLimiterWithoutTypedResponse(t *testing.T) {
	sse := "event: response.completed\ndata: " +
		`{"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-terminal",` +
		`"metadata":{"attempt":1},"output":[],"usage":{"input_tokens":700,"output_tokens":42,"total_tokens":742}}}` +
		"\n\ndata: [DONE]\n\n"
	prx, dbStub := passthroughResponsesStreamProxy(t, sse, &pricing.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})

	w := postResponsesStream(t, prx, `{"model":"gpt-terminal","input":"hi","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 742, prx.rateLimiter.GetCurrentTPM("openai-terminal"))
	require.Len(t, dbStub.loggedEntries, 1)
	assert.Equal(t, 700, dbStub.loggedEntries[0].PromptTokens)
	assert.Equal(t, 42, dbStub.loggedEntries[0].CompletionTokens)
}

type terminalEventResponseStore struct {
	fakeResponseStore
	saved []*responses.Response
}

func (s *terminalEventResponseStore) SaveResponse(_ context.Context, _ string, resp *responses.Response, _ map[string]string, _ int, _ json.RawMessage, _ string) error {
	s.saved = append(s.saved, resp)
	return nil
}

// TestProxyRequest_ResponsesStreamStoreSavesTerminalResponse: with store=true
// the terminal event's whole response, here spanning several reads, is still
// decoded and saved.
func TestProxyRequest_ResponsesStreamStoreSavesTerminalResponse(t *testing.T) {
	reasoning := strings.Repeat("E", 64*1024)
	sse := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n" +
		"event: response.completed\ndata: " +
		`{"type":"response.completed","response":{"id":"resp_stored","object":"response","status":"completed","model":"gpt-terminal",` +
		`"output":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"` + reasoning + `"},` +
		`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}],` +
		`"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}}}` +
		"\n\ndata: [DONE]\n\n"
	prx, dbStub := passthroughResponsesStreamProxy(t, sse, &pricing.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	store := &terminalEventResponseStore{}
	prx.responseStore = store

	w := postResponsesStream(t, prx, `{"model":"gpt-terminal","input":"hi","stream":true,"store":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, store.saved, 1)
	saved := store.saved[0]
	assert.Equal(t, "resp_stored", saved.ID)
	assert.Equal(t, "completed", saved.Status)
	require.Len(t, saved.Output, 2)
	require.NotNil(t, saved.Usage)
	assert.Equal(t, 15, saved.Usage.TotalTokens)
	assert.Equal(t, 15, prx.rateLimiter.GetCurrentTPM("openai-terminal"))
	require.Len(t, dbStub.loggedEntries, 1)
	assert.Equal(t, 12, dbStub.loggedEntries[0].PromptTokens)
}
