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

// chatStreamCreditTypes are the credential kinds whose chat streams reach the spend
// writer through different code paths: a direct provider finalises through
// finalizeStreamingLog, a proxy-like one (ProviderTypeProxy/AIR) has
// writeProxyStreamingResponseWithTokens write the row itself. Both have to honour
// the same provider-report rules, so the usage cases below run against both.
var chatStreamCreditTypes = []config.ProviderType{config.ProviderTypeOpenAI, config.ProviderTypeProxy, config.ProviderTypeAIR}

// newChatStreamTestProxy builds a proxy serving the given credential, with the spend
// stub and a price for model attached.
func newChatStreamTestProxy(t *testing.T, upstreamURL string, credType config.ProviderType, model string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: string(credType), Type: credType,
		BaseURL: upstreamURL, APIKey: "upstream-key", RPM: 100,
	}).WithMasterKey("master-key").Build()
	prx.responseCompat = compatlitellm.New()

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		model: {InputCostPerToken: 1, OutputCostPerToken: 2},
	})
	prx.priceRegistry = registry
	return prx, dbStub
}

// postChatStream sends a streaming chat completion for model and returns the recorder.
func postChatStream(t *testing.T, prx *Proxy, model string) *httptest.ResponseRecorder {
	t.Helper()
	return postChatStreamPrompt(t, prx, model, "hello")
}

// postChatStreamPrompt is postChatStream with an explicit request text, for cases where the
// prompt-token estimate itself is what the test reads.
func postChatStreamPrompt(t *testing.T, prx *Proxy, model, content string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model": model, "max_tokens": 32, "stream": true,
		"messages": []map[string]string{{"role": "user", "content": content}},
	})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// TestProxyRequest_PartialUsageStillBillsStreamedAnswer covers the [P1] review case: a
// provider reports prompt_tokens only — no completion_tokens anywhere in the stream — and
// then streams the answer text. The old completion gate keyed off the coarse providerUsage
// flag, which this partial usage object already raises, so the local estimate was vetoed
// and the text the client actually received was billed as zero output.
func TestProxyRequest_PartialUsageStillBillsStreamedAnswer(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		// Usage carries prompt tokens only; the answer arrives after it, and no later
		// chunk repeats the usage.
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-partial\",\"model\":\"test-partial\",\"choices\":[],\"usage\":{\"prompt_tokens\":900}}\n\n" +
				"data: {\"id\":\"chat-partial\",\"model\":\"test-partial\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat-partial\",\"model\":\"test-partial\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	for _, credType := range chatStreamCreditTypes {
		t.Run(string(credType), func(t *testing.T) {
			prx, dbStub := newChatStreamTestProxy(t, upstream.URL, credType, "test-partial")

			w := postChatStream(t, prx, "test-partial")

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 900, entry.PromptTokens)
			assert.Greater(t, entry.CompletionTokens, 0,
				"text was streamed to the client, so the missing provider count must fall back to the local estimate")
			assert.Equal(t, float64(900+2*entry.CompletionTokens), entry.Spend)
		})
	}
}

// TestProxyRequest_PlaceholderZeroBeforeTextStillBillsStreamedAnswer covers the [P2]
// counterpart of the [P1] case above: the usage object is not partial — it reports
// completion_tokens:0 — but it arrives before the answer text it describes, and no
// later frame repeats the count. That zero means "nothing generated yet", not
// "nothing generated", so it must not veto the estimate for the text the client
// received (this is the CometAPI-style placeholder usage those providers put in the
// first frame; without a guard the tracked final-usage frame would make it final).
func TestProxyRequest_PlaceholderZeroBeforeTextStillBillsStreamedAnswer(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		// A zero completion count first, the answer after it, and no usage object
		// afterwards: the zero is the last thing the provider said about completion.
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-placeholder\",\"model\":\"test-placeholder\",\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":0,\"total_tokens\":900}}\n\n" +
				"data: {\"id\":\"chat-placeholder\",\"model\":\"test-placeholder\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat-placeholder\",\"model\":\"test-placeholder\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	for _, credType := range chatStreamCreditTypes {
		t.Run(string(credType), func(t *testing.T) {
			prx, dbStub := newChatStreamTestProxy(t, upstream.URL, credType, "test-placeholder")

			w := postChatStream(t, prx, "test-placeholder")

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 900, entry.PromptTokens)
			assert.Greater(t, entry.CompletionTokens, 0,
				"the zero arrived before the answer, so it cannot stand for the whole stream")
			assert.Equal(t, float64(900+2*entry.CompletionTokens), entry.Spend)
		})
	}
}

// TestProxyRequest_FinalZeroInLastAnswerChunkBillsNoOutput pins the other side of the
// staleness boundary: a completion_tokens:0 that arrives together with the last answer
// text (or after all of it) is the provider's final word about the whole stream, so it
// wins over the estimate even though text was streamed — the local count is not allowed
// to override an explicit provider zero.
func TestProxyRequest_FinalZeroInLastAnswerChunkBillsNoOutput(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		// The usage frame carries the final text of the answer itself.
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-samechunk\",\"model\":\"test-samechunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello there\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat-samechunk\",\"model\":\"test-samechunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" again\"}}],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":0,\"total_tokens\":900}}\n\n" +
				"data: {\"id\":\"chat-samechunk\",\"model\":\"test-samechunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	for _, credType := range chatStreamCreditTypes {
		t.Run(string(credType), func(t *testing.T) {
			prx, dbStub := newChatStreamTestProxy(t, upstream.URL, credType, "test-samechunk")

			w := postChatStream(t, prx, "test-samechunk")

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 900, entry.PromptTokens)
			assert.Zero(t, entry.CompletionTokens,
				"the zero came with the last answer text, so it speaks for the whole stream")
			assert.Equal(t, float64(900), entry.Spend)
		})
	}
}

// TestProxyRequest_FinalZeroCompletionOverridesIntermediate covers the [P2] review case: an
// intermediate usage chunk says completion_tokens:1, the final usage report says 0. The
// stream merge skips zeros (MergeNonZero), so the finalize step used to keep the stale 1
// and bill one output token for an answer the provider itself counted as empty.
func TestProxyRequest_FinalZeroCompletionOverridesIntermediate(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		// No content at all: the final usage object is the provider's correction of the
		// intermediate count, and it is the stream's last data chunk.
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-zero\",\"model\":\"test-zero\",\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":1,\"total_tokens\":901}}\n\n" +
				"data: {\"id\":\"chat-zero\",\"model\":\"test-zero\",\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":0,\"total_tokens\":900}}\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	for _, credType := range chatStreamCreditTypes {
		t.Run(string(credType), func(t *testing.T) {
			prx, dbStub := newChatStreamTestProxy(t, upstream.URL, credType, "test-zero")

			w := postChatStream(t, prx, "test-zero")

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 900, entry.PromptTokens)
			assert.Zero(t, entry.CompletionTokens,
				"an explicit final completion_tokens:0 must overwrite the intermediate 1")
			assert.Equal(t, float64(900), entry.Spend)
		})
	}
}

// TestProxyRequest_FinalZeroNotInLastChunk_Messages covers the same [P2] rule on the
// Messages wire, where the provider's final usage is not the stream's final data frame:
// usage rides in message_delta, and message_stop closes the stream with no usage at all.
// Presence therefore cannot be read from lastChunk; it is read from the last chunk that
// carried usage, which is what the merging paths now keep. Both a direct Anthropic
// credential (finalizeStreamingLog) and a proxy-like one (which writes its own spend
// row) have to honour it.
func TestProxyRequest_FinalZeroNotInLastChunk_Messages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/messages", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_mid","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1000,"output_tokens":1}}}

`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":null},"usage":{"output_tokens":1}}

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

	for _, credType := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeProxy} {
		t.Run(string(credType), func(t *testing.T) {
			cred := config.CredentialConfig{
				Name: string(credType), Type: credType,
				BaseURL: upstream.URL, APIKey: "upstream-key", RPM: 100, TPM: 10000,
			}
			prx := NewTestProxyBuilder().WithCredentials(cred).WithMasterKey("master-key").Build()
			dbStub := &stubLiteLLMManager{}
			prx.LiteLLMDB = dbStub
			prx.responseCompat = compatlitellm.New()

			registry := pricing.NewModelPriceRegistry()
			registry.Update(map[string]*pricing.ModelPrice{
				"claude-mid": {InputCostPerToken: 1, OutputCostPerToken: 2},
			})
			prx.priceRegistry = registry

			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-mid",
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
			assert.Equal(t, 1000, entry.PromptTokens)
			assert.Zero(t, entry.CompletionTokens,
				"message_stop is the last data chunk, but the final usage lives in the message_delta before it")
			assert.Equal(t, float64(1000), entry.Spend)
		})
	}
}

// TestProxyRequest_FinalZeroNotInLastChunk_Chat is the Chat Completions twin: the final
// usage frame says completion_tokens:0, but one more non-usage frame follows it. Using
// lastChunk alone would only see that trailing frame and leave the intermediate 1 in
// place — the shape the empty-chat regression test already exercises from the other side.
func TestProxyRequest_FinalZeroNotInLastChunk_Chat(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chat-mid\",\"model\":\"test-mid\",\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":1,\"total_tokens\":901}}\n\n" +
				"data: {\"id\":\"chat-mid\",\"model\":\"test-mid\",\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":0,\"total_tokens\":900}}\n\n" +
				"data: {\"id\":\"chat-mid\",\"model\":\"test-mid\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
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
		"test-mid": {InputCostPerToken: 1, OutputCostPerToken: 2},
	})
	prx.priceRegistry = registry

	body, err := json.Marshal(map[string]any{
		"model": "test-mid", "max_tokens": 32, "stream": true,
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
		"the zero usage frame is followed by a non-usage frame, so lastChunk alone cannot see it")
	assert.Equal(t, float64(900), entry.Spend)
}

// TestProxyRequest_ZeroOnlyUsageEstimatesPrompt covers the review case of a usage object that
// reports prompt_tokens:0 for a non-empty request (the placeholder some providers put in the
// first frame). The direct path estimates the prompt and bills it; the proxy-like path used to
// bill nothing at all, because the shared converter turns an all-zero usage into no TokenUsage,
// so the caller's PromptTokens backfill never ran. The rule now lives in buildFallbackUsage, so
// every credential kind reports the same prompt for the same stream and request.
func TestProxyRequest_ZeroOnlyUsageEstimatesPrompt(t *testing.T) {
	const model = "test-zero-only"
	longPrompt := strings.Repeat("word ", 300)
	zeroUsage := `data: {"id":"z","model":"test-zero-only","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}` + "\n\n"
	zeroPromptUsage := `data: {"id":"z","model":"test-zero-only","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":6,"total_tokens":6}}` + "\n\n"
	finish := `data: {"id":"z","model":"test-zero-only","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	done := "data: [DONE]\n\n"

	tests := []struct {
		name           string
		sse            string
		wantEstimate   bool
		wantCompletion int
	}{
		// The review's probe: a usage-less finish frame follows, so the provider's last word
		// about the stream is the answer, not the placeholder zero — the prompt must be
		// estimated rather than billed as zero input for a 300-word request.
		{name: "zero usage followed by a finish frame", sse: zeroUsage + finish + done, wantEstimate: true},
		// A zero that closes the stream is the provider's final report and is trusted, the
		// same explicitZeroUsage rule TestFinalizeStreamingLogKeepsExplicitZeroProviderUsage
		// pins for the direct path.
		{name: "zero usage closing the stream", sse: zeroUsage + done},
		// A zero prompt alongside a real completion count is the internally inconsistent
		// CometAPI placeholder: the completion is kept as reported, the prompt is estimated.
		{name: "zero prompt with a reported completion", sse: zeroPromptUsage + done,
			wantEstimate: true, wantCompletion: 6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(tt.sse))
			}))
			defer upstream.Close()

			prompts := make(map[config.ProviderType]int, len(chatStreamCreditTypes))
			for _, credType := range chatStreamCreditTypes {
				prx, dbStub := newChatStreamTestProxy(t, upstream.URL, credType, model)

				w := postChatStreamPrompt(t, prx, model, longPrompt)

				require.Equal(t, http.StatusOK, w.Code)
				require.Len(t, dbStub.loggedEntries, 1)
				entry := dbStub.loggedEntries[0]
				assert.Equal(t, tt.wantCompletion, entry.CompletionTokens,
					"a reported completion count is kept, a zero one is the provider's final word")
				if tt.wantEstimate {
					assert.Greater(t, entry.PromptTokens, 0,
						"the request was not empty, so a placeholder zero cannot bill zero input")
				} else {
					assert.Zero(t, entry.PromptTokens,
						"an all-zero usage that closes the stream is the provider's final report")
				}
				assert.Equal(t, float64(entry.PromptTokens+2*entry.CompletionTokens), entry.Spend)
				prompts[credType] = entry.PromptTokens
			}

			want := prompts[chatStreamCreditTypes[0]]
			for _, credType := range chatStreamCreditTypes {
				assert.Equal(t, want, prompts[credType],
					"%s must apply the same prompt rule as %s", credType, chatStreamCreditTypes[0])
			}
		})
	}
}
