package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimatePromptTokensForModel_OpenAIChatUsesTokenizer(t *testing.T) {
	body := []byte(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}`)

	got := estimatePromptTokensForModel(body, "gpt-4o-mini")

	assert.Equal(t, 8, got)
}

func TestEstimatePromptTokensForModel_GPT5FamilyUsesTokenizer(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	for _, model := range []string{"gpt-5", "gpt-5-mini", "gpt-5.5"} {
		t.Run(model, func(t *testing.T) {
			got := estimatePromptTokensForModel(body, model)

			assert.Equal(t, 8, got)
		})
	}
}

func TestEstimatePromptTokensForModel_UnknownModelUsesDefaultTokenizer(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	got := estimatePromptTokensForModel(body, "claude-sonnet-4")

	assert.Equal(t, 8, got)
}

func TestEstimatePromptTokensForModel_ResponsesAPIUsesTokenizer(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-4","input":"hello world","instructions":"be brief"}`)

	got := estimatePromptTokensForModel(body, "")

	expected := countTextTokensForModel("claude-sonnet-4", "hello world") +
		countTextTokensForModel("claude-sonnet-4", "be brief")
	assert.Equal(t, expected, got)
}

func TestCountTextTokens_LongUnbrokenText(t *testing.T) {
	text := strings.Repeat("a", 280000)

	got := countTextTokensForModel("gpt-4o", text)

	assert.Greater(t, got, 0)
}

func TestCountTextTokens_LongBase64LikeText(t *testing.T) {
	text := strings.Repeat("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo", 8192)

	got := countTextTokensForModel("gpt-4o", text)

	assert.Greater(t, got, 0)
}

func TestCountTextTokens_OrdinaryLongTextMatchesDirectTokenizer(t *testing.T) {
	enc := tokenizerForModel("gpt-4o")
	text := strings.Repeat("ordinary text with spaces ", 4096)

	got := countTextTokens(enc, text)

	assert.Equal(t, countTextTokensDirect(enc, text), got)
}

func TestCountTextTokens_OnlyLongWordsUseEstimate(t *testing.T) {
	enc := tokenizerForModel("gpt-4o")
	prefix := "ordinary text before "
	longWord := strings.Repeat("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo", 256)
	suffix := " ordinary text after"

	got := countTextTokens(enc, prefix+longWord+suffix)

	expected := countTextTokensDirect(enc, prefix) +
		estimateLongWordTokens(enc, longWord) +
		countTextTokensDirect(enc, suffix)
	assert.Equal(t, expected, got)
}

func TestCountTextTokens_IsNotAdditiveAcrossSubstrings(t *testing.T) {
	model := "gpt-4o"

	full := countTextTokensForModel(model, "hello")
	parts := countTextTokensForModel(model, "he") + countTextTokensForModel(model, "llo")

	assert.NotEqual(t, full, parts)
}

func TestCompletionTokenAccumulator_GPT5FamilyUsesTokenizer(t *testing.T) {
	for _, model := range []string{"gpt-5", "gpt-5-mini", "gpt-5.5"} {
		t.Run(model, func(t *testing.T) {
			acc := newCompletionTokenAccumulator(model)
			acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"))
			acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":" world"}}]}` + "\n\n"))

			assert.Equal(t, 2, acc.TokenCount())
		})
	}
}

func TestCompletionTokenAccumulator_CountsJoinedOpenAIText(t *testing.T) {
	acc := newCompletionTokenAccumulator("gpt-4")
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"))
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":" world"}}]}` + "\n\n"))

	assert.Equal(t, 2, acc.TokenCount())
}

func TestCompletionTokenAccumulator_UnknownModelUsesDefaultTokenizer(t *testing.T) {
	acc := newCompletionTokenAccumulator("claude-sonnet-4")
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"))
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":" world"}}]}` + "\n\n"))

	assert.Equal(t, 2, acc.TokenCount())
}

func TestExtractCompletionDeltaText_ResponsesAPI(t *testing.T) {
	chunk := []byte(`data: {"type":"response.output_text.delta","delta":"hello"}` + "\n\n" +
		`data: {"type":"response.output_text.delta","delta":" world"}` + "\n\n")

	assert.Equal(t, "hello world", extractCompletionDeltaText(chunk))
}

func TestExtractCompletionDeltaText_ResponsesReasoningAndTools(t *testing.T) {
	chunk := []byte(
		`data: {"type":"response.reasoning_text.delta","delta":"think"}` + "\n\n" +
			`data: {"type":"response.reasoning_summary_text.delta","delta":"sum"}` + "\n\n" +
			`data: {"type":"response.function_call_arguments.delta","delta":"fn"}` + "\n\n" +
			`data: {"type":"response.mcp_call_arguments.delta","delta":"mcp"}` + "\n\n" +
			`data: {"type":"response.custom_tool_call_input.delta","delta":"custom"}` + "\n\n" +
			`data: {"type":"response.code_interpreter_call_code.delta","delta":"code"}` + "\n\n")

	assert.Equal(t, "thinksumfnmcpcustomcode", extractCompletionDeltaText(chunk))
}

func TestExtractCompletionDeltaText_MessagesAPI(t *testing.T) {
	chunk := []byte(
		`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}` + "\n\n" +
			`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"think"}}` + "\n\n" +
			`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Moscow\"}"}}` + "\n\n")

	assert.Equal(t, `hellothink{"city":"Moscow"}`, extractCompletionDeltaText(chunk))
}

func TestExtractCompletionDeltaText_IgnoresAudioBytes(t *testing.T) {
	chunk := []byte(`data: {"type":"response.output_audio.delta","delta":"QUJDREVGRw=="}` + "\n\n" +
		`data: {"type":"response.audio.delta","delta":"QUJDREVGRw=="}` + "\n\n")

	assert.Equal(t, "", extractCompletionDeltaText(chunk))
}

// --- Fix A: lazy prompt-token estimate + tiktoken_enabled toggle ---

func TestRequestLogContext_PromptTokensEstimate_MemoizedAfterFirstCall(t *testing.T) {
	calls := 0
	logCtx := &RequestLogContext{
		promptTokensEstimateFn: func() int {
			calls++
			return 42
		},
	}

	assert.Equal(t, 42, logCtx.promptTokensEstimate())
	assert.Equal(t, 42, logCtx.promptTokensEstimate())
	assert.Equal(t, 42, logCtx.promptTokensEstimate())
	assert.Equal(t, 1, calls, "estimator closure must run at most once, memoized afterward")
}

func TestRequestLogContext_PromptTokensEstimate_ZeroWhenUnarmed(t *testing.T) {
	logCtx := &RequestLogContext{}
	assert.Equal(t, 0, logCtx.promptTokensEstimate())

	var nilLogCtx *RequestLogContext
	assert.Equal(t, 0, nilLogCtx.promptTokensEstimate())
}

func TestProxy_SetPromptTokensEstimate_ComputesWhenTiktokenEnabled(t *testing.T) {
	body := []byte(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello world"}]}`)
	prx := NewTestProxyBuilder().Build() // tiktoken_enabled defaults to true

	logCtx := &RequestLogContext{}
	prx.setPromptTokensEstimate(logCtx, body, "gpt-4o-mini")

	assert.NotNil(t, logCtx.promptTokensEstimateFn)
	assert.Equal(t, estimatePromptTokensForModel(body, "gpt-4o-mini"), logCtx.promptTokensEstimate())
}

func TestProxy_SetPromptTokensEstimate_NoOpWhenTiktokenDisabled(t *testing.T) {
	body := []byte(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello world"}]}`)
	prx := NewTestProxyBuilder().WithTiktokenEnabled(false).Build()

	logCtx := &RequestLogContext{}
	prx.setPromptTokensEstimate(logCtx, body, "gpt-4o-mini")

	assert.Nil(t, logCtx.promptTokensEstimateFn, "no estimator should be armed when tiktoken_enabled=false")
	assert.Equal(t, 0, logCtx.promptTokensEstimate())
}

func TestProxy_NewCompletionTokenAccumulator_NilWhenTiktokenDisabled(t *testing.T) {
	prx := NewTestProxyBuilder().WithTiktokenEnabled(false).Build()

	acc := prx.newCompletionTokenAccumulator("gpt-4o-mini")
	require.Nil(t, acc)

	// AddChunk/TokenCount on a nil accumulator must stay safe no-ops — this is how
	// the streaming hot path skips the per-chunk delta-text JSON decode entirely
	// when tiktoken_enabled=false, rather than just discarding its result.
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"))
	assert.Equal(t, 0, acc.TokenCount())
}

func TestProxy_NewCompletionTokenAccumulator_ActiveWhenTiktokenEnabled(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	acc := prx.newCompletionTokenAccumulator("gpt-4o-mini")
	require.NotNil(t, acc)
	acc.AddChunk([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"))
	assert.Greater(t, acc.TokenCount(), 0)
}

// TestFinalizeStreamingLog_EstimatorNeverInvokedWhenProviderSendsUsage confirms the
// call-count claim behind fix A: once a provider genuinely reports usage
// (providerUsage=true), the armed local-tokenizer fallback closure must never run at
// all, not merely have its result discarded.
func TestFinalizeStreamingLog_EstimatorNeverInvokedWhenProviderSendsUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	calls := 0
	logCtx := &RequestLogContext{
		Request: httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		promptTokensEstimateFn: func() int {
			calls++
			return 999
		},
		TokenUsage: &converter.TokenUsage{},
	}
	lastChunk := []byte(`data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60}}`)

	prx.finalizeStreamingLog(logCtx, 0, lastChunk, "openai", http.StatusOK, false)

	assert.Equal(t, 0, calls, "estimator closure must never be invoked when the provider sent real usage")
	assert.Equal(t, 50, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, "provider", logCtx.UsageSource)
}

// TestFinalizeStreamingLog_EstimatorInvokedOnceWhenProviderOmitsUsage is the
// complementary case: the estimator must still run exactly once — not zero times,
// not on every access — when the provider never sends usage.
func TestFinalizeStreamingLog_EstimatorInvokedOnceWhenProviderOmitsUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	calls := 0
	logCtx := &RequestLogContext{
		Request: httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		promptTokensEstimateFn: func() int {
			calls++
			return 77
		},
		TokenUsage: &converter.TokenUsage{},
	}

	prx.finalizeStreamingLog(logCtx, 5, nil, "openai", http.StatusOK, false)

	assert.Equal(t, 1, calls)
	assert.Equal(t, 77, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, "estimated", logCtx.UsageSource)
}
