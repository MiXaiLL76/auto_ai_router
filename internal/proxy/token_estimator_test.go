package proxy

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
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
