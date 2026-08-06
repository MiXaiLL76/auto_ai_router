package litellm

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatStreamCompatibility(t *testing.T) {
	source := strings.NewReader(
		"data: {\"id\":\"provider-id\",\"created\":10,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"other-id\",\"created\":20,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"other-id\",\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
		IncludeUsage:   false,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 3)
	assert.Equal(t, "[DONE]", frames[2])

	var first map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[0]), &first))
	assert.Equal(t, "provider-id", first["id"])
	assert.Equal(t, float64(10), first["created"])
	assert.Equal(t, "public-model", first["model"])
	firstChoice := first["choices"].([]any)[0].(map[string]any)
	firstDelta := firstChoice["delta"].(map[string]any)
	assert.Equal(t, "assistant", firstDelta["role"])

	var second map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &second))
	assert.Equal(t, "provider-id", second["id"])
	assert.Equal(t, float64(10), second["created"])
	secondChoice := second["choices"].([]any)[0].(map[string]any)
	secondDelta := secondChoice["delta"].(map[string]any)
	assert.NotContains(t, secondDelta, "role")
	assert.Equal(t, "stop", secondChoice["finish_reason"])

	assert.NotContains(t, string(output), `"usage"`)
}

func TestResponsesStreamCompatibility(t *testing.T) {
	source := strings.NewReader(
		"event: response.completed\n" +
			"data: {\"type\":\"response.completed\",\"model\":\"public-model\",\"response\":{\"id\":\"resp-1\",\"model\":\"provider-model\"},\"error\":{\"code\":null}}\n\n",
	)
	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/responses",
		RequestedModel: "public-model",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	assert.NotContains(t, string(output), "event: response.completed")
	frames := splitDataFrames(string(output))
	require.Len(t, frames, 2)
	var completed map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[0]), &completed))
	assert.Equal(t, "public-model", completed["model"])
	assert.Equal(t, "provider-model", completed["response"].(map[string]any)["model"])
	assert.Contains(t, string(output), `"code":"unknown_error"`)
	assert.True(t, strings.HasSuffix(string(output), "data: [DONE]\n\n"))
}

func TestChatStreamDropsProviderOnlyChoiceFields(t *testing.T) {
	source := strings.NewReader(
		": OPENROUTER PROCESSING\n\n" +
			`data: {"id":"deepseek-id","created":10,"choices":[{"index":0,"delta":{"role":"assistant","reasoning":"think","content":null},"matched_stop":1}]}` + "\n\n" +
			`data: {"id":"deepseek-id","created":10,"choices":[{"index":0,"delta":{"content":"","reasoning":null},"native_finish_reason":"length"}]}` + "\n\n" +
			`data: {"id":"deepseek-id","created":10,"choices":[{"index":0,"delta":{"content":""},"finish_reason":"length","matched_stop":1}]}` + "\n\n" +
			`data: {"id":"deepseek-id","created":10,"choices":[],"system_fingerprint":null,"usage":{"prompt_tokens":11,"completion_tokens":25,"total_tokens":36,"completion_tokens_details":{"reasoning_tokens":20}}}` + "\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "deepseek/deepseek-v4-flash",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 4)
	assert.Contains(t, frames[0], `"reasoning_content":"think"`)
	assert.NotContains(t, frames[0], `"content"`)
	assert.NotContains(t, string(output), "matched_stop")
	assert.NotContains(t, string(output), "native_finish_reason")
	assert.NotContains(t, string(output), "OPENROUTER PROCESSING")
	assert.NotContains(t, string(output), `"content":null`)
	assert.Contains(t, frames[1], `"finish_reason":"length"`)
	assert.Contains(t, frames[2], `"total_tokens":36`)
	assert.Equal(t, "[DONE]", frames[3])
}

func TestChatStreamPreservesUsageWithoutSystemFingerprint(t *testing.T) {
	source := strings.NewReader(
		`data: {"id":"gpt-id","created":10,"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"content_filter_results":{"hate":{"filtered":false}}}]}` + "\n\n" +
			`data: {"id":"gpt-id","created":10,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			`data: {"id":"gpt-id","created":10,"choices":[],"latency_checkpoint":{"engine_ttft_ms":10},"obfuscation":"","service_tier":"default","system_fingerprint":null,"usage":{"prompt_tokens":12,"completion_tokens":6,"total_tokens":18}}` + "\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "openai/gpt-5.4-mini",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 4)
	var usageChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &usageChunk))
	usage := usageChunk["usage"].(map[string]any)
	assert.Equal(t, float64(18), usage["total_tokens"])
	assert.NotContains(t, string(output), "latency_checkpoint")
	assert.NotContains(t, string(output), "service_tier")
}

func TestChatStreamKeepsOpenAIMetadataAndZeroUsage(t *testing.T) {
	source := strings.NewReader(
		`data: {"id":"gpt-id","created":10,"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}],"obfuscation":"abc","service_tier":"default","system_fingerprint":"fp-1"}` + "\n\n" +
			`data: {"id":"gpt-id","created":10,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"obfuscation":"abc","service_tier":"default","system_fingerprint":"fp-1"}` + "\n\n" +
			`data: {"id":"gpt-id","created":10,"choices":[],"latency_checkpoint":{"engine_ttft_ms":10},"obfuscation":"abc","service_tier":"default","system_fingerprint":"fp-1","usage":{"prompt_tokens":13,"completion_tokens":3,"total_tokens":16}}` + "\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "openai/gpt-4.1-mini",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 4)
	assert.Contains(t, frames[0], `"obfuscation":"abc"`)
	assert.Contains(t, frames[0], `"service_tier":"default"`)
	assert.Contains(t, frames[0], `"system_fingerprint":"fp-1"`)
	var usageChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &usageChunk))
	usage := usageChunk["usage"].(map[string]any)
	assert.Equal(t, float64(0), usage["total_tokens"])
	assert.NotContains(t, frames[2], "service_tier")
}

func TestChatStreamMovesFinishReasonToTerminalChunk(t *testing.T) {
	source := strings.NewReader(
		"data: {\"model\":\"provider-model\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 3)

	var content map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[0]), &content))
	contentChoice := content["choices"].([]any)[0].(map[string]any)
	assert.NotContains(t, contentChoice, "finish_reason")

	var terminal map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &terminal))
	terminalChoice := terminal["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, "tool_calls", terminalChoice["finish_reason"])
	assert.Equal(t, "[DONE]", frames[2])
}

func TestChatStreamTracksEachChoiceIndependently(t *testing.T) {
	source := strings.NewReader(
		"data: {\"model\":\"provider-model\",\"choices\":[" +
			"{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":\"stop\"}," +
			"{\"index\":1,\"delta\":{\"content\":\"alternative\"},\"finish_reason\":\"stop\"}" +
			"]}\n\n" +
			"data: [DONE]\n\n",
	)
	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 3)

	var content map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[0]), &content))
	contentChoices := content["choices"].([]any)
	require.Len(t, contentChoices, 2)
	for _, rawChoice := range contentChoices {
		choice := rawChoice.(map[string]any)
		assert.NotContains(t, choice, "finish_reason")
		assert.Equal(t, "assistant", choice["delta"].(map[string]any)["role"])
	}

	var terminal map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &terminal))
	terminalChoices := terminal["choices"].([]any)
	require.Len(t, terminalChoices, 2)
	first := terminalChoices[0].(map[string]any)
	second := terminalChoices[1].(map[string]any)
	assert.Equal(t, float64(0), first["index"])
	assert.Equal(t, "tool_calls", first["finish_reason"])
	assert.Equal(t, float64(1), second["index"])
	assert.Equal(t, "stop", second["finish_reason"])
	assert.Equal(t, "[DONE]", frames[2])
}

func TestChatStreamPreservesFinalUsage(t *testing.T) {
	source := strings.NewReader(
		`data: {"id":"qwen-id","created":10,"model":"provider-model","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}` + "\n\n" +
			`data: {"id":"qwen-id","created":10,"model":"provider-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			`data: {"id":"qwen-id","created":10,"model":"provider-model","choices":[],"usage":{"prompt_tokens":16,"completion_tokens":165,"total_tokens":181,"completion_tokens_details":{"reasoning_tokens":158,"text_tokens":165},"prompt_tokens_details":{"cached_tokens":0,"text_tokens":16}}}` + "\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "qwen/qwen3.7-plus",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 4)
	var usageChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &usageChunk))
	usage := usageChunk["usage"].(map[string]any)
	assert.Equal(t, float64(16), usage["prompt_tokens"])
	assert.Equal(t, float64(165), usage["completion_tokens"])
	assert.Equal(t, float64(181), usage["total_tokens"])
	assert.Equal(t, float64(158), usage["completion_tokens_details"].(map[string]any)["reasoning_tokens"])
}

func TestChatStreamDropsIntermediateZeroUsage(t *testing.T) {
	source := strings.NewReader(
		`data: {"id":"gemini-id","created":10,"model":"provider-model","choices":[{"index":0,"delta":{"role":"assistant","content":"stream-"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}` + "\n\n" +
			`data: {"id":"gemini-id","created":10,"model":"provider-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}` + "\n\n" +
			`data: {"id":"gemini-id","created":10,"model":"provider-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10,"prompt_tokens_details":{},"completion_tokens_details":{}}}` + "\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "google/gemini-3.6-flash",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 5)
	assert.Contains(t, frames[0], `"content":"stream-"`)
	assert.Contains(t, frames[1], `"content":"ok"`)
	assert.Contains(t, frames[2], `"finish_reason":"stop"`)
	assert.Contains(t, frames[3], `"total_tokens":10`)
	assert.Equal(t, "[DONE]", frames[4])
}

func TestChatStreamMatchesLiteLLMFraming(t *testing.T) {
	source := strings.NewReader(
		"data: {\"id\":\"\",\"created\":0,\"model\":\"provider-model\",\"object\":\"\",\"choices\":[],\"prompt_filter_results\":[{\"prompt_index\":0}]}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":10,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":10,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":10,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":0,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"content_filter_results\":{\"hate\":{\"filtered\":false}},\"content_filter_offsets\":{\"start_offset\":0}}]}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":0,\"model\":\"provider-model\",\"choices\":[],\"latency_checkpoint\":{\"engine_ttft_ms\":10},\"obfuscation\":\"\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp-1\",\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":3,\"total_tokens\":16}}\n\n" +
			"data: [DONE]\n\n",
	)

	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	frames := splitDataFrames(string(output))
	require.Len(t, frames, 6)

	var prelude map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[0]), &prelude))
	assert.Empty(t, prelude["choices"])
	assert.Equal(t, "", prelude["id"])
	assert.Equal(t, float64(10), prelude["created"])

	var roleChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &roleChunk))
	roleChoice := roleChunk["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, map[string]any{"content": "", "role": "assistant"}, roleChoice["delta"])
	assert.Equal(t, "air-id", roleChunk["id"])

	var contentChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &contentChunk))
	contentChoice := contentChunk["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, map[string]any{"content": "hello"}, contentChoice["delta"])

	var finishChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[3]), &finishChunk))
	finishChoice := finishChunk["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, "stop", finishChoice["finish_reason"])

	assert.NotContains(t, string(output), "content_filter_results")
	assert.NotContains(t, string(output), "content_filter_offsets")
	assert.NotContains(t, string(output), "latency_checkpoint")
	assert.NotContains(t, frames[4], "obfuscation")
	assert.NotContains(t, frames[4], "service_tier")
	assert.NotContains(t, frames[4], "system_fingerprint")

	var usage map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[4]), &usage))
	assert.Equal(t, map[string]any{
		"completion_tokens":         float64(0),
		"completion_tokens_details": map[string]any{"reasoning_tokens": float64(0)},
		"prompt_tokens":             float64(0),
		"total_tokens":              float64(0),
	}, usage["usage"])
	usageChoices := usage["choices"].([]any)
	require.Len(t, usageChoices, 1)
	assert.Equal(t, map[string]any{}, usageChoices[0].(map[string]any)["delta"])
	assert.Equal(t, "[DONE]", frames[5])
}

func splitDataFrames(stream string) []string {
	var frames []string
	for _, frame := range strings.Split(strings.TrimSpace(stream), "\n\n") {
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data:") {
				frames = append(frames, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
	}
	return frames
}
