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
	require.Len(t, frames, 4)
	assert.Equal(t, "[DONE]", frames[3])

	var first map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &first))
	assert.Equal(t, "provider-id", first["id"])
	assert.Equal(t, float64(10), first["created"])
	assert.Equal(t, "public-model", first["model"])
	firstChoice := first["choices"].([]any)[0].(map[string]any)
	firstDelta := firstChoice["delta"].(map[string]any)
	assert.Equal(t, "assistant", firstDelta["role"])

	var second map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &second))
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
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"model\":\"provider-model\"},\"error\":{\"code\":null}}\n\n",
	)
	output, err := io.ReadAll(New().Stream(Context{
		Endpoint:       "/v1/responses",
		RequestedModel: "public-model",
		IncludeUsage:   true,
	}, source))
	require.NoError(t, err)

	assert.Contains(t, string(output), "event: response.completed")
	assert.Contains(t, string(output), `"model":"public-model"`)
	assert.Contains(t, string(output), `"code":"unknown_error"`)
	assert.True(t, strings.HasSuffix(string(output), "data: [DONE]\n\n"))
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
	require.Len(t, frames, 4)

	var content map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &content))
	contentChoice := content["choices"].([]any)[0].(map[string]any)
	assert.NotContains(t, contentChoice, "finish_reason")

	var terminal map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &terminal))
	terminalChoice := terminal["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, "tool_calls", terminalChoice["finish_reason"])
	assert.Equal(t, "[DONE]", frames[3])
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
	require.Len(t, frames, 4)

	var content map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &content))
	contentChoices := content["choices"].([]any)
	require.Len(t, contentChoices, 2)
	for _, rawChoice := range contentChoices {
		choice := rawChoice.(map[string]any)
		assert.NotContains(t, choice, "finish_reason")
		assert.Equal(t, "assistant", choice["delta"].(map[string]any)["role"])
	}

	var terminal map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[2]), &terminal))
	terminalChoices := terminal["choices"].([]any)
	require.Len(t, terminalChoices, 2)
	first := terminalChoices[0].(map[string]any)
	second := terminalChoices[1].(map[string]any)
	assert.Equal(t, float64(0), first["index"])
	assert.Equal(t, "tool_calls", first["finish_reason"])
	assert.Equal(t, float64(1), second["index"])
	assert.Equal(t, "stop", second["finish_reason"])
	assert.Equal(t, "[DONE]", frames[3])
}

func TestChatStreamMatchesLiteLLMFraming(t *testing.T) {
	source := strings.NewReader(
		"data: {\"id\":\"\",\"created\":10,\"model\":\"public-model\",\"object\":\"chat.completion.chunk\",\"choices\":[]}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":0,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":0,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n" +
			"data: {\"id\":\"air-id\",\"created\":0,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
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

	var roleChunk map[string]any
	require.NoError(t, json.Unmarshal([]byte(frames[1]), &roleChunk))
	roleChoice := roleChunk["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, map[string]any{"content": "", "role": "assistant"}, roleChoice["delta"])

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
