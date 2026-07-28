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
