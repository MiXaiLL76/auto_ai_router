package anthropicresponses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const nativeAdaptiveModel = "MiniMax-M3.1-Flash"

func decodeRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &request))
	return request
}

// reasoning.effort reaches the model unchanged as adaptive thinking, with no budget and
// no raised max_tokens; without it the request has no thinking controls at all.
func TestResponsesRequestToAnthropic_NativeAdaptiveEffort(t *testing.T) {
	for _, effort := range []string{"low", "medium", "high", "xhigh", "max"} {
		t.Run(effort, func(t *testing.T) {
			body := `{"model":"minimax-m3.1-flash","input":"hi","max_output_tokens":1000,"reasoning":{"effort":"` + effort + `"}}`
			converted, err := ResponsesRequestToAnthropic([]byte(body), nativeAdaptiveModel)
			require.NoError(t, err)
			request := decodeRequest(t, converted)
			assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"])
			assert.Equal(t, map[string]interface{}{"effort": effort}, request["output_config"])
			assert.Equal(t, float64(1000), request["max_tokens"])
		})
	}

	t.Run("no reasoning keeps the model default", func(t *testing.T) {
		converted, err := ResponsesRequestToAnthropic([]byte(`{"model":"m","input":"hi","max_output_tokens":1000}`), nativeAdaptiveModel)
		require.NoError(t, err)
		request := decodeRequest(t, converted)
		assert.NotContains(t, request, "thinking")
		assert.NotContains(t, request, "output_config")
	})

	// Regression guard: models outside the rule keep the budget mapping.
	t.Run("other minimax model keeps budget thinking", func(t *testing.T) {
		body := `{"model":"m","input":"hi","max_output_tokens":1000,"reasoning":{"effort":"high"}}`
		converted, err := ResponsesRequestToAnthropic([]byte(body), "MiniMax-M3")
		require.NoError(t, err)
		request := decodeRequest(t, converted)
		assert.Equal(t, map[string]interface{}{"type": "enabled", "budget_tokens": float64(30000)}, request["thinking"])
		assert.Equal(t, float64(31024), request["max_tokens"])
	})
}

func TestAnthropicToResponsesResponse_ThinkingTokens(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"MiniMax-M3.1-Flash",
		"content":[{"type":"thinking","thinking":"Compare.","signature":"sig"},{"type":"text","text":"9.9"}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":100,"output_tokens":100,"cache_read_input_tokens":900,"output_tokens_details":{"thinking_tokens":60}}}`
	response, err := AnthropicToResponsesResponse([]byte(body), "minimax-m3.1-flash", "", 0)
	require.NoError(t, err)
	require.NotNil(t, response.Usage)
	assert.Equal(t, 1000, response.Usage.InputTokens)
	assert.Equal(t, 900, response.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, 100, response.Usage.OutputTokens)
	assert.Equal(t, 60, response.Usage.OutputTokensDetails.ReasoningTokens)
	require.Len(t, response.Output, 2)
	assert.Equal(t, "reasoning", response.Output[0].Type)
	assert.Equal(t, "message", response.Output[1].Type)
}

func TestTransformAnthropicStreamToResponses_ThinkingTokens(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":900}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Compare."}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"9.9"}}`,
		`data: {"type":"content_block_stop","index":1}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":100,"output_tokens_details":{"thinking_tokens":60}}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n")
	var completed *responses.Response
	var output strings.Builder
	require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(stream), &output, "minimax-m3.1-flash", "", nil,
		func(response *responses.Response) { completed = response }))
	require.NotNil(t, completed)
	require.NotNil(t, completed.Usage)
	assert.Equal(t, 1000, completed.Usage.InputTokens)
	assert.Equal(t, 100, completed.Usage.OutputTokens)
	assert.Equal(t, 60, completed.Usage.OutputTokensDetails.ReasoningTokens)
	assert.Contains(t, output.String(), `"reasoning_tokens":60`)
}
