package responses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatStreamRefusalSurvivesFinalResponse(t *testing.T) {
	for _, partialText := range []bool{false, true} {
		name := "empty"
		if partialText {
			name = "after text"
		}
		t.Run(name, func(t *testing.T) {
			stream := ""
			if partialText {
				stream += buildSSEChunk(buildChatChunk("Partial answer", nil))
			}
			stream += buildSSEChunk(`{"choices":[{"index":0,"delta":{"refusal":"Request "}}]}`)
			stream += buildSSEChunk(`{"choices":[{"index":0,"delta":{"refusal":"declined."},"finish_reason":"content_filter"}]}`)
			stream += buildSSEChunk(buildUsageChunk(17, 0, 17)) + "data: [DONE]\n\n"
			var out bytes.Buffer
			var final *Response
			require.NoError(t, TransformChatStreamToResponses(strings.NewReader(stream), &out, "claude-test", func(r *Response) { final = r }))
			require.NotNil(t, final)
			index := 0
			if partialText {
				index = 1
				assert.Equal(t, "Partial answer", final.Output[0].Content[0].Text)
			}
			require.Len(t, final.Output, index+1)
			refusal := final.Output[index]
			assert.Equal(t, "refusal", refusal.Content[0].Type)
			assert.Equal(t, "Request declined.", refusal.Content[0].Refusal)
			assert.Equal(t, "incomplete", final.Status)
			assert.Equal(t, "content_filter", final.IncompleteDetails.Reason)
			assert.Equal(t, 17, final.Usage.InputTokens)
			assert.Zero(t, final.Usage.OutputTokens)
			var lifecycle []string
			var snapshot Response
			for _, line := range strings.Split(out.String(), "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var event map[string]interface{}
				require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event))
				if event["item_id"] == refusal.ID {
					assert.Equal(t, float64(index), event["output_index"])
					assert.Equal(t, float64(0), event["content_index"])
					lifecycle = append(lifecycle, event["type"].(string))
					if event["type"] == "response.content_part.added" {
						assert.Equal(t, map[string]interface{}{"type": "refusal", "refusal": ""}, event["part"])
					}
				}
				if event["type"] == "response.output_item.done" && event["output_index"] == float64(index) {
					lifecycle = append(lifecycle, "response.output_item.done")
				}
				if event["type"] == "response.incomplete" {
					body, err := json.Marshal(event["response"])
					require.NoError(t, err)
					require.NoError(t, json.Unmarshal(body, &snapshot))
				}
			}
			assert.Equal(t, []string{"response.content_part.added", "response.refusal.delta", "response.refusal.delta", "response.refusal.done", "response.content_part.done", "response.output_item.done"}, lifecycle)
			assert.Equal(t, final.Output, snapshot.Output)
		})
	}
}

func TestRefusalHistoryRoundTrip(t *testing.T) {
	for _, kind := range []string{"refusal", "output_refusal"} {
		t.Run(kind, func(t *testing.T) {
			items := outputToInputItems([]OutputItem{{Type: "message", Role: "assistant", Content: []OutputContent{{Type: kind, Refusal: "Request declined."}}}})
			require.Len(t, items, 1)
			parts := items[0].(map[string]interface{})["content"].([]interface{})
			require.Len(t, parts, 1)
			assert.Equal(t, "refusal", parts[0].(map[string]interface{})["type"])
			converted, err := convertContentParts(parts)
			require.NoError(t, err)
			assert.Equal(t, []interface{}{map[string]interface{}{"type": "text", "text": "Request declined."}}, converted)
		})
	}
}

func TestRefusalHistoryDoesNotSplitToolCallFromResult(t *testing.T) {
	for _, withText := range []bool{false, true} {
		var input []interface{}
		if withText {
			input = append(input, map[string]interface{}{"type": "message", "role": "assistant", "content": []interface{}{map[string]interface{}{"type": "output_text", "text": "Partial answer"}}})
		}
		input = append(input,
			map[string]interface{}{"type": "function_call", "call_id": "call_fixture", "name": "lookup", "arguments": "{}"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": []interface{}{map[string]interface{}{"type": "refusal", "refusal": "Request declined."}}},
			map[string]interface{}{"type": "function_call_output", "call_id": "call_fixture", "output": "Not executed"})
		body, err := json.Marshal(map[string]interface{}{"input": input})
		require.NoError(t, err)
		converted, err := RequestToChat(body)
		require.NoError(t, err)
		var request struct {
			Messages []struct {
				Role       string                   `json:"role"`
				Content    json.RawMessage          `json:"content"`
				ToolCallID string                   `json:"tool_call_id"`
				ToolCalls  []map[string]interface{} `json:"tool_calls"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(converted, &request))
		require.Len(t, request.Messages, 2)
		assert.Equal(t, "assistant", request.Messages[0].Role)
		assert.Contains(t, string(request.Messages[0].Content), "Request declined.")
		if withText {
			assert.Contains(t, string(request.Messages[0].Content), "Partial answer")
		}
		require.Len(t, request.Messages[0].ToolCalls, 1)
		assert.Equal(t, "call_fixture", request.Messages[0].ToolCalls[0]["id"])
		assert.Equal(t, "tool", request.Messages[1].Role)
		assert.Equal(t, "call_fixture", request.Messages[1].ToolCallID)
	}
}
