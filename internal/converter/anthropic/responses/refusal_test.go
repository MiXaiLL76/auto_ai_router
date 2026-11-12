package anthropicresponses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropicRefusalResponses(t *testing.T) {
	for _, tc := range []struct {
		name, content, details, wantRefusal string
		beforeRefusal                       int
	}{
		{"empty", `[]`, `{"category":"reasoning_extraction","explanation":"Request declined.","fallback_credit_token":"private-fixture-credit"}`, "The safety classifier identified a request to reproduce internal reasoning. (category: reasoning_extraction)", 0},
		{"missing details", `[]`, `null`, "The model declined this request; no public reason was provided.", 0},
		{"null fields", `[]`, `{"category":null,"explanation":null}`, "The model declined this request; no public reason was provided.", 0},
		{"after text", `[{"type":"text","text":"Partial answer"}]`, `{"explanation":"Request declined."}`, "The model declined this request; no public reason was provided.", 1},
		{"after thinking", `[{"type":"thinking","thinking":"","signature":"fixture-signature"}]`, `null`, "The model declined this request; no public reason was provided.", 1},
		{"after text and tool", `[{"type":"text","text":"Partial answer"},{"type":"tool_use","id":"tool_fixture","name":"lookup","input":{"q":"test"}}]`, `null`, "The model declined this request; no public reason was provided.", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Sprintf(`{"content":%s,"stop_reason":"refusal","stop_details":%s,"usage":{"input_tokens":17,"output_tokens":0,"cache_read_input_tokens":3}}`, tc.content, tc.details)
			nonstream, err := AnthropicToResponsesResponse([]byte(body), "claude-test", "resp_fixture", 1)
			require.NoError(t, err)
			require.Len(t, nonstream.Output, tc.beforeRefusal+1)
			assert.Equal(t, "incomplete", nonstream.Status)
			require.NotNil(t, nonstream.IncompleteDetails)
			assert.Equal(t, "content_filter", nonstream.IncompleteDetails.Reason)
			assert.Equal(t, "refusal", nonstream.Output[tc.beforeRefusal].Content[0].Type)
			assert.Equal(t, tc.wantRefusal, nonstream.Output[tc.beforeRefusal].Content[0].Refusal)

			var blocks []map[string]interface{}
			require.NoError(t, json.Unmarshal([]byte(tc.content), &blocks))
			events := []map[string]interface{}{{"type": "message_start", "message": map[string]interface{}{
				"id": "msg_fixture", "usage": map[string]interface{}{"input_tokens": 17, "cache_read_input_tokens": 3},
			}}}
			for i, block := range blocks {
				// Text/tool content is delivered as deltas, as on the native wire.
				start := map[string]interface{}{"type": block["type"], "id": block["id"], "name": block["name"]}
				events = append(events, map[string]interface{}{"type": "content_block_start", "index": i, "content_block": start})
				var delta map[string]interface{}
				switch block["type"] {
				case "text":
					delta = map[string]interface{}{"type": "text_delta", "text": block["text"]}
				case "thinking":
					delta = map[string]interface{}{"type": "signature_delta", "signature": block["signature"]}
				case "tool_use":
					args, err := json.Marshal(block["input"])
					require.NoError(t, err)
					delta = map[string]interface{}{"type": "input_json_delta", "partial_json": string(args)}
				}
				events = append(events, map[string]interface{}{"type": "content_block_delta", "index": i, "delta": delta},
					map[string]interface{}{"type": "content_block_stop", "index": i})
			}
			var details interface{}
			require.NoError(t, json.Unmarshal([]byte(tc.details), &details))
			events = append(events,
				map[string]interface{}{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": "refusal", "stop_details": details}, "usage": map[string]interface{}{"output_tokens": 0}},
				map[string]interface{}{"type": "message_stop"})
			var out bytes.Buffer
			var final *responses.Response
			require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(buildAnthropicSSEStream(events)), &out,
				"claude-test", "resp_fixture", nil, func(r *responses.Response) { final = r }))
			require.NotNil(t, final)
			require.Len(t, final.Output, tc.beforeRefusal+1)
			assert.Equal(t, nonstream.Status, final.Status)
			assert.Equal(t, nonstream.IncompleteDetails, final.IncompleteDetails)
			assert.Equal(t, nonstream.Usage, final.Usage)
			refusal := final.Output[tc.beforeRefusal]
			assert.Equal(t, nonstream.Output[tc.beforeRefusal].Content, refusal.Content)
			wireEvents := parseSSEEvents(out.String())
			var refusalEvents []string
			for i, event := range wireEvents {
				assert.Equal(t, float64(i+1), event["sequence_number"])
				if event["item_id"] == refusal.ID {
					assert.Equal(t, float64(tc.beforeRefusal), event["output_index"])
					assert.Equal(t, float64(0), event["content_index"])
					refusalEvents = append(refusalEvents, event["type"].(string))
				}
				if event["type"] == "response.refusal.delta" {
					assert.Equal(t, tc.wantRefusal, event["delta"])
				}
				if event["type"] == "response.refusal.done" {
					assert.Equal(t, tc.wantRefusal, event["refusal"])
				}
			}
			assert.Equal(t, []string{"response.content_part.added", "response.refusal.delta", "response.refusal.done", "response.content_part.done"}, refusalEvents)
			last := wireEvents[len(wireEvents)-1]
			assert.Equal(t, "response.incomplete", last["type"])
			encoded, err := json.Marshal(last["response"])
			require.NoError(t, err)
			var snapshot responses.Response
			require.NoError(t, json.Unmarshal(encoded, &snapshot))
			wantOutput, err := json.Marshal(final.Output)
			require.NoError(t, err)
			gotOutput, err := json.Marshal(snapshot.Output)
			require.NoError(t, err)
			assert.JSONEq(t, string(wantOutput), string(gotOutput))
			assert.NotContains(t, out.String(), "private-fixture-credit")
		})
	}
}

func TestAnthropicRefusalOutputCanBeReplayed(t *testing.T) {
	response, err := AnthropicToResponsesResponse([]byte(`{"content":[{"type":"text","text":"Partial answer"}],"stop_reason":"refusal","stop_details":{"explanation":"Request declined."}}`), "claude-test", "", 0)
	require.NoError(t, err)
	for _, kind := range []string{"refusal", "output_refusal"} {
		t.Run(kind, func(t *testing.T) {
			response.Output[1].Content[0].Type = kind
			var input []interface{}
			for _, item := range response.Output {
				input = append(input, item)
			}
			input = append(input, map[string]interface{}{"role": "user", "content": "Explain the result."})
			request, err := json.Marshal(map[string]interface{}{"input": input, "max_output_tokens": 100})
			require.NoError(t, err)
			converted, err := ResponsesRequestToAnthropic(request, "claude-test")
			require.NoError(t, err)
			var native struct {
				Messages []struct {
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(converted, &native))
			require.Len(t, native.Messages, 3)
			assert.Equal(t, "assistant", native.Messages[1].Role)
			assert.JSONEq(t, `[{"type":"text","text":"The model declined this request; no public reason was provided."}]`, string(native.Messages[1].Content))
		})
	}
}
