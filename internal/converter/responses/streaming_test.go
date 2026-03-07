package responses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func buildSSEChunk(data string) string {
	return "data: " + data + "\n\n"
}

func buildChatChunk(content string, finishReason *string) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"content": content,
				},
				"finish_reason": finishReason,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func buildChatChunkWithRole(role string) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"role": role,
				},
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func buildUsageChunk(promptTokens, completionTokens, totalTokens int) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      totalTokens,
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func buildToolCallStartChunk(callID, name string) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"tool_calls": []interface{}{
						map[string]interface{}{
							"index": 0,
							"id":    callID,
							"type":  "function",
							"function": map[string]interface{}{
								"name":      name,
								"arguments": "",
							},
						},
					},
				},
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func buildToolCallArgChunk(arguments string) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"tool_calls": []interface{}{
						map[string]interface{}{
							"index": 0,
							"function": map[string]interface{}{
								"arguments": arguments,
							},
						},
					},
				},
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func TestStreamTransform_BasicText(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildChatChunkWithRole("assistant")) +
		buildSSEChunk(buildChatChunk("Hello", nil)) +
		buildSSEChunk(buildChatChunk(" world", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(10, 5, 15)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Should contain text delta events
	assert.Contains(t, result, "response.output_text.delta")
	assert.Contains(t, result, "Hello")
	assert.Contains(t, result, " world")

	// Should contain completion events
	assert.Contains(t, result, "response.completed")
}

func TestStreamTransform_EventSequence(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildChatChunk("Hi", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(5, 2, 7)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Verify event ordering by finding positions
	events := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}

	lastPos := -1
	for _, event := range events {
		pos := strings.Index(result, "event: "+event+"\n")
		if pos == -1 {
			t.Errorf("event %q not found in output", event)
			continue
		}
		assert.Greater(t, pos, lastPos, "event %q should come after previous events", event)
		lastPos = pos
	}
}

func TestStreamTransform_Usage(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildChatChunk("test", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(100, 50, 150)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Find the response.completed event data
	completedIdx := strings.Index(result, "event: response.completed\n")
	require.NotEqual(t, -1, completedIdx)

	// Extract data line after the event line
	afterEvent := result[completedIdx:]
	dataIdx := strings.Index(afterEvent, "data: ")
	require.NotEqual(t, -1, dataIdx)

	dataLine := afterEvent[dataIdx+6:]
	endIdx := strings.Index(dataLine, "\n")
	if endIdx > 0 {
		dataLine = dataLine[:endIdx]
	}

	var completedEvent struct {
		Response struct {
			Usage map[string]interface{} `json:"usage"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal([]byte(dataLine), &completedEvent))

	assert.Equal(t, float64(100), completedEvent.Response.Usage["input_tokens"])
	assert.Equal(t, float64(50), completedEvent.Response.Usage["output_tokens"])
	assert.Equal(t, float64(150), completedEvent.Response.Usage["total_tokens"])
}

func TestStreamTransform_ConsistentMessageIDs(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildChatChunk("Hello", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(5, 2, 7)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Extract message ID from output_item.added event
	extractMsgID := func(eventName string) string {
		idx := strings.Index(result, "event: "+eventName+"\n")
		if idx == -1 {
			return ""
		}
		after := result[idx:]
		dataIdx := strings.Index(after, "data: ")
		if dataIdx == -1 {
			return ""
		}
		dataLine := after[dataIdx+6:]
		endIdx := strings.Index(dataLine, "\n")
		if endIdx > 0 {
			dataLine = dataLine[:endIdx]
		}
		var event map[string]interface{}
		if err := json.Unmarshal([]byte(dataLine), &event); err != nil {
			return ""
		}
		if item, ok := event["item"].(map[string]interface{}); ok {
			if id, ok := item["id"].(string); ok {
				return id
			}
		}
		if resp, ok := event["response"].(map[string]interface{}); ok {
			if output, ok := resp["output"].([]interface{}); ok && len(output) > 0 {
				if msg, ok := output[0].(map[string]interface{}); ok {
					if id, ok := msg["id"].(string); ok {
						return id
					}
				}
			}
		}
		return ""
	}

	addedID := extractMsgID("response.output_item.added")
	doneID := extractMsgID("response.output_item.done")
	completedID := extractMsgID("response.completed")

	require.NotEmpty(t, addedID, "output_item.added should have message ID")
	require.NotEmpty(t, doneID, "output_item.done should have message ID")
	require.NotEmpty(t, completedID, "response.completed should have message ID")

	// All three events must reference the same message ID
	assert.Equal(t, addedID, doneID, "output_item.added and output_item.done should have same ID")
	assert.Equal(t, addedID, completedID, "output_item.added and response.completed should have same ID")
}

func TestStreamTransform_NoDoneEmitsCompletion(t *testing.T) {
	// Stream without [DONE] should still emit completion events
	input := buildSSEChunk(buildChatChunk("Hello", nil))
	// No [DONE] — connection dropped

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Should still contain completion events
	assert.Contains(t, result, "response.created")
	assert.Contains(t, result, "response.completed")
	assert.Contains(t, result, "response.output_text.done")
}

func TestStreamTransform_ToolCall(t *testing.T) {
	// Build a tool call streaming sequence
	roleChunk := buildChatChunkWithRole("assistant")

	// Tool call start (with ID and name)
	tcStartChunk := buildToolCallStartChunk("call_abc", "get_weather")

	// Tool call argument delta
	tcArgChunk := buildToolCallArgChunk("{\"city\":\"Paris\"}")

	stopReason := "tool_calls"
	finishChunk := buildChatChunk("", &stopReason)

	input := buildSSEChunk(roleChunk) +
		buildSSEChunk(tcStartChunk) +
		buildSSEChunk(tcArgChunk) +
		buildSSEChunk(finishChunk) +
		buildSSEChunk(buildUsageChunk(10, 8, 18)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()

	// Should contain function call events
	assert.Contains(t, result, "response.output_item.added")
	assert.Contains(t, result, "response.function_call_arguments.delta")
	assert.Contains(t, result, "response.function_call_arguments.done")
	assert.Contains(t, result, "get_weather")
	assert.Contains(t, result, "call_abc")
	assert.Contains(t, result, "response.completed")
}
