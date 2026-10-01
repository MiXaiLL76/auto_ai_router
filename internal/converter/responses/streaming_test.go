package responses

import (
	"bytes"
	"encoding/base64"
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

func buildToolCallStartChunkWithIndex(callID, name string, index int) string {
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
							"index": index,
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

func buildToolCallArgChunkWithIndex(arguments string, index int) string {
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
							"index": index,
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

// parseSSEEvents splits raw SSE output into individual events, each decoded
// as a map with its event name stored under "_event".
func parseSSEEvents(t *testing.T, raw string) []map[string]interface{} {
	t.Helper()
	var events []map[string]interface{}
	for _, block := range strings.Split(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		lines := strings.SplitN(block, "\n", 2)
		require.Len(t, lines, 2, "malformed SSE block: %q", block)
		eventName := strings.TrimPrefix(lines[0], "event: ")
		dataLine := strings.TrimPrefix(lines[1], "data: ")
		var decoded map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(dataLine), &decoded), "block: %q", block)
		decoded["_event"] = eventName
		events = append(events, decoded)
	}
	return events
}

func buildReasoningChunk(reasoningContent string) string {
	chunk := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "deepseek/deepseek-v4.1-flash",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"reasoning_content": reasoningContent,
				},
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}

func TestStreamTransform_ReasoningContent(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildReasoningChunk("Let me think. ")) +
		buildSSEChunk(buildReasoningChunk("The answer is ok.")) +
		buildSSEChunk(buildChatChunk("Ok.", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(10, 8, 18)) +
		"data: [DONE]\n\n"

	var capturedResp *Response
	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "deepseek/deepseek-v4.1-flash",
		func(r *Response) { capturedResp = r })
	require.NoError(t, err)

	events := parseSSEEvents(t, output.String())

	itemType := func(e map[string]interface{}) string {
		item, ok := e["item"].(map[string]interface{})
		if !ok {
			return ""
		}
		t, _ := item["type"].(string)
		return t
	}

	var reasoningAddedIdx, messageAddedIdx = -1, -1
	for i, e := range events {
		switch {
		case e["_event"] == "response.output_item.added" && itemType(e) == "reasoning":
			reasoningAddedIdx = i
			assert.Equal(t, float64(0), e["output_index"])
		case e["_event"] == "response.output_item.added" && itemType(e) == "message":
			messageAddedIdx = i
			assert.Equal(t, float64(1), e["output_index"], "message should be shifted to output_index 1 by the reasoning item")
		}
	}
	require.NotEqual(t, -1, reasoningAddedIdx, "reasoning output_item.added not found")
	require.NotEqual(t, -1, messageAddedIdx, "message output_item.added not found")
	assert.Less(t, reasoningAddedIdx, messageAddedIdx, "reasoning item should be announced before the message item")

	require.NotNil(t, capturedResp)
	require.Len(t, capturedResp.Output, 2)
	assert.Equal(t, "reasoning", capturedResp.Output[0].Type)
	require.Len(t, capturedResp.Output[0].Summary, 1)
	assert.Equal(t, "Let me think. The answer is ok.", capturedResp.Output[0].Summary[0].Text)
	assert.Equal(t, "message", capturedResp.Output[1].Type)
	assert.Equal(t, "Ok.", capturedResp.Output[1].Content[0].Text)
}

// Covers the case where the entire max_output_tokens budget is consumed by
// reasoning before any visible content streams — the reasoning item must
// still be closed and surfaced, instead of leaving a dangling "added" event
// or dropping the reasoning text entirely.
func TestStreamTransform_ReasoningOnly_NoVisibleContent(t *testing.T) {
	lengthReason := "length"

	input := buildSSEChunk(buildReasoningChunk("Thinking very hard...")) +
		buildSSEChunk(buildChatChunk("", &lengthReason)) +
		buildSSEChunk(buildUsageChunk(10, 50, 60)) +
		"data: [DONE]\n\n"

	var capturedResp *Response
	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "deepseek/deepseek-v4.1-flash",
		func(r *Response) { capturedResp = r })
	require.NoError(t, err)

	events := parseSSEEvents(t, output.String())

	var addedCount, doneCount int
	for _, e := range events {
		item, ok := e["item"].(map[string]interface{})
		if !ok || item["type"] != "reasoning" {
			continue
		}
		assert.Equal(t, float64(0), e["output_index"])
		switch e["_event"] {
		case "response.output_item.added":
			addedCount++
			assert.Equal(t, "in_progress", item["status"])
		case "response.output_item.done":
			doneCount++
			assert.Equal(t, "completed", item["status"])
		}
	}
	assert.Equal(t, 1, addedCount, "reasoning item should be added exactly once")
	assert.Equal(t, 1, doneCount, "reasoning item should be closed exactly once (safety net at completion)")

	require.NotNil(t, capturedResp)
	require.Len(t, capturedResp.Output, 1)
	assert.Equal(t, "reasoning", capturedResp.Output[0].Type)
	assert.Equal(t, "Thinking very hard...", capturedResp.Output[0].Summary[0].Text)
}

func TestStreamTransform_ReasoningThenToolCall(t *testing.T) {
	stopReason := "tool_calls"

	input := buildSSEChunk(buildReasoningChunk("I should call the weather tool.")) +
		buildSSEChunk(buildToolCallStartChunk("call_abc", "get_weather")) +
		buildSSEChunk(buildToolCallArgChunk(`{"city":"Paris"}`)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		"data: [DONE]\n\n"

	var capturedResp *Response
	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "deepseek/deepseek-v4.1-flash",
		func(r *Response) { capturedResp = r })
	require.NoError(t, err)

	events := parseSSEEvents(t, output.String())

	var found bool
	for _, e := range events {
		item, ok := e["item"].(map[string]interface{})
		if !ok || item["type"] != "function_call" || e["_event"] != "response.output_item.added" {
			continue
		}
		found = true
		// Tool call has no preceding message, so with reasoning at index 0
		// the function_call must be shifted to output_index 1 (not 0).
		assert.Equal(t, float64(1), e["output_index"])
	}
	assert.True(t, found, "function_call output_item.added not found")

	require.NotNil(t, capturedResp)
	require.Len(t, capturedResp.Output, 2)
	assert.Equal(t, "reasoning", capturedResp.Output[0].Type)
	assert.Equal(t, "function_call", capturedResp.Output[1].Type)
	assert.Equal(t, "call_abc", capturedResp.Output[1].CallID)
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

// A chat chunk can carry a generated image as one base64 blob in delta.images,
// which pushes a single SSE line past 1 MiB. The scanner used to cap tokens at
// 1 MiB and aborted the whole stream with "bufio.Scanner: token too long", so
// the text and usage that followed the image never reached the client either.
func TestStreamTransform_OversizedImageLineDoesNotAbortStream(t *testing.T) {
	stopReason := "stop"
	imageURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0xab}, 900*1024))
	imageChunk, err := json.Marshal(map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"delta": map[string]interface{}{
					"images": []interface{}{
						map[string]interface{}{
							"type":      "image_url",
							"index":     0,
							"image_url": map[string]interface{}{"url": imageURL},
						},
					},
				},
				"finish_reason": nil,
			},
		},
	})
	require.NoError(t, err)
	require.Greater(t, len(imageChunk), 1024*1024)

	input := buildSSEChunk(buildChatChunkWithRole("assistant")) +
		buildSSEChunk(string(imageChunk)) +
		buildSSEChunk(buildChatChunk("after image", &stopReason)) +
		buildSSEChunk(buildUsageChunk(10, 5, 15)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o"))

	result := output.String()
	assert.Contains(t, result, "after image")
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
	usageChunk := `{"id":"chatcmpl-test","object":"chat.completion.chunk","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":20,"cached_audio_tokens":5,"audio_tokens":12,"cache_write_tokens":15,"cache_creation_token_details":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":11}},"completion_tokens_details":{"audio_tokens":3}}}`

	input := buildSSEChunk(buildChatChunk("test", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(usageChunk) +
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
	details := completedEvent.Response.Usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(20), details["cached_tokens"])
	assert.Equal(t, float64(5), details["cached_audio_tokens"])
	assert.Equal(t, float64(7), details["audio_tokens"])
	assert.Equal(t, float64(15), details["cache_creation_tokens"])
	ttlDetails := details["cache_creation_token_details"].(map[string]interface{})
	assert.Equal(t, float64(4), ttlDetails["ephemeral_5m_input_tokens"])
	assert.Equal(t, float64(11), ttlDetails["ephemeral_1h_input_tokens"])
	outputDetails := completedEvent.Response.Usage["output_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(3), outputDetails["audio_tokens"])
}

func TestStreamTransform_UsageSanitizesCachedAudioFields(t *testing.T) {
	stopReason := "stop"
	usageChunk := `{"id":"chatcmpl-test","object":"chat.completion.chunk","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":200,"completion_tokens":1,"total_tokens":201,"prompt_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}`

	input := buildSSEChunk(buildChatChunk("test", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(usageChunk) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()
	completedIdx := strings.Index(result, "event: response.completed\n")
	require.NotEqual(t, -1, completedIdx)

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

	details := completedEvent.Response.Usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(0), details["cached_tokens"])
	assert.NotContains(t, details, "cached_audio_tokens")
	assert.Equal(t, float64(100), details["audio_tokens"])
}

func TestStreamTransform_UsagePreservesNormalizedAudioInput(t *testing.T) {
	stopReason := "stop"
	usageChunk := `{"id":"chatcmpl-test","object":"chat.completion.chunk","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":200,"completion_tokens":1,"total_tokens":201,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":60}}}`
	input := buildSSEChunk(buildChatChunk("test", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(usageChunk) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponsesWithMetaAndUsage(
		strings.NewReader(input), &output, "gpt-4o", nil, false,
	)
	require.NoError(t, err)

	result := output.String()
	completedIdx := strings.Index(result, "event: response.completed\n")
	require.NotEqual(t, -1, completedIdx)
	afterEvent := result[completedIdx:]
	dataIdx := strings.Index(afterEvent, "data: ")
	require.NotEqual(t, -1, dataIdx)
	dataLine := afterEvent[dataIdx+6:]
	if endIdx := strings.Index(dataLine, "\n"); endIdx > 0 {
		dataLine = dataLine[:endIdx]
	}

	var completedEvent struct {
		Response struct {
			Usage map[string]interface{} `json:"usage"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal([]byte(dataLine), &completedEvent))
	details := completedEvent.Response.Usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(80), details["cached_tokens"])
	assert.Equal(t, float64(40), details["cached_audio_tokens"])
	assert.Equal(t, float64(60), details["audio_tokens"])
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

func TestStreamTransform_ResponseEventsIncludeRequiredSchemaFields(t *testing.T) {
	stopReason := "stop"

	input := buildSSEChunk(buildChatChunk("Hello", nil)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		buildSSEChunk(buildUsageChunk(5, 2, 7)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()
	for _, eventName := range []string{"response.created", "response.in_progress", "response.completed"} {
		idx := strings.Index(result, "event: "+eventName+"\n")
		require.NotEqual(t, -1, idx, "missing event %s", eventName)

		afterEvent := result[idx:]
		dataIdx := strings.Index(afterEvent, "data: ")
		require.NotEqual(t, -1, dataIdx)

		dataLine := afterEvent[dataIdx+6:]
		endIdx := strings.Index(dataLine, "\n")
		require.Greater(t, endIdx, 0)
		dataLine = dataLine[:endIdx]

		var event map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(dataLine), &event))
		_, hasSeq := event["sequence_number"]
		assert.True(t, hasSeq)

		resp, ok := event["response"].(map[string]interface{})
		require.True(t, ok, "event %s must include response object", eventName)
		for _, key := range []string{
			"completed_at", "truncation", "text", "top_p", "temperature",
			"service_tier", "presence_penalty", "frequency_penalty",
			"top_logprobs", "reasoning", "max_output_tokens", "max_tool_calls",
			"background", "safety_identifier", "prompt_cache_key",
		} {
			_, ok := resp[key]
			assert.True(t, ok, "response.%s must be present in %s", key, eventName)
		}
	}
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

func TestStreamTransform_ToolCall_Interleaved(t *testing.T) {
	roleChunk := buildChatChunkWithRole("assistant")
	tc1Start := buildToolCallStartChunkWithIndex("call_1", "fn1", 0)
	tc2Start := buildToolCallStartChunkWithIndex("call_2", "fn2", 1)
	tc1Arg1 := buildToolCallArgChunkWithIndex("{\"a\":", 0)
	tc2Arg1 := buildToolCallArgChunkWithIndex("{\"b\":", 1)
	tc1Arg2 := buildToolCallArgChunkWithIndex("1}", 0)
	tc2Arg2 := buildToolCallArgChunkWithIndex("2}", 1)
	stopReason := "tool_calls"
	finishChunk := buildChatChunk("", &stopReason)

	input := buildSSEChunk(roleChunk) +
		buildSSEChunk(tc1Start) +
		buildSSEChunk(tc2Start) +
		buildSSEChunk(tc1Arg1) +
		buildSSEChunk(tc2Arg1) +
		buildSSEChunk(tc1Arg2) +
		buildSSEChunk(tc2Arg2) +
		buildSSEChunk(finishChunk) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()
	assert.Contains(t, result, "\"call_id\":\"call_1\"")
	assert.Contains(t, result, "\"call_id\":\"call_2\"")
	assert.Contains(t, result, "{\\\"a\\\":1}")
	assert.Contains(t, result, "{\\\"b\\\":2}")
}

func TestStreamTransform_IncompleteFinishReason(t *testing.T) {
	lengthReason := "length"
	input := buildSSEChunk(buildChatChunk("Hello", nil)) +
		buildSSEChunk(buildChatChunk("", &lengthReason)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()
	assert.Contains(t, result, "\"status\":\"incomplete\"")
	assert.Contains(t, result, "\"incomplete_details\":{\"reason\":\"max_output_tokens\"}")
}

func TestStreamTransform_ContentFilterFinishReason(t *testing.T) {
	filterReason := "content_filter"
	input := buildSSEChunk(buildChatChunk("Hello", nil)) +
		buildSSEChunk(buildChatChunk("", &filterReason)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gpt-4o")
	require.NoError(t, err)

	result := output.String()
	assert.Contains(t, result, "\"status\":\"incomplete\"")
	assert.Contains(t, result, "\"incomplete_details\":{\"reason\":\"content_filter\"}")
}

// TestStreamTransform_ContentAndFinishReasonSameChunk verifies that when content
// and finish_reason arrive in the same chunk (common with Vertex GoogleSearch),
// the content is still processed and emitted.
// Regression: finish_reason was checked first with `continue`, skipping content.
func TestStreamTransform_ContentAndFinishReasonSameChunk(t *testing.T) {
	stopReason := "stop"

	// Simulate Vertex GoogleSearch response: role-only chunk, then content+stop in one chunk
	input := buildSSEChunk(buildChatChunkWithRole("assistant")) +
		buildSSEChunk(buildChatChunk("Search result: Tokyo population is 14 million", &stopReason)) +
		buildSSEChunk(buildUsageChunk(36, 67, 103)) +
		"data: [DONE]\n\n"

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gemini-2.5-flash")
	require.NoError(t, err)

	result := output.String()
	// Must contain the actual text content (not an empty response)
	assert.Contains(t, result, "Search result: Tokyo population is 14 million",
		"Content must be emitted even when finish_reason is in the same chunk")
	assert.Contains(t, result, "response.output_text.delta")
	assert.Contains(t, result, "response.output_text.done")
	assert.Contains(t, result, "response.completed")
	assert.Contains(t, result, "\"status\":\"completed\"")
}

// TestStreamTransform_ContentAndFinishReasonNoDone verifies the same scenario
// but without [DONE] (Gemini API doesn't send [DONE], stream just closes).
func TestStreamTransform_ContentAndFinishReasonNoDone(t *testing.T) {
	stopReason := "stop"

	// No [DONE] — stream ends when connection closes
	input := buildSSEChunk(buildChatChunkWithRole("assistant")) +
		buildSSEChunk(buildChatChunk("The answer is 42.", &stopReason)) +
		buildSSEChunk(buildUsageChunk(10, 20, 30))

	var output bytes.Buffer
	err := TransformChatStreamToResponses(strings.NewReader(input), &output, "gemini-2.5-flash")
	require.NoError(t, err)

	result := output.String()
	assert.Contains(t, result, "The answer is 42.")
	assert.Contains(t, result, "response.output_text.delta")
	assert.Contains(t, result, "response.completed")
}
