package responses

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
)

// streamState tracks the current state of the Responses API stream transformer.
type streamState int

const (
	stateInitial streamState = iota
	stateStreamingText
	stateStreamingToolCall
)

// streamAccumulator accumulates data across streaming chunks.
type streamAccumulator struct {
	responseID string
	model      string
	createdAt  int64

	// Accumulated content
	fullText      string
	toolCalls     []accumulatedToolCall
	currentToolID int // index into toolCalls for the active tool call

	// Usage from final chunk
	usage *chatCompletionsUsage

	// State
	state streamState

	// Whether header events have been emitted
	headerEmitted bool
	// Whether a message output item has been started
	messageStarted bool
	// Stable message item ID (generated once, reused across all events)
	messageItemID string
	// Whether completion events have been emitted (via [DONE])
	completed bool
}

type accumulatedToolCall struct {
	id        string
	name      string
	arguments string
	itemID    string // Responses API item ID
}

// chatCompletionsUsage represents usage from a Chat Completions streaming chunk.
type chatCompletionsUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	ReasoningTokens  int
}

// chatStreamChunk represents a parsed Chat Completions streaming chunk.
type chatStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string `json:"role,omitempty"`
			Content   string `json:"content,omitempty"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function *struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens,omitempty"`
		} `json:"prompt_tokens_details,omitempty"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
		} `json:"completion_tokens_details,omitempty"`
	} `json:"usage,omitempty"`
}

// TransformChatStreamToResponses reads Chat Completions SSE from reader,
// transforms to Responses API SSE events, and writes to writer.
func TransformChatStreamToResponses(reader io.Reader, writer io.Writer, model string) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	acc := &streamAccumulator{
		responseID: generateResponseID(),
		model:      model,
	}

	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		if !strings.HasPrefix(line, "data: ") {
			if line != "" {
				slog.Debug("[responses/streaming] skipping non-data line",
					"line_num", lineCount, "line_prefix", truncate(line, 80), "line_len", len(line))
			}
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		slog.Debug("[responses/streaming] received SSE data line",
			"line_num", lineCount, "data_prefix", truncate(data, 200))

		if data == "[DONE]" {
			slog.Debug("[responses/streaming] received [DONE], emitting completion events",
				"has_usage", acc.usage != nil, "full_text_len", len(acc.fullText),
				"tool_calls", len(acc.toolCalls), "header_emitted", acc.headerEmitted,
				"message_started", acc.messageStarted)
			// Emit completion events
			if err := emitCompletionEvents(writer, acc); err != nil {
				return err
			}
			acc.completed = true
			break
		}

		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("[responses/streaming] failed to parse chunk JSON",
				"error", err, "data_prefix", truncate(data, 200))
			continue
		}

		slog.Debug("[responses/streaming] parsed chunk",
			"choices", len(chunk.Choices), "has_usage", chunk.Usage != nil,
			"model", chunk.Model, "id", chunk.ID)

		// Capture metadata from first chunk
		if acc.createdAt == 0 {
			acc.createdAt = chunk.Created
			if chunk.Model != "" {
				acc.model = chunk.Model
			}
		}

		// Capture usage if present
		if chunk.Usage != nil {
			acc.usage = &chatCompletionsUsage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
			if chunk.Usage.PromptTokensDetails != nil {
				acc.usage.CachedTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				acc.usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Handle finish_reason
		if choice.FinishReason != nil {
			slog.Debug("[responses/streaming] finish_reason received",
				"reason", *choice.FinishReason,
				"accumulated_text_len", len(acc.fullText),
				"tool_calls", len(acc.toolCalls))
			// The stream is ending; completion events will be emitted on [DONE]
			continue
		}

		// Handle text content delta
		if choice.Delta.Content != "" {
			slog.Debug("[responses/streaming] text delta",
				"content_len", len(choice.Delta.Content),
				"header_emitted", acc.headerEmitted,
				"message_started", acc.messageStarted)
			if !acc.headerEmitted {
				if err := emitHeaderEvents(writer, acc); err != nil {
					return err
				}
			}
			if !acc.messageStarted {
				if err := emitMessageStartEvents(writer, acc); err != nil {
					return err
				}
			}

			acc.fullText += choice.Delta.Content
			acc.state = stateStreamingText

			// Emit text delta
			deltaEvent := map[string]interface{}{
				"type":          "response.output_text.delta",
				"output_index":  0,
				"content_index": 0,
				"delta":         choice.Delta.Content,
			}
			if err := writeSSE(writer, "response.output_text.delta", deltaEvent); err != nil {
				return err
			}
		}

		// Handle tool call deltas
		for _, tc := range choice.Delta.ToolCalls {
			if !acc.headerEmitted {
				if err := emitHeaderEvents(writer, acc); err != nil {
					return err
				}
			}

			// New tool call (has ID)
			if tc.ID != "" {
				toolCall := accumulatedToolCall{
					id:     tc.ID,
					itemID: generateItemID("fc_"),
				}
				if tc.Function != nil {
					toolCall.name = tc.Function.Name
				}
				acc.toolCalls = append(acc.toolCalls, toolCall)
				acc.currentToolID = len(acc.toolCalls) - 1
				acc.state = stateStreamingToolCall

				// Emit output_item.added for function_call
				outputIndex := 0
				if acc.messageStarted {
					outputIndex = 1
				}
				outputIndex += tc.Index

				itemAddedEvent := map[string]interface{}{
					"type":         "response.output_item.added",
					"output_index": outputIndex,
					"item": map[string]interface{}{
						"type":      "function_call",
						"id":        toolCall.itemID,
						"call_id":   toolCall.id,
						"name":      toolCall.name,
						"arguments": "",
						"status":    "in_progress",
					},
				}
				if err := writeSSE(writer, "response.output_item.added", itemAddedEvent); err != nil {
					return err
				}
			}

			// Accumulate arguments
			if tc.Function != nil && tc.Function.Arguments != "" && len(acc.toolCalls) > 0 {
				idx := acc.currentToolID
				acc.toolCalls[idx].arguments += tc.Function.Arguments

				outputIndex := 0
				if acc.messageStarted {
					outputIndex = 1
				}
				outputIndex += tc.Index

				argDeltaEvent := map[string]interface{}{
					"type":         "response.function_call_arguments.delta",
					"output_index": outputIndex,
					"delta":        tc.Function.Arguments,
				}
				if err := writeSSE(writer, "response.function_call_arguments.delta", argDeltaEvent); err != nil {
					return err
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("[responses/streaming] scanner error", "error", err)
		return fmt.Errorf("scanner error: %w", err)
	}

	slog.Debug("[responses/streaming] scanner finished",
		"lines_read", lineCount, "completed", acc.completed,
		"header_emitted", acc.headerEmitted, "message_started", acc.messageStarted,
		"full_text_len", len(acc.fullText), "tool_calls", len(acc.toolCalls),
		"has_usage", acc.usage != nil)

	// If the stream ended without [DONE] (e.g., connection dropped),
	// still emit completion events so the client gets a proper ending.
	if acc.headerEmitted && !acc.completed {
		slog.Debug("[responses/streaming] stream ended without [DONE], emitting fallback completion")
		if err := emitCompletionEvents(writer, acc); err != nil {
			return err
		}
	}

	// If no header events were emitted, emit them now along with completion
	// to ensure the client receives a valid response even if no data was received
	if !acc.headerEmitted {
		slog.Warn("[responses/streaming] stream ended without emitting any events, emitting empty response",
			"lines_read", lineCount, "completed", acc.completed)
		if err := emitHeaderEvents(writer, acc); err != nil {
			return err
		}
		if err := emitCompletionEvents(writer, acc); err != nil {
			return err
		}
	}

	return nil
}

// truncate returns at most n bytes of s for safe debug logging.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// emitHeaderEvents emits the initial response.created and response.in_progress events.
func emitHeaderEvents(w io.Writer, acc *streamAccumulator) error {
	acc.headerEmitted = true

	respObj := buildInProgressResponse(acc)

	createdEvent := map[string]interface{}{
		"type":     "response.created",
		"response": respObj,
	}
	if err := writeSSE(w, "response.created", createdEvent); err != nil {
		return err
	}

	inProgressEvent := map[string]interface{}{
		"type":     "response.in_progress",
		"response": respObj,
	}
	return writeSSE(w, "response.in_progress", inProgressEvent)
}

// emitMessageStartEvents emits output_item.added and content_part.added for a message.
func emitMessageStartEvents(w io.Writer, acc *streamAccumulator) error {
	acc.messageStarted = true
	acc.messageItemID = generateItemID("msg_")

	msgItemID := acc.messageItemID

	itemAddedEvent := map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]interface{}{
			"type":    "message",
			"id":      msgItemID,
			"status":  "in_progress",
			"role":    "assistant",
			"content": []interface{}{},
		},
	}
	if err := writeSSE(w, "response.output_item.added", itemAddedEvent); err != nil {
		return err
	}

	contentPartEvent := map[string]interface{}{
		"type":          "response.content_part.added",
		"output_index":  0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type":        "output_text",
			"text":        "",
			"annotations": []interface{}{},
		},
	}
	return writeSSE(w, "response.content_part.added", contentPartEvent)
}

// emitCompletionEvents emits all closing events and the final response.completed.
func emitCompletionEvents(w io.Writer, acc *streamAccumulator) error {
	// Close text content if we were streaming text
	if acc.messageStarted {
		// output_text.done
		textDoneEvent := map[string]interface{}{
			"type":          "response.output_text.done",
			"output_index":  0,
			"content_index": 0,
			"text":          acc.fullText,
		}
		if err := writeSSE(w, "response.output_text.done", textDoneEvent); err != nil {
			return err
		}

		// content_part.done
		contentPartDoneEvent := map[string]interface{}{
			"type":          "response.content_part.done",
			"output_index":  0,
			"content_index": 0,
			"part": map[string]interface{}{
				"type":        "output_text",
				"text":        acc.fullText,
				"annotations": []interface{}{},
			},
		}
		if err := writeSSE(w, "response.content_part.done", contentPartDoneEvent); err != nil {
			return err
		}

		// output_item.done for message
		msgDoneEvent := map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": 0,
			"item": map[string]interface{}{
				"type":   "message",
				"id":     acc.messageItemID,
				"status": "completed",
				"role":   "assistant",
				"content": []interface{}{
					map[string]interface{}{
						"type":        "output_text",
						"text":        acc.fullText,
						"annotations": []interface{}{},
					},
				},
			},
		}
		if err := writeSSE(w, "response.output_item.done", msgDoneEvent); err != nil {
			return err
		}
	}

	// Close tool calls
	for i, tc := range acc.toolCalls {
		outputIndex := i
		if acc.messageStarted {
			outputIndex = i + 1
		}

		// function_call_arguments.done
		argsDoneEvent := map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"output_index": outputIndex,
			"arguments":    tc.arguments,
		}
		if err := writeSSE(w, "response.function_call_arguments.done", argsDoneEvent); err != nil {
			return err
		}

		// output_item.done for function_call
		fcDoneEvent := map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": outputIndex,
			"item": map[string]interface{}{
				"type":      "function_call",
				"id":        tc.itemID,
				"call_id":   tc.id,
				"name":      tc.name,
				"arguments": tc.arguments,
				"status":    "completed",
			},
		}
		if err := writeSSE(w, "response.output_item.done", fcDoneEvent); err != nil {
			return err
		}
	}

	// response.completed with full response object
	completedResp := buildCompletedResponse(acc)
	completedEvent := map[string]interface{}{
		"type":     "response.completed",
		"response": completedResp,
	}
	return writeSSE(w, "response.completed", completedEvent)
}

// buildInProgressResponse builds the response object for in-progress events.
func buildInProgressResponse(acc *streamAccumulator) map[string]interface{} {
	return map[string]interface{}{
		"id":                   acc.responseID,
		"object":               "response",
		"created_at":           acc.createdAt,
		"model":                acc.model,
		"status":               "in_progress",
		"output":               []interface{}{},
		"usage":                nil,
		"error":                nil,
		"incomplete_details":   nil,
		"metadata":             map[string]interface{}{},
		"tools":                []interface{}{},
		"parallel_tool_calls":  true,
		"instructions":         nil,
		"previous_response_id": nil,
		"store":                false,
	}
}

// buildCompletedResponse builds the full response object for the completed event.
func buildCompletedResponse(acc *streamAccumulator) map[string]interface{} {
	var output []interface{}

	if acc.messageStarted && acc.fullText != "" {
		output = append(output, map[string]interface{}{
			"type":   "message",
			"id":     acc.messageItemID,
			"status": "completed",
			"role":   "assistant",
			"content": []interface{}{
				map[string]interface{}{
					"type":        "output_text",
					"text":        acc.fullText,
					"annotations": []interface{}{},
				},
			},
		})
	}

	for _, tc := range acc.toolCalls {
		output = append(output, map[string]interface{}{
			"type":      "function_call",
			"id":        tc.itemID,
			"call_id":   tc.id,
			"name":      tc.name,
			"arguments": tc.arguments,
			"status":    "completed",
		})
	}

	var usageObj interface{}
	if acc.usage != nil {
		usageObj = map[string]interface{}{
			"input_tokens":  acc.usage.PromptTokens,
			"output_tokens": acc.usage.CompletionTokens,
			"total_tokens":  acc.usage.TotalTokens,
			"input_tokens_details": map[string]interface{}{
				"cached_tokens": acc.usage.CachedTokens,
			},
			"output_tokens_details": map[string]interface{}{
				"reasoning_tokens": acc.usage.ReasoningTokens,
			},
		}
	}

	return map[string]interface{}{
		"id":                   acc.responseID,
		"object":               "response",
		"created_at":           acc.createdAt,
		"model":                acc.model,
		"status":               "completed",
		"output":               output,
		"usage":                usageObj,
		"error":                nil,
		"incomplete_details":   nil,
		"metadata":             map[string]interface{}{},
		"tools":                []interface{}{},
		"parallel_tool_calls":  true,
		"instructions":         nil,
		"previous_response_id": nil,
		"store":                false,
	}
}

// writeSSE writes a single SSE event to the writer.
func writeSSE(w io.Writer, eventType string, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal SSE data: %w", err)
	}
	slog.Debug("[responses/streaming] writeSSE",
		"event", eventType, "data_len", len(jsonData))
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, jsonData)
	if err != nil {
		slog.Error("[responses/streaming] writeSSE failed",
			"event", eventType, "error", err)
	}
	return err
}
