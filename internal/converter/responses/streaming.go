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
	fullRefusal   string // accumulated refusal text
	toolCalls     []accumulatedToolCall
	currentToolID int // index into toolCalls for the active tool call

	// Usage from final chunk
	usage *chatCompletionsUsage

	// State
	state streamState

	// Finish reason status
	finishReason *string

	// Whether header events have been emitted
	headerEmitted bool
	// Whether a message output item has been started
	messageStarted bool
	// Stable message item ID (generated once, reused across all events)
	messageItemID string
	// Whether completion events have been emitted (via [DONE])
	completed bool
	// Monotonic sequence counter for SSE events (required by OpenAI SDK)
	sequenceNumber int

	// Original Chat Completions ID for correlation
	originalChatCompletionID string

	// Request-echoed metadata (populated from ResponsesMetadata when available)
	storeFlag          bool
	previousResponseID string
	requestMetadata    map[string]string
}

type accumulatedToolCall struct {
	id        string
	name      string
	arguments string
	itemID    string // Responses API item ID
}

// chatCompletionsUsage represents usage from a Chat Completions streaming chunk.
type chatCompletionsUsage struct {
	PromptTokens      int
	CompletionTokens  int
	TotalTokens       int
	CachedTokens      int
	ReasoningTokens   int
	AudioInputTokens  int
	AudioOutputTokens int
	ImageOutputTokens int
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
			Refusal   string `json:"refusal,omitempty"`
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
			AudioTokens  int `json:"audio_tokens,omitempty"`
		} `json:"prompt_tokens_details,omitempty"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
			AudioTokens     int `json:"audio_tokens,omitempty"`
			ImageTokens     int `json:"image_tokens,omitempty"`
		} `json:"completion_tokens_details,omitempty"`
	} `json:"usage,omitempty"`
}

// TransformChatStreamToResponses reads Chat Completions SSE from reader,
// transforms to Responses API SSE events, and writes to writer.
// The optional onComplete callback is invoked with the fully-built Response
// once the stream is complete (on [DONE] or stream end).
// Pass a *ResponsesMetadata as the second variadic element to have store,
// previous_response_id and metadata echoed back in all SSE response objects.
func TransformChatStreamToResponses(reader io.Reader, writer io.Writer, model string, onComplete ...func(*Response)) error {
	return transformChatStreamToResponsesInner(reader, writer, model, nil, onComplete...)
}

// TransformChatStreamToResponsesWithMeta is like TransformChatStreamToResponses but
// additionally echoes request-side fields (store, previous_response_id, metadata) into
// every emitted response object so the wire payload matches the stored record.
func TransformChatStreamToResponsesWithMeta(reader io.Reader, writer io.Writer, model string, meta *ResponsesMetadata, onComplete ...func(*Response)) error {
	return transformChatStreamToResponsesInner(reader, writer, model, meta, onComplete...)
}

func transformChatStreamToResponsesInner(reader io.Reader, writer io.Writer, model string, meta *ResponsesMetadata, onComplete ...func(*Response)) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	acc := &streamAccumulator{
		responseID: GenerateResponseID(),
		model:      model,
	}
	if meta != nil {
		acc.storeFlag = meta.Store
		acc.previousResponseID = meta.PreviousResponseID
		acc.requestMetadata = meta.Metadata
	}

	callOnComplete := func() {
		if len(onComplete) > 0 && onComplete[0] != nil {
			onComplete[0](buildTypedCompletedResponse(acc))
		}
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
			callOnComplete()
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
			if chunk.ID != "" {
				acc.originalChatCompletionID = chunk.ID
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
				acc.usage.AudioInputTokens = chunk.Usage.PromptTokensDetails.AudioTokens
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				acc.usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
				acc.usage.AudioOutputTokens = chunk.Usage.CompletionTokensDetails.AudioTokens
				acc.usage.ImageOutputTokens = chunk.Usage.CompletionTokensDetails.ImageTokens
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Handle text content delta BEFORE finish_reason.
		// Some providers (Vertex with GoogleSearch, short responses) send both
		// content and finish_reason in the same chunk. Processing finish_reason
		// first with `continue` would skip the content entirely.
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
				"item_id":       acc.messageItemID,
				"output_index":  0,
				"content_index": 0,
				"delta":         choice.Delta.Content,
			}
			if err := writeSSEWithSeq(writer, "response.output_text.delta", deltaEvent, acc); err != nil {
				return err
			}
		}

		// handle refusal deltas
		if choice.Delta.Refusal != "" {
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
			acc.fullRefusal += choice.Delta.Refusal
			refusalEvent := map[string]interface{}{
				"type":          "response.refusal.delta",
				"item_id":       acc.messageItemID,
				"output_index":  0,
				"content_index": 0,
				"delta":         choice.Delta.Refusal,
			}
			if err := writeSSEWithSeq(writer, "response.refusal.delta", refusalEvent, acc); err != nil {
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
				idx := tc.Index
				for len(acc.toolCalls) <= idx {
					acc.toolCalls = append(acc.toolCalls, accumulatedToolCall{})
				}
				toolCall := accumulatedToolCall{
					id:     tc.ID,
					itemID: GenerateItemID("fc_"),
				}
				if tc.Function != nil {
					toolCall.name = tc.Function.Name
				}
				acc.toolCalls[idx] = toolCall
				acc.currentToolID = idx
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
						"id":        acc.toolCalls[idx].itemID,
						"call_id":   acc.toolCalls[idx].id,
						"name":      acc.toolCalls[idx].name,
						"arguments": "",
						"status":    "in_progress",
					},
				}
				if err := writeSSEWithSeq(writer, "response.output_item.added", itemAddedEvent, acc); err != nil {
					return err
				}
			}

			// Accumulate arguments
			if tc.Function != nil && tc.Function.Arguments != "" {
				idx := tc.Index
				if idx >= len(acc.toolCalls) {
					for len(acc.toolCalls) <= idx {
						acc.toolCalls = append(acc.toolCalls, accumulatedToolCall{})
					}
					acc.toolCalls[idx].itemID = GenerateItemID("fc_")
				}
				acc.toolCalls[idx].arguments += tc.Function.Arguments

				outputIndex := 0
				if acc.messageStarted {
					outputIndex = 1
				}
				outputIndex += tc.Index

				argDeltaEvent := map[string]interface{}{
					"type":         "response.function_call_arguments.delta",
					"item_id":      acc.toolCalls[idx].itemID,
					"output_index": outputIndex,
					"delta":        tc.Function.Arguments,
				}
				if err := writeSSEWithSeq(writer, "response.function_call_arguments.delta", argDeltaEvent, acc); err != nil {
					return err
				}
			}
		}

		// Handle finish_reason AFTER content and tool_calls so that chunks
		// carrying both content and finish_reason are fully processed.
		if choice.FinishReason != nil {
			slog.Debug("[responses/streaming] finish_reason received",
				"reason", *choice.FinishReason,
				"accumulated_text_len", len(acc.fullText),
				"tool_calls", len(acc.toolCalls))
			acc.finishReason = choice.FinishReason
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
		callOnComplete()
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
		callOnComplete()
	}

	return nil
}

// buildTypedCompletedResponse builds a typed *Response from the stream accumulator.
func buildTypedCompletedResponse(acc *streamAccumulator) *Response {
	var output []OutputItem

	if acc.messageStarted && acc.fullText != "" {
		output = append(output, OutputItem{
			Type:   "message",
			ID:     acc.messageItemID,
			Status: "completed",
			Role:   "assistant",
			Content: []OutputContent{{
				Type:        "output_text",
				Text:        acc.fullText,
				Annotations: []Annotation{},
			}},
		})
	}

	for _, tc := range acc.toolCalls {
		output = append(output, OutputItem{
			Type:      "function_call",
			ID:        tc.itemID,
			Status:    "completed",
			CallID:    tc.id,
			Name:      tc.name,
			Arguments: tc.arguments,
		})
	}

	var usage *Usage
	if acc.usage != nil {
		usage = &Usage{
			InputTokens:         acc.usage.PromptTokens,
			OutputTokens:        acc.usage.CompletionTokens,
			TotalTokens:         acc.usage.TotalTokens,
			InputTokensDetails:  InputDetails{CachedTokens: acc.usage.CachedTokens, AudioTokens: acc.usage.AudioInputTokens},
			OutputTokensDetails: OutputDetails{ReasoningTokens: acc.usage.ReasoningTokens, AudioTokens: acc.usage.AudioOutputTokens, ImageTokens: acc.usage.ImageOutputTokens},
		}
	}

	status := "completed"
	var incompleteDetails *IncompleteDetails
	if acc.finishReason != nil {
		switch *acc.finishReason {
		case "length":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "max_output_tokens"}
		case "content_filter":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "content_filter"}
		}
	}

	metadata := acc.requestMetadata
	var prevRespID interface{}
	if acc.previousResponseID != "" {
		prevRespID = acc.previousResponseID
	}

	return BuildCompletedResponse(CompletedResponseParams{
		ID:                 acc.responseID,
		Model:              acc.model,
		CreatedAt:          acc.createdAt,
		Status:             status,
		IncompleteDetails:  incompleteDetails,
		Output:             output,
		Usage:              usage,
		Metadata:           metadata,
		PreviousResponseID: prevRespID,
		Store:              acc.storeFlag,
		ToolChoice:         "auto",
	})
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

	createdEvent := BuildResponseEvent("response.created", respObj)
	if err := writeSSEWithSeq(w, "response.created", createdEvent, acc); err != nil {
		return err
	}

	inProgressEvent := BuildResponseEvent("response.in_progress", respObj)
	return writeSSEWithSeq(w, "response.in_progress", inProgressEvent, acc)
}

// emitMessageStartEvents emits output_item.added and content_part.added for a message.
func emitMessageStartEvents(w io.Writer, acc *streamAccumulator) error {
	acc.messageStarted = true
	acc.messageItemID = GenerateItemID("msg_")

	msgItemID := acc.messageItemID

	itemAddedEvent := BuildMessageItemAddedEvent(0, msgItemID)
	if err := writeSSEWithSeq(w, "response.output_item.added", itemAddedEvent, acc); err != nil {
		return err
	}

	contentPartEvent := BuildContentPartAddedEvent(acc.messageItemID, 0, 0)
	return writeSSEWithSeq(w, "response.content_part.added", contentPartEvent, acc)
}

// emitCompletionEvents emits all closing events and the final response.completed.
func emitCompletionEvents(w io.Writer, acc *streamAccumulator) error {
	// only emit text closing events if there's actual text content.
	// This matches the condition in buildCompletedResponse (messageStarted && fullText != "").
	if acc.messageStarted && acc.fullText != "" {
		// output_text.done
		textDoneEvent := BuildOutputTextDoneEvent(acc.messageItemID, 0, 0, acc.fullText)
		if err := writeSSEWithSeq(w, "response.output_text.done", textDoneEvent, acc); err != nil {
			return err
		}

		// content_part.done
		contentPartDoneEvent := BuildContentPartDoneEvent(acc.messageItemID, 0, 0, acc.fullText)
		if err := writeSSEWithSeq(w, "response.content_part.done", contentPartDoneEvent, acc); err != nil {
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
		if err := writeSSEWithSeq(w, "response.output_item.done", msgDoneEvent, acc); err != nil {
			return err
		}
	}

	// Close refusal part if any refusal text was accumulated
	if acc.messageStarted && acc.fullRefusal != "" {
		refusalDoneEvent := map[string]interface{}{
			"type":          "response.refusal.done",
			"item_id":       acc.messageItemID,
			"output_index":  0,
			"content_index": 0,
			"refusal":       acc.fullRefusal,
		}
		if err := writeSSEWithSeq(w, "response.refusal.done", refusalDoneEvent, acc); err != nil {
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
			"item_id":      tc.itemID,
			"output_index": outputIndex,
			"arguments":    tc.arguments,
		}
		if err := writeSSEWithSeq(w, "response.function_call_arguments.done", argsDoneEvent, acc); err != nil {
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
		if err := writeSSEWithSeq(w, "response.output_item.done", fcDoneEvent, acc); err != nil {
			return err
		}
	}

	// emit correct event type based on finish reason
	completedResp := buildCompletedResponse(acc)
	var eventType string
	status, _ := completedResp["status"].(string)

	switch status {
	case "incomplete":
		eventType = "response.incomplete"
	case "failed":
		eventType = "response.failed"
	default:
		eventType = "response.completed"
	}

	completedEvent := BuildResponseEvent(eventType, completedResp)
	return writeSSEWithSeq(w, eventType, completedEvent, acc)
}

// buildInProgressResponse builds the response object for in-progress events.
func buildInProgressResponse(acc *streamAccumulator) map[string]interface{} {
	status := "in_progress"
	var incompleteDetails *IncompleteDetails
	if acc.finishReason != nil {
		switch *acc.finishReason {
		case "length":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "max_output_tokens"}
		case "content_filter":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "content_filter"}
		}
	}
	var prevRespID interface{}
	if acc.previousResponseID != "" {
		prevRespID = acc.previousResponseID
	}
	return ResponseToMap(NewResponse(ResponseParams{
		ID:                 acc.responseID,
		Model:              acc.model,
		CreatedAt:          acc.createdAt,
		Status:             status,
		IncompleteDetails:  incompleteDetails,
		Metadata:           acc.requestMetadata,
		PreviousResponseID: prevRespID,
		Store:              acc.storeFlag,
		ToolChoice:         "auto",
	}))
}

// buildCompletedResponse builds the full response object for the completed event.
func buildCompletedResponse(acc *streamAccumulator) map[string]interface{} {
	output := make([]OutputItem, 0)

	if acc.messageStarted && acc.fullText != "" {
		output = append(output, OutputItem{
			Type:   "message",
			ID:     acc.messageItemID,
			Status: "completed",
			Role:   "assistant",
			Content: []OutputContent{{
				Type:        "output_text",
				Text:        acc.fullText,
				Annotations: []Annotation{},
			}},
		})
	}

	for _, tc := range acc.toolCalls {
		output = append(output, OutputItem{
			Type:      "function_call",
			ID:        tc.itemID,
			CallID:    tc.id,
			Name:      tc.name,
			Arguments: tc.arguments,
			Status:    "completed",
		})
	}

	var usageObj *Usage
	if acc.usage != nil {
		usageObj = &Usage{
			InputTokens:  acc.usage.PromptTokens,
			OutputTokens: acc.usage.CompletionTokens,
			TotalTokens:  acc.usage.TotalTokens,
			InputTokensDetails: InputDetails{
				CachedTokens: acc.usage.CachedTokens,
				AudioTokens:  acc.usage.AudioInputTokens,
			},
			OutputTokensDetails: OutputDetails{
				ReasoningTokens: acc.usage.ReasoningTokens,
				AudioTokens:     acc.usage.AudioOutputTokens,
				ImageTokens:     acc.usage.ImageOutputTokens,
			},
		}
	}

	status := "completed"
	var incompleteDetails *IncompleteDetails
	if acc.finishReason != nil {
		switch *acc.finishReason {
		case "length":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "max_output_tokens"}
		case "content_filter":
			status = "incomplete"
			incompleteDetails = &IncompleteDetails{Reason: "content_filter"}
		}
	}

	var prevRespID interface{}
	if acc.previousResponseID != "" {
		prevRespID = acc.previousResponseID
	}
	return ResponseToMap(NewResponse(ResponseParams{
		ID:                 acc.responseID,
		Model:              acc.model,
		CreatedAt:          acc.createdAt,
		Status:             status,
		IncompleteDetails:  incompleteDetails,
		Output:             output,
		Usage:              usageObj,
		Metadata:           acc.requestMetadata,
		PreviousResponseID: prevRespID,
		Store:              acc.storeFlag,
		ToolChoice:         "auto",
	}))
}

// writeSSEWithSeq writes a single SSE event to the writer, injecting the
// monotonic sequence_number required by the OpenAI Python SDK.
func writeSSEWithSeq(w io.Writer, eventType string, data map[string]interface{}, acc *streamAccumulator) error {
	acc.sequenceNumber++
	data["sequence_number"] = acc.sequenceNumber
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal SSE data: %w", err)
	}
	slog.Debug("[responses/streaming] writeSSE",
		"event", eventType, "data_len", len(jsonData), "seq", acc.sequenceNumber)
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, jsonData)
	if err != nil {
		slog.Error("[responses/streaming] writeSSE failed",
			"event", eventType, "error", err)
	}
	return err
}
