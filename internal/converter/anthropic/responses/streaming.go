package anthropicresponses

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// anthropicStreamAccumulator tracks state across Anthropic SSE streaming events.
type anthropicStreamAccumulator struct {
	responseID string
	model      string
	createdAt  int64
	meta       *responses.ResponsesMetadata

	// Accumulated content by block index
	currentBlockType   string
	currentBlockID     string
	currentBlockName   string
	currentText        string
	currentThinking    string
	currentToolArgs    string
	currentReasoningID string // ID assigned at content_block_start for "thinking"

	// Completed output items
	msgContent  []responses.OutputContent
	outputItems []responses.OutputItem
	// reasoningItem *responses.OutputItem

	// Usage
	inputTokens           int
	outputTokens          int
	cachedTokens          int
	cacheCreationTokens   int
	cacheCreation5mTokens int
	cacheCreation1hTokens int

	// Stream status
	stopReason         string
	headerEmitted      bool
	messageStarted     bool
	messageItemID      string
	messageOutputIndex int // output_index where the message item was placed

	// Current tool_use block state (set at content_block_start, used through block stop)
	currentToolItemID      string // pre-generated fc_ ID for output_item.added / done consistency
	currentToolOutputIndex int    // output_index at which the tool item was announced
	sequenceNumber         int
}

// TransformAnthropicStreamToResponses reads Anthropic SSE and writes Responses API SSE events.
// onComplete is called with the fully-built Response once the stream ends.
func TransformAnthropicStreamToResponses(
	reader io.Reader,
	writer io.Writer,
	model, responseID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	if responseID == "" {
		responseID = generateResponseID()
	}
	acc := &anthropicStreamAccumulator{
		responseID: responseID,
		model:      model,
		createdAt:  time.Now().Unix(),
		meta:       meta,
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event anthropic.AnthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			slog.Debug("[anthropicresponses/streaming] failed to parse event", "error", err)
			continue
		}

		if err := processAnthropicEvent(writer, acc, &event); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return fmt.Errorf("stream read error: %w", err)
	}

	if err := emitAnthropicCompletionEvents(writer, acc); err != nil {
		return err
	}

	if onComplete != nil {
		onComplete(buildAnthropicCompletedResponse(acc))
	}
	return nil
}

func processAnthropicEvent(w io.Writer, acc *anthropicStreamAccumulator, event *anthropic.AnthropicStreamEvent) error {
	switch event.Type {
	case "message_start":
		if event.Message != nil {
			acc.inputTokens = event.Message.Usage.InputTokens
			acc.cachedTokens = event.Message.Usage.CacheReadInputTokens
			acc.cacheCreationTokens, acc.cacheCreation5mTokens, acc.cacheCreation1hTokens = anthropic.NormalizeCacheCreationUsage(
				event.Message.Usage.CacheCreationInputTokens, event.Message.Usage.CacheCreation,
			)
		}

	case "content_block_start":
		if event.ContentBlock == nil {
			return nil
		}
		acc.currentBlockType = event.ContentBlock.Type
		acc.currentBlockID = event.ContentBlock.ID
		acc.currentBlockName = event.ContentBlock.Name
		acc.currentText = ""
		acc.currentThinking = ""
		acc.currentToolArgs = ""

		switch event.ContentBlock.Type {
		case "thinking":
			if !acc.headerEmitted {
				if err := emitAnthropicHeaderEvents(w, acc); err != nil {
					return err
				}
			}
			// Announce the reasoning item immediately so the client knows its output_index.
			acc.currentReasoningID = generateItemID("rs_")
			outputIdx := len(acc.outputItems)
			if err := writeAnthropicSSE(w, "response.output_item.added", map[string]interface{}{
				"type":         "response.output_item.added",
				"output_index": outputIdx,
				"item": map[string]interface{}{
					"type":    "reasoning",
					"id":      acc.currentReasoningID,
					"status":  "in_progress",
					"summary": []interface{}{},
				},
			}, acc); err != nil {
				return err
			}

		case "tool_use":
			if !acc.headerEmitted {
				if err := emitAnthropicHeaderEvents(w, acc); err != nil {
					return err
				}
			}
			// Calculate output_index: finalized outputItems + 1 for message (not in slice).
			outputIdx := len(acc.outputItems)
			if acc.messageStarted {
				outputIdx++
			}
			acc.currentToolItemID = generateItemID("fc_")
			acc.currentToolOutputIndex = outputIdx
			if err := writeAnthropicSSE(w, "response.output_item.added", map[string]interface{}{
				"type":         "response.output_item.added",
				"output_index": outputIdx,
				"item": map[string]interface{}{
					"type":      "function_call",
					"id":        acc.currentToolItemID,
					"status":    "in_progress",
					"call_id":   event.ContentBlock.ID,
					"name":      event.ContentBlock.Name,
					"arguments": "",
				},
			}, acc); err != nil {
				return err
			}
		}

	case "content_block_delta":
		if event.Delta == nil {
			return nil
		}
		switch event.Delta.Type {
		case "text_delta":
			if event.Delta.Text != "" {
				if err := handleAnthropicTextDelta(w, acc, event.Delta.Text); err != nil {
					return err
				}
			}
		case "thinking_delta":
			acc.currentThinking += event.Delta.Thinking
		case "input_json_delta":
			acc.currentToolArgs += event.Delta.PartialJSON
			if event.Delta.PartialJSON != "" && acc.currentToolItemID != "" {
				if err := writeAnthropicSSE(w, "response.function_call_arguments.delta", map[string]interface{}{
					"type":         "response.function_call_arguments.delta",
					"item_id":      acc.currentToolItemID,
					"output_index": acc.currentToolOutputIndex,
					"delta":        event.Delta.PartialJSON,
				}, acc); err != nil {
					return err
				}
			}
		}

	case "content_block_stop":
		if err := finalizeCurrentBlock(w, acc); err != nil {
			return err
		}

	case "message_delta":
		if event.Delta != nil && event.Delta.StopReason != "" {
			acc.stopReason = event.Delta.StopReason
		}
		if event.Usage != nil {
			acc.outputTokens = event.Usage.OutputTokens
			if event.Usage.CacheReadInputTokens > 0 {
				acc.cachedTokens = event.Usage.CacheReadInputTokens
			}
			if event.Usage.CacheCreationInputTokens > 0 {
				acc.cacheCreationTokens = event.Usage.CacheCreationInputTokens
				if event.Usage.CacheCreation != nil {
					acc.cacheCreationTokens, acc.cacheCreation5mTokens, acc.cacheCreation1hTokens = anthropic.NormalizeCacheCreationUsage(
						event.Usage.CacheCreationInputTokens, event.Usage.CacheCreation,
					)
				}
			} else if event.Usage.CacheCreation != nil {
				acc.cacheCreationTokens, acc.cacheCreation5mTokens, acc.cacheCreation1hTokens = anthropic.NormalizeCacheCreationUsage(
					acc.cacheCreationTokens, event.Usage.CacheCreation,
				)
			}
		}

	case "message_stop":
		// Stream is ending; completion events emitted after the scan loop.
	}
	return nil
}

func handleAnthropicTextDelta(w io.Writer, acc *anthropicStreamAccumulator, delta string) error {
	if !acc.headerEmitted {
		if err := emitAnthropicHeaderEvents(w, acc); err != nil {
			return err
		}
	}
	if !acc.messageStarted {
		// Save the output index BEFORE emitAnthropicMessageStart sets messageStarted=true,
		acc.messageOutputIndex = len(acc.outputItems)
		if err := emitAnthropicMessageStart(w, acc); err != nil {
			return err
		}
	}
	acc.currentText += delta
	return writeAnthropicSSE(w, "response.output_text.delta",
		responses.BuildOutputTextDeltaEvent(acc.messageItemID, acc.messageOutputIndex, 0, delta), acc)
}

func finalizeCurrentBlock(w io.Writer, acc *anthropicStreamAccumulator) error {
	switch acc.currentBlockType {
	case "text":
		if acc.currentText != "" {
			acc.msgContent = append(acc.msgContent, responses.OutputContent{
				Type:        "output_text",
				Text:        acc.currentText,
				Annotations: []responses.Annotation{},
			})
		}
		acc.currentText = ""

	case "thinking":
		itemID := acc.currentReasoningID
		if itemID == "" {
			itemID = generateItemID("rs_")
		}
		outputIdx := len(acc.outputItems)
		if acc.currentThinking != "" {
			item := responses.OutputItem{
				Type:   "reasoning",
				ID:     itemID,
				Status: "completed",
				Summary: []responses.OutputContent{
					{Type: "summary_text", Text: acc.currentThinking},
				},
			}
			acc.outputItems = append(acc.outputItems, item)
		}
		// Always emit output_item.done to close the added event emitted at block_start,
		// even when the thinking block was empty (avoids a dangling added with no done).
		if err := writeAnthropicSSE(w, "response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": outputIdx,
			"item": map[string]interface{}{
				"type":    "reasoning",
				"id":      itemID,
				"status":  "completed",
				"summary": []interface{}{},
			},
		}, acc); err != nil {
			return err
		}
		acc.currentReasoningID = ""

	case "tool_use":
		argsJSON := acc.currentToolArgs
		if argsJSON == "" {
			argsJSON = "{}"
		}
		itemID := acc.currentToolItemID
		if itemID == "" {
			itemID = generateItemID("fc_")
		}
		if err := writeAnthropicSSE(w, "response.function_call_arguments.done", map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      itemID,
			"output_index": acc.currentToolOutputIndex,
			"arguments":    argsJSON,
		}, acc); err != nil {
			return err
		}
		item := responses.OutputItem{
			Type:      "function_call",
			ID:        itemID,
			Status:    "completed",
			CallID:    acc.currentBlockID,
			Name:      acc.currentBlockName,
			Arguments: argsJSON,
		}
		acc.outputItems = append(acc.outputItems, item)
		acc.currentToolItemID = ""
		acc.currentToolOutputIndex = 0
	}

	acc.currentBlockType = ""
	return nil
}

func emitAnthropicHeaderEvents(w io.Writer, acc *anthropicStreamAccumulator) error {
	acc.headerEmitted = true
	respObj := buildAnthropicInProgressResponse(acc)
	if err := writeAnthropicSSE(w, "response.created",
		responses.BuildResponseEvent("response.created", respObj), acc); err != nil {
		return err
	}
	return writeAnthropicSSE(w, "response.in_progress",
		responses.BuildResponseEvent("response.in_progress", respObj), acc)
}

func emitAnthropicMessageStart(w io.Writer, acc *anthropicStreamAccumulator) error {
	acc.messageStarted = true
	acc.messageItemID = generateItemID("msg_")
	outputIdx := len(acc.outputItems)

	if err := writeAnthropicSSE(w, "response.output_item.added",
		responses.BuildMessageItemAddedEvent(outputIdx, acc.messageItemID), acc); err != nil {
		return err
	}

	return writeAnthropicSSE(w, "response.content_part.added",
		responses.BuildContentPartAddedEvent(acc.messageItemID, outputIdx, 0), acc)
}

func emitAnthropicCompletionEvents(w io.Writer, acc *anthropicStreamAccumulator) error {
	if !acc.headerEmitted {
		if err := emitAnthropicHeaderEvents(w, acc); err != nil {
			return err
		}
	}

	// Split point: outputItems[0..msgIdx-1] were announced before the message;
	// outputItems[msgIdx..] were announced after. When no message, close all items.
	msgIdx := acc.messageOutputIndex
	if !acc.messageStarted {
		msgIdx = len(acc.outputItems)
	}

	outputIdx := 0

	// Close items announced before the message (e.g. reasoning blocks).
	for i := 0; i < msgIdx && i < len(acc.outputItems); i++ {
		if err := writeAnthropicSSE(w, "response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": outputIdx,
			"item":         acc.outputItems[i],
		}, acc); err != nil {
			return err
		}
		outputIdx++
	}

	// Close the message item at its announced output_index.
	if acc.messageStarted {
		fullText := ""
		for _, c := range acc.msgContent {
			fullText += c.Text
		}
		if fullText != "" {
			if err := writeAnthropicSSE(w, "response.output_text.done",
				responses.BuildOutputTextDoneEvent(acc.messageItemID, outputIdx, 0, fullText), acc); err != nil {
				return err
			}
			if err := writeAnthropicSSE(w, "response.content_part.done",
				responses.BuildContentPartDoneEvent(acc.messageItemID, outputIdx, 0, fullText), acc); err != nil {
				return err
			}
			if err := writeAnthropicSSE(w, "response.output_item.done", map[string]interface{}{
				"type":         "response.output_item.done",
				"output_index": outputIdx,
				"item": map[string]interface{}{
					"type":    "message",
					"id":      acc.messageItemID,
					"status":  "completed",
					"role":    "assistant",
					"content": acc.msgContent,
				},
			}, acc); err != nil {
				return err
			}
		}
		outputIdx++ // advance past the message slot regardless of content
	}

	// Close items announced after the message (e.g. tool calls).
	for i := msgIdx; i < len(acc.outputItems); i++ {
		if err := writeAnthropicSSE(w, "response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": outputIdx,
			"item":         acc.outputItems[i],
		}, acc); err != nil {
			return err
		}
		outputIdx++
	}

	return writeAnthropicSSE(w, "response.completed",
		responses.BuildResponseEvent("response.completed", buildAnthropicCompletedResponse(acc)), acc)
}

func buildAnthropicInProgressResponse(acc *anthropicStreamAccumulator) map[string]interface{} {
	resp := responses.BuildInProgressResponse(acc.responseID, acc.model, acc.createdAt)
	if acc.meta != nil && acc.meta.PreviousResponseID != "" {
		resp["previous_response_id"] = acc.meta.PreviousResponseID
	}
	return resp
}

func buildAnthropicCompletedResponse(acc *anthropicStreamAccumulator) *responses.Response {
	status, incompleteDetails := anthropicStopReasonToStatus(acc.stopReason)

	var output []responses.OutputItem
	output = append(output, acc.outputItems...)

	if acc.messageStarted {
		msgContent := acc.msgContent
		if len(msgContent) == 0 {
			msgContent = []responses.OutputContent{{Type: "output_text", Text: "", Annotations: []responses.Annotation{}}}
		}
		output = append(output, responses.OutputItem{
			Type:    "message",
			ID:      acc.messageItemID,
			Status:  "completed",
			Role:    "assistant",
			Content: msgContent,
		})
	}

	if len(output) == 0 {
		output = []responses.OutputItem{{
			Type:    "message",
			ID:      generateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: []responses.OutputContent{{Type: "output_text", Text: "", Annotations: []responses.Annotation{}}},
		}}
	}

	totalInputTokens := acc.inputTokens + acc.cachedTokens + acc.cacheCreationTokens
	usage := &responses.Usage{
		InputTokens:  totalInputTokens,
		OutputTokens: acc.outputTokens,
		TotalTokens:  totalInputTokens + acc.outputTokens,
		InputTokensDetails: responses.InputDetails{
			CachedTokens:        acc.cachedTokens,
			CacheCreationTokens: acc.cacheCreationTokens,
		},
	}
	if acc.cacheCreation5mTokens > 0 || acc.cacheCreation1hTokens > 0 {
		usage.InputTokensDetails.CacheCreationTokenDetails = &responses.CacheCreationTokenDetails{
			Ephemeral5mInputTokens: acc.cacheCreation5mTokens,
			Ephemeral1hInputTokens: acc.cacheCreation1hTokens,
		}
	}

	completedAt := acc.createdAt
	metadata := map[string]string{}
	var prevRespID interface{}
	if acc.meta != nil {
		for k, v := range acc.meta.Metadata {
			metadata[k] = v
		}
		if acc.meta.PreviousResponseID != "" {
			prevRespID = acc.meta.PreviousResponseID
		}
	}

	return responses.BuildCompletedResponse(responses.CompletedResponseParams{
		ID:                 acc.responseID,
		Model:              acc.model,
		CreatedAt:          acc.createdAt,
		CompletedAt:        &completedAt,
		Status:             status,
		IncompleteDetails:  incompleteDetails,
		Output:             output,
		Usage:              usage,
		Metadata:           metadata,
		PreviousResponseID: prevRespID,
	})
}

func writeAnthropicSSE(w io.Writer, eventType string, data interface{}, acc *anthropicStreamAccumulator) error {
	return responses.WriteSSEEvent(w, eventType, data, &acc.sequenceNumber)
}
