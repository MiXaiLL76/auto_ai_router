package responses

import (
	"bufio"
	"cmp"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"

	// goccy/go-json instead of encoding/json: both json.* calls in this file run
	// once per streamed chunk/event — chatStreamChunk unmarshal (benchmarked
	// ~5.2x faster) and the outgoing SSE event map[string]interface{} marshal
	// (~1.5x faster) in writeSSEWithSeq.
	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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
	fullText    string
	fullRefusal string // accumulated refusal text
	toolCalls   []accumulatedToolCall
	// toolSlots maps an upstream tool_calls[].index to the toolCalls slot of
	// the call currently streaming under it. Slots are allocated on demand,
	// so a sparse or huge upstream index never allocates the gap before it.
	toolSlots map[int]int

	// Reasoning output items, one per contiguous run of reasoning deltas.
	// Usually a single item at output_index 0: DeepSeek-style providers
	// stream reasoning before content/tool_calls.
	reasoningItems []streamReasoningItem

	// nextOutputIndex is the output_index the next output item opens at.
	// An item keeps the index it opened with, so one opening late (e.g.
	// reasoning after text) never shifts indices already sent to the client.
	nextOutputIndex    int
	messageOutputIndex int

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

// allocOutputIndex reserves the output_index for an output item being opened.
func (acc *streamAccumulator) allocOutputIndex() int {
	idx := acc.nextOutputIndex
	acc.nextOutputIndex++
	return idx
}

// openReasoningItem returns the reasoning item currently receiving deltas,
// or nil once visible output closed it (or before any reasoning arrived).
func (acc *streamAccumulator) openReasoningItem() *streamReasoningItem {
	if n := len(acc.reasoningItems); n > 0 && !acc.reasoningItems[n-1].closed {
		return &acc.reasoningItems[n-1]
	}
	return nil
}

// maxStreamToolCalls caps the function_call items one streamed response may
// open, bounding memory against a malformed or hostile upstream.
const maxStreamToolCalls = 1024

// toolCallSlot resolves an upstream tool call delta to its toolCalls slot,
// opening a new call when the delta starts one. A new call starts when the
// delta carries an ID other than the one streaming under its index — some
// OpenAI-compatible backends send index 0 for every parallel call — or when
// an unseen index carries data without an ID, which gets a generated call_id
// so the client never receives a function_call it cannot answer.
// Returns -1 for a delta to ignore.
func (acc *streamAccumulator) toolCallSlot(index int, id string, hasData bool) (slot int, isNew bool) {
	if index < 0 {
		// Malformed upstream chunk.
		return -1, false
	}
	slot, seen := acc.toolSlots[index]
	if seen && (id == "" || acc.toolCalls[slot].id == id) {
		return slot, false
	}
	if id == "" && !hasData {
		return -1, false
	}
	if len(acc.toolCalls) >= maxStreamToolCalls {
		slog.Warn("[responses/streaming] tool call limit reached, dropping further calls",
			"limit", maxStreamToolCalls)
		return -1, false
	}
	if id == "" {
		id = GenerateItemID("call_")
	}
	if acc.toolSlots == nil {
		acc.toolSlots = make(map[int]int)
	}
	slot = len(acc.toolCalls)
	acc.toolSlots[index] = slot
	acc.toolCalls = append(acc.toolCalls, accumulatedToolCall{
		id:          id,
		itemID:      GenerateItemID("fc_"),
		outputIndex: acc.allocOutputIndex(),
	})
	return slot, true
}

type accumulatedToolCall struct {
	id          string
	name        string
	arguments   string
	itemID      string // Responses API item ID
	outputIndex int
}

// streamReasoningItem is a reasoning output item built from reasoning deltas.
type streamReasoningItem struct {
	itemID      string
	outputIndex int
	text        string
	closed      bool
}

// chatCompletionsUsage represents usage from a Chat Completions streaming chunk.
type chatCompletionsUsage struct {
	PromptTokens          int
	CompletionTokens      int
	TotalTokens           int
	CachedTokens          int
	CachedAudioTokens     int
	CacheCreationTokens   int
	CacheCreation5mTokens int
	CacheCreation1hTokens int
	// CacheType is Alibaba/Qwen's explicit cache mode marker
	// (converter.CacheTypeExplicit) from prompt_tokens_details.cache_type.
	CacheType         string
	ReasoningTokens   int
	AudioInputTokens  int
	AudioOutputTokens int
	ImageOutputTokens int
	WebSearchRequests int
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
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
			Refusal string `json:"refusal,omitempty"`
			// See converterutil.PickReasoningField for the two spellings and why interface{}.
			ReasoningContent interface{} `json:"reasoning_content,omitempty"`
			Reasoning        interface{} `json:"reasoning,omitempty"`
			ToolCalls        []struct {
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
			CachedTokens              int `json:"cached_tokens,omitempty"`
			CachedAudioTokens         int `json:"cached_audio_tokens,omitempty"`
			CacheCreationTokens       int `json:"cache_creation_tokens,omitempty"`
			CacheWriteTokens          int `json:"cache_write_tokens,omitempty"`
			CacheCreationTokenDetails *struct {
				Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
				Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
			} `json:"cache_creation_token_details,omitempty"`
			// Alibaba returns the explicit cache creation TTL detail under
			// cache_creation.ephemeral_5m_input_tokens (no _token_details suffix).
			CacheCreation *struct {
				Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
				Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
			} `json:"cache_creation,omitempty"`
			AudioTokens int    `json:"audio_tokens,omitempty"`
			CacheType   string `json:"cache_type,omitempty"`
		} `json:"prompt_tokens_details,omitempty"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
			AudioTokens     int `json:"audio_tokens,omitempty"`
			ImageTokens     int `json:"image_tokens,omitempty"`
		} `json:"completion_tokens_details,omitempty"`
		ServerToolUse *struct {
			WebSearchRequests int `json:"web_search_requests,omitempty"`
		} `json:"server_tool_use,omitempty"`
	} `json:"usage,omitempty"`
}

// TransformChatStreamToResponses reads Chat Completions SSE from reader,
// transforms to Responses API SSE events, and writes to writer.
// The optional onComplete callback is invoked with the fully-built Response
// once the stream is complete (on [DONE] or stream end).
// Pass a *ResponsesMetadata as the second variadic element to have store,
// previous_response_id and metadata echoed back in all SSE response objects.
func TransformChatStreamToResponses(reader io.Reader, writer io.Writer, model string, onComplete ...func(*Response)) error {
	return transformChatStreamToResponsesInner(reader, writer, model, nil, true, 0, 0, onComplete...)
}

// TransformChatStreamToResponsesWithMeta is like TransformChatStreamToResponses but
// additionally echoes request-side fields (store, previous_response_id, metadata) into
// every emitted response object so the wire payload matches the stored record.
func TransformChatStreamToResponsesWithMeta(reader io.Reader, writer io.Writer, model string, meta *ResponsesMetadata, onComplete ...func(*Response)) error {
	return transformChatStreamToResponsesInner(reader, writer, model, meta, true, 0, 0, onComplete...)
}

// TransformChatStreamToResponsesWithMetaAndUsage is like
// TransformChatStreamToResponsesWithMeta but also defines whether the source
// audio_tokens value includes cached audio, and carries the cache-write TTL
// split sourced from upstream response headers (used by Kimi/Moonshot, whose
// chunks never carry cache_creation_token_details/cache_creation — the split
// is only available via response headers read by the caller). Applied only
// as a fallback when a chunk's own usage carries no 5m/1h breakdown.
func TransformChatStreamToResponsesWithMetaAndUsage(
	reader io.Reader,
	writer io.Writer,
	model string,
	meta *ResponsesMetadata,
	audioInputIncludesCachedAudio bool,
	cacheWriteTTLFallback5mTokens int,
	cacheWriteTTLFallback1hTokens int,
	onComplete ...func(*Response),
) error {
	return transformChatStreamToResponsesInner(
		reader, writer, model, meta, audioInputIncludesCachedAudio,
		cacheWriteTTLFallback5mTokens, cacheWriteTTLFallback1hTokens, onComplete...,
	)
}

func transformChatStreamToResponsesInner(
	reader io.Reader,
	writer io.Writer,
	model string,
	meta *ResponsesMetadata,
	audioInputIncludesCachedAudio bool,
	cacheWriteTTLFallback5mTokens int,
	cacheWriteTTLFallback1hTokens int,
	onComplete ...func(*Response),
) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), converterutil.MaxSSELineBytes)

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
				cachedTokens, cachedAudioTokens := converterutil.NormalizeCachedAudioBreakdown(
					chunk.Usage.PromptTokensDetails.CachedTokens,
					chunk.Usage.PromptTokensDetails.CachedAudioTokens,
				)
				acc.usage.CachedTokens = cachedTokens
				acc.usage.CachedAudioTokens = cachedAudioTokens
				acc.usage.CacheType = chunk.Usage.PromptTokensDetails.CacheType
				acc.usage.CacheCreationTokens = chunk.Usage.PromptTokensDetails.CacheCreationTokens
				if acc.usage.CacheCreationTokens == 0 {
					acc.usage.CacheCreationTokens = chunk.Usage.PromptTokensDetails.CacheWriteTokens
				}
				acc.usage.AudioInputTokens = normalizedAudioTokens(
					chunk.Usage.PromptTokensDetails.AudioTokens,
					cachedTokens,
					cachedAudioTokens,
					audioInputIncludesCachedAudio,
				)
				details := chunk.Usage.PromptTokensDetails.CacheCreationTokenDetails
				if details == nil {
					// Alibaba spells the TTL detail cache_creation.ephemeral_5m_input_tokens
					// (nested in prompt_tokens_details, no _token_details suffix).
					details = chunk.Usage.PromptTokensDetails.CacheCreation
				}
				if details != nil {
					acc.usage.CacheCreation5mTokens = details.Ephemeral5mInputTokens
					acc.usage.CacheCreation1hTokens = details.Ephemeral1hInputTokens
					if acc.usage.CacheCreationTokens == 0 {
						acc.usage.CacheCreationTokens = details.Ephemeral5mInputTokens + details.Ephemeral1hInputTokens
					}
				}
				// Kimi/Moonshot never reports a TTL split in the body — fall back
				// to the header-sourced split the caller supplied.
				if acc.usage.CacheCreation5mTokens == 0 && acc.usage.CacheCreation1hTokens == 0 {
					acc.usage.CacheCreation5mTokens = cacheWriteTTLFallback5mTokens
					acc.usage.CacheCreation1hTokens = cacheWriteTTLFallback1hTokens
				}
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				acc.usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
				acc.usage.AudioOutputTokens = chunk.Usage.CompletionTokensDetails.AudioTokens
				acc.usage.ImageOutputTokens = chunk.Usage.CompletionTokensDetails.ImageTokens
			}
			if chunk.Usage.ServerToolUse != nil && chunk.Usage.ServerToolUse.WebSearchRequests > 0 {
				acc.usage.WebSearchRequests = chunk.Usage.ServerToolUse.WebSearchRequests
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Handle reasoning delta BEFORE text/tool_calls: DeepSeek-style
		// providers stream reasoning ahead of the visible content, and the
		// reasoning output item must be announced (and later closed) before
		// the message/tool_call item that follows it.
		if reasoningDelta := converterutil.ReasoningText(choice.Delta.ReasoningContent, choice.Delta.Reasoning); reasoningDelta != "" {
			if !acc.headerEmitted {
				if err := emitHeaderEvents(writer, acc); err != nil {
					return err
				}
			}
			if err := emitReasoningDelta(writer, acc, reasoningDelta); err != nil {
				return err
			}
		}

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
			if err := closeReasoningItem(writer, acc); err != nil {
				return err
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
				"output_index":  acc.messageOutputIndex,
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
			if err := closeReasoningItem(writer, acc); err != nil {
				return err
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
				"output_index":  acc.messageOutputIndex,
				"content_index": 0,
				"delta":         choice.Delta.Refusal,
			}
			if err := writeSSEWithSeq(writer, "response.refusal.delta", refusalEvent, acc); err != nil {
				return err
			}
		}

		// Handle tool call deltas
		for _, tc := range choice.Delta.ToolCalls {
			var name, args string
			if tc.Function != nil {
				name, args = tc.Function.Name, tc.Function.Arguments
			}
			slot, isNew := acc.toolCallSlot(tc.Index, tc.ID, name != "" || args != "")
			if slot < 0 {
				continue
			}
			if !acc.headerEmitted {
				if err := emitHeaderEvents(writer, acc); err != nil {
					return err
				}
			}
			if err := closeReasoningItem(writer, acc); err != nil {
				return err
			}

			call := &acc.toolCalls[slot]
			if isNew {
				call.name = name
				acc.state = stateStreamingToolCall
				if err := writeSSEWithSeq(writer, "response.output_item.added",
					BuildFunctionCallItemAddedEvent(call.outputIndex, call.itemID, call.id, call.name), acc); err != nil {
					return err
				}
			} else if name != "" && call.name == "" {
				// The call's name arrived after the chunk that opened it.
				call.name = name
			}

			if args != "" {
				call.arguments += args
				argDeltaEvent := map[string]interface{}{
					"type":         "response.function_call_arguments.delta",
					"item_id":      call.itemID,
					"output_index": call.outputIndex,
					"delta":        args,
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

// completedOutputItems returns the stream's output items in output_index
// order, the order the client saw them open in.
func completedOutputItems(acc *streamAccumulator) []OutputItem {
	type indexedItem struct {
		outputIndex int
		item        OutputItem
	}
	// Appended in the usual reasoning → message → tool calls order, which the
	// stable sort keeps for items without an assigned index.
	var items []indexedItem

	for _, r := range acc.reasoningItems {
		if r.text == "" {
			continue
		}
		items = append(items, indexedItem{r.outputIndex, OutputItem{
			Type:   "reasoning",
			ID:     r.itemID,
			Status: "completed",
			Summary: []OutputContent{
				{Type: "summary_text", Text: r.text},
			},
		}})
	}

	if acc.messageStarted && acc.fullText != "" {
		items = append(items, indexedItem{acc.messageOutputIndex, OutputItem{
			Type:   "message",
			ID:     acc.messageItemID,
			Status: "completed",
			Role:   "assistant",
			Content: []OutputContent{{
				Type:        "output_text",
				Text:        acc.fullText,
				Annotations: []Annotation{},
			}},
		}})
	}

	for _, tc := range acc.toolCalls {
		items = append(items, indexedItem{tc.outputIndex, OutputItem{
			Type:      "function_call",
			ID:        tc.itemID,
			Status:    "completed",
			CallID:    tc.id,
			Name:      tc.name,
			Arguments: tc.arguments,
		}})
	}

	slices.SortStableFunc(items, func(a, b indexedItem) int {
		return cmp.Compare(a.outputIndex, b.outputIndex)
	})
	output := make([]OutputItem, 0, len(items))
	for _, it := range items {
		output = append(output, it.item)
	}
	return output
}

// buildTypedCompletedResponse builds a typed *Response from the stream accumulator.
func buildTypedCompletedResponse(acc *streamAccumulator) *Response {
	output := completedOutputItems(acc)

	var usage *Usage
	if acc.usage != nil {
		usage = &Usage{
			InputTokens:  acc.usage.PromptTokens,
			OutputTokens: acc.usage.CompletionTokens,
			TotalTokens:  acc.usage.TotalTokens,
			InputTokensDetails: InputDetails{
				CachedTokens:        acc.usage.CachedTokens,
				CachedAudioTokens:   acc.usage.CachedAudioTokens,
				CacheCreationTokens: acc.usage.CacheCreationTokens,
				AudioTokens:         acc.usage.AudioInputTokens,
				CacheType:           acc.usage.CacheType,
			},
			OutputTokensDetails: OutputDetails{ReasoningTokens: acc.usage.ReasoningTokens, AudioTokens: acc.usage.AudioOutputTokens, ImageTokens: acc.usage.ImageOutputTokens},
		}
		if acc.usage.WebSearchRequests > 0 {
			usage.ServerToolUse = &ServerToolUseDetails{
				WebSearchRequests: acc.usage.WebSearchRequests,
			}
		}
		if acc.usage.CacheCreation5mTokens > 0 || acc.usage.CacheCreation1hTokens > 0 {
			usage.InputTokensDetails.CacheCreationTokenDetails = &CacheCreationTokenDetails{
				Ephemeral5mInputTokens: acc.usage.CacheCreation5mTokens,
				Ephemeral1hInputTokens: acc.usage.CacheCreation1hTokens,
			}
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

// emitReasoningDelta streams a reasoning delta as a reasoning output item with
// a single summary_text part. Chat's reasoning_content is the raw chain of
// thought, which the spec would put in content[{type:"reasoning_text"}]; it is
// mapped to summary_text on purpose, because that is what Responses clients
// (Codex) render, and it matches ChatToResponse and the Anthropic/Vertex
// converters. RequestToChat accepts either form on the way back.
// The item is opened (output_item.added +
// reasoning_summary_part.added) on its first delta — the event sequence
// OpenAI's own Responses API streams reasoning summaries with.
func emitReasoningDelta(w io.Writer, acc *streamAccumulator, delta string) error {
	item := acc.openReasoningItem()
	if item == nil {
		acc.reasoningItems = append(acc.reasoningItems, streamReasoningItem{
			itemID:      GenerateItemID("rs_"),
			outputIndex: acc.allocOutputIndex(),
		})
		item = &acc.reasoningItems[len(acc.reasoningItems)-1]

		for _, ev := range BuildReasoningItemOpenEvents(item.outputIndex, item.itemID) {
			if err := writeSSEWithSeq(w, ev["type"].(string), ev, acc); err != nil {
				return err
			}
		}
	}

	item.text += delta
	deltaEvent := BuildReasoningSummaryTextDeltaEvent(item.itemID, item.outputIndex, 0, delta)
	return writeSSEWithSeq(w, "response.reasoning_summary_text.delta", deltaEvent, acc)
}

// closeReasoningItem closes the reasoning item still receiving deltas, if
// any. Called as soon as visible output (text, refusal, tool call) arrives,
// and as a safety net at completion time in case the stream ended with
// reasoning only (e.g. max_output_tokens was exhausted entirely by
// reasoning, before any visible content).
func closeReasoningItem(w io.Writer, acc *streamAccumulator) error {
	item := acc.openReasoningItem()
	if item == nil {
		return nil
	}
	item.closed = true

	for _, ev := range BuildReasoningItemCloseEvents(item.outputIndex, item.itemID, item.text) {
		if err := writeSSEWithSeq(w, ev["type"].(string), ev, acc); err != nil {
			return err
		}
	}
	return nil
}

// emitMessageStartEvents emits output_item.added and content_part.added for a message.
func emitMessageStartEvents(w io.Writer, acc *streamAccumulator) error {
	if err := closeReasoningItem(w, acc); err != nil {
		return err
	}

	acc.messageStarted = true
	acc.messageItemID = GenerateItemID("msg_")
	acc.messageOutputIndex = acc.allocOutputIndex()

	msgItemID := acc.messageItemID
	outputIndex := acc.messageOutputIndex

	itemAddedEvent := BuildMessageItemAddedEvent(outputIndex, msgItemID)
	if err := writeSSEWithSeq(w, "response.output_item.added", itemAddedEvent, acc); err != nil {
		return err
	}

	contentPartEvent := BuildContentPartAddedEvent(acc.messageItemID, outputIndex, 0)
	return writeSSEWithSeq(w, "response.content_part.added", contentPartEvent, acc)
}

// emitCompletionEvents emits all closing events and the final response.completed.
func emitCompletionEvents(w io.Writer, acc *streamAccumulator) error {
	// Safety net: close a reasoning item that was opened but never followed
	// by a message or tool_call (e.g. max_output_tokens was exhausted entirely
	// by reasoning, before any visible content).
	if err := closeReasoningItem(w, acc); err != nil {
		return err
	}

	msgOutputIndex := acc.messageOutputIndex

	// only emit text closing events if there's actual text content.
	// This matches the condition in buildCompletedResponse (messageStarted && fullText != "").
	if acc.messageStarted && acc.fullText != "" {
		// output_text.done
		textDoneEvent := BuildOutputTextDoneEvent(acc.messageItemID, msgOutputIndex, 0, acc.fullText)
		if err := writeSSEWithSeq(w, "response.output_text.done", textDoneEvent, acc); err != nil {
			return err
		}

		// content_part.done
		contentPartDoneEvent := BuildContentPartDoneEvent(acc.messageItemID, msgOutputIndex, 0, acc.fullText)
		if err := writeSSEWithSeq(w, "response.content_part.done", contentPartDoneEvent, acc); err != nil {
			return err
		}

		// output_item.done for message
		msgDoneEvent := map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": msgOutputIndex,
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
			"output_index":  msgOutputIndex,
			"content_index": 0,
			"refusal":       acc.fullRefusal,
		}
		if err := writeSSEWithSeq(w, "response.refusal.done", refusalDoneEvent, acc); err != nil {
			return err
		}
	}

	// Close tool calls
	for _, tc := range acc.toolCalls {
		outputIndex := tc.outputIndex

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
	output := completedOutputItems(acc)

	var usageObj *Usage
	if acc.usage != nil {
		usageObj = &Usage{
			InputTokens:  acc.usage.PromptTokens,
			OutputTokens: acc.usage.CompletionTokens,
			TotalTokens:  acc.usage.TotalTokens,
			InputTokensDetails: InputDetails{
				CachedTokens:        acc.usage.CachedTokens,
				CachedAudioTokens:   acc.usage.CachedAudioTokens,
				CacheCreationTokens: acc.usage.CacheCreationTokens,
				AudioTokens:         acc.usage.AudioInputTokens,
				CacheType:           acc.usage.CacheType,
			},
			OutputTokensDetails: OutputDetails{
				ReasoningTokens: acc.usage.ReasoningTokens,
				AudioTokens:     acc.usage.AudioOutputTokens,
				ImageTokens:     acc.usage.ImageOutputTokens,
			},
		}
		if acc.usage.WebSearchRequests > 0 {
			usageObj.ServerToolUse = &ServerToolUseDetails{
				WebSearchRequests: acc.usage.WebSearchRequests,
			}
		}
		if acc.usage.CacheCreation5mTokens > 0 || acc.usage.CacheCreation1hTokens > 0 {
			usageObj.InputTokensDetails.CacheCreationTokenDetails = &CacheCreationTokenDetails{
				Ephemeral5mInputTokens: acc.usage.CacheCreation5mTokens,
				Ephemeral1hInputTokens: acc.usage.CacheCreation1hTokens,
			}
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
