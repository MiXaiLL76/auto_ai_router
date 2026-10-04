package responses

import "encoding/json"

// ResponseParams holds provider-neutral fields used to build a full Responses API object.
type ResponseParams struct {
	ID                 string
	Model              string
	CreatedAt          int64
	Status             string
	CompletedAt        *int64
	IncompleteDetails  *IncompleteDetails
	Output             []OutputItem
	Usage              *Usage
	Metadata           map[string]string
	PreviousResponseID interface{}
	Store              bool
	ToolChoice         interface{}
}

// NewResponse builds a fully-populated Responses API object with schema-required defaults.
func NewResponse(p ResponseParams) *Response {
	metadata := p.Metadata
	if metadata == nil {
		metadata = map[string]string{}
	}
	output := p.Output
	if output == nil {
		output = []OutputItem{}
	}
	temperature := 1.0
	topP := 1.0
	toolChoice := p.ToolChoice
	if toolChoice == nil {
		toolChoice = "auto"
	}

	return &Response{
		ID:                 p.ID,
		Object:             "response",
		CreatedAt:          p.CreatedAt,
		CompletedAt:        p.CompletedAt,
		Model:              p.Model,
		Status:             p.Status,
		IncompleteDetails:  p.IncompleteDetails,
		Output:             output,
		Usage:              p.Usage,
		Error:              nil,
		Metadata:           metadata,
		Tools:              []Tool{},
		ParallelToolCalls:  true,
		PreviousResponseID: p.PreviousResponseID,
		Instructions:       nil,
		MaxOutputTokens:    nil,
		Reasoning:          nil,
		SafetyIdentifier:   nil,
		PromptCacheKey:     nil,
		Store:              p.Store,
		Background:         false,
		PresencePenalty:    0,
		FrequencyPenalty:   0,
		TopLogprobs:        0,
		Temperature:        &temperature,
		TopP:               &topP,
		Truncation:         "disabled",
		ServiceTier:        "default",
		ToolChoice:         toolChoice,
		Text: map[string]interface{}{
			"format": map[string]interface{}{
				"type": "text",
			},
		},
		MaxToolCalls: nil,
	}
}

// ResponseToMap converts a typed Response to a generic map payload suitable for SSE events.
func ResponseToMap(resp *Response) map[string]interface{} {
	raw, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(raw, &out); err != nil {
		panic(err)
	}
	return out
}

func BuildMessageItemAddedEvent(outputIndex int, itemID string) map[string]interface{} {
	return map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": outputIndex,
		"item": map[string]interface{}{
			"type":    "message",
			"id":      itemID,
			"status":  "in_progress",
			"role":    "assistant",
			"content": []interface{}{},
		},
	}
}

func BuildResponseEvent(eventType string, response interface{}) map[string]interface{} {
	return map[string]interface{}{
		"type":     eventType,
		"response": response,
	}
}

func BuildTextPart(text string) map[string]interface{} {
	return map[string]interface{}{
		"type":        "output_text",
		"text":        text,
		"annotations": []interface{}{},
	}
}

func BuildContentPartAddedEvent(itemID string, outputIndex, contentIndex int) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.content_part.added",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"content_index": contentIndex,
		"part":          BuildTextPart(""),
	}
}

func BuildOutputTextDeltaEvent(itemID string, outputIndex, contentIndex int, delta string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.output_text.delta",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"content_index": contentIndex,
		"delta":         delta,
	}
}

func BuildOutputTextDoneEvent(itemID string, outputIndex, contentIndex int, text string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.output_text.done",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"content_index": contentIndex,
		"text":          text,
	}
}

func BuildContentPartDoneEvent(itemID string, outputIndex, contentIndex int, text string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.content_part.done",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"content_index": contentIndex,
		"part":          BuildTextPart(text),
	}
}

func BuildReasoningSummaryPart(text string) map[string]interface{} {
	return map[string]interface{}{
		"type": "summary_text",
		"text": text,
	}
}

func BuildReasoningSummaryPartAddedEvent(itemID string, outputIndex, summaryIndex int) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.reasoning_summary_part.added",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"summary_index": summaryIndex,
		"part":          BuildReasoningSummaryPart(""),
	}
}

func BuildReasoningSummaryTextDeltaEvent(itemID string, outputIndex, summaryIndex int, delta string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.reasoning_summary_text.delta",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"summary_index": summaryIndex,
		"delta":         delta,
	}
}

func BuildReasoningSummaryTextDoneEvent(itemID string, outputIndex, summaryIndex int, text string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.reasoning_summary_text.done",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"summary_index": summaryIndex,
		"text":          text,
	}
}

func BuildReasoningSummaryPartDoneEvent(itemID string, outputIndex, summaryIndex int, text string) map[string]interface{} {
	return map[string]interface{}{
		"type":          "response.reasoning_summary_part.done",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"summary_index": summaryIndex,
		"part":          BuildReasoningSummaryPart(text),
	}
}
