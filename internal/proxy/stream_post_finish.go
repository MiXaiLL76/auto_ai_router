package proxy

import (
	json "github.com/goccy/go-json"
)

// chatPostFinishDetector tracks terminal choices independently. Usage-only
// chunks have no choices and therefore cannot signal a violation.
type chatPostFinishDetector struct {
	finished map[int]string
	reported map[int]bool
}

type chatPostFinishViolation struct {
	index        int
	finishReason string
	field        string
}

// observePayload accepts JSON from an already parsed SSE data line. A delta
// in the same choice object as finish_reason is allowed.
func (d *chatPostFinishDetector) observePayload(payload []byte) []chatPostFinishViolation {
	if len(payload) == 0 || payload[0] != '{' {
		return nil
	}
	var chunk struct {
		Choices []struct {
			Index        *int    `json:"index"`
			FinishReason *string `json:"finish_reason"`
			Delta        struct {
				Content          json.RawMessage `json:"content"`
				ReasoningContent json.RawMessage `json:"reasoning_content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(payload, &chunk) != nil {
		return nil
	}
	var violations []chatPostFinishViolation
	for _, choice := range chunk.Choices {
		if choice.Index == nil {
			continue
		}
		index := *choice.Index
		if reason, finished := d.finished[index]; finished && !d.reported[index] {
			field := ""
			switch {
			case nonEmptyJSONText(choice.Delta.Content):
				field = "content"
			case nonEmptyJSONText(choice.Delta.ReasoningContent):
				field = "reasoning_content"
			}
			if field != "" {
				violations = append(violations, chatPostFinishViolation{index, reason, field})
				if d.reported == nil {
					d.reported = make(map[int]bool)
				}
				d.reported[index] = true
			}
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			if d.finished == nil {
				d.finished = make(map[int]string)
			}
			if _, exists := d.finished[index]; !exists {
				d.finished[index] = *choice.FinishReason
			}
		}
	}
	return violations
}

func nonEmptyJSONText(raw json.RawMessage) bool {
	var value string
	return json.Unmarshal(raw, &value) == nil && value != ""
}
