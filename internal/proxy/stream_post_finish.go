package proxy

import (
	"encoding/json"
	"strconv"
)

// chatPostFinishDetector tracks terminal choices independently. The SSE
// sanitizer passes its already decoded JSON object to observeJSON.
type chatPostFinishDetector struct {
	choices map[int]chatPostFinishChoice
}

type chatPostFinishChoice struct {
	reason   string
	reported bool
}

type chatPostFinishViolation struct {
	index        int
	finishReason string
	field        string
}

func (d *chatPostFinishDetector) observeJSON(value any) []chatPostFinishViolation {
	chunk, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	choices, ok := chunk["choices"].([]any)
	if !ok {
		return nil
	}
	var violations []chatPostFinishViolation
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		indexNumber, ok := choice["index"].(json.Number)
		if !ok {
			continue
		}
		index, err := strconv.Atoi(string(indexNumber))
		if err != nil {
			continue
		}
		state := d.choices[index]
		if state.reason != "" && !state.reported {
			delta, _ := choice["delta"].(map[string]any)
			field := ""
			if content, _ := delta["content"].(string); content != "" {
				field = "content"
			} else if reasoning, _ := delta["reasoning_content"].(string); reasoning != "" {
				field = "reasoning_content"
			}
			if field != "" {
				violations = append(violations, chatPostFinishViolation{index, state.reason, field})
				state.reported = true
			}
		}
		if reason, _ := choice["finish_reason"].(string); reason != "" && state.reason == "" {
			state.reason = reason
		}
		if state.reason != "" {
			if d.choices == nil {
				d.choices = make(map[int]chatPostFinishChoice)
			}
			d.choices[index] = state
		}
	}
	return violations
}
