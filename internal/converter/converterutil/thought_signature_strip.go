package converterutil

import (
	"bytes"
	"encoding/json"
	"strings"
)

// StripThoughtSignaturesFromToolIDs removes Gemini-only call ID suffixes before
// sending a conversation to a different provider. It also removes the matching
// Gemini metadata from Chat tool calls. RawMessage keeps numbers and unrelated
// JSON values intact while the changed objects are rebuilt.
func StripThoughtSignaturesFromToolIDs(body []byte) []byte {
	if !bytes.Contains(body, []byte(ThoughtSignatureSeparator)) &&
		!bytes.Contains(body, []byte("thought_signature")) {
		return body
	}
	cleaned, changed, err := stripThoughtSignaturesJSON(body, "")
	if err != nil || !changed {
		return body
	}
	return cleaned
}

func stripThoughtSignaturesJSON(raw []byte, parent string) ([]byte, bool, error) {
	if !bytes.Contains(raw, []byte(ThoughtSignatureSeparator)) &&
		!bytes.Contains(raw, []byte("thought_signature")) {
		return raw, false, nil
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return raw, false, nil
	}
	switch trimmed[0] {
	case '{':
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return nil, false, err
		}
		var itemType string
		_ = json.Unmarshal(fields["type"], &itemType)
		isToolCall := parent == "tool_calls" || itemType == "function_call" || itemType == "tool_use"
		changed := false
		for key, value := range fields {
			isCallID := key == "tool_call_id" || key == "call_id" || key == "tool_use_id" || (key == "id" && isToolCall)
			if isCallID {
				var id string
				if json.Unmarshal(value, &id) == nil {
					if base, _, found := strings.Cut(id, ThoughtSignatureSeparator); found {
						fields[key], _ = json.Marshal(base)
						changed = true
					}
				}
				continue
			}
			if key == "provider_specific_fields" && isToolCall {
				var metadata map[string]json.RawMessage
				if json.Unmarshal(value, &metadata) == nil {
					_, signature := metadata["thought_signature"]
					_, skip := metadata["skip_thought_signature_validator"]
					if signature || skip {
						delete(metadata, "thought_signature")
						delete(metadata, "skip_thought_signature_validator")
						if len(metadata) == 0 {
							delete(fields, key)
						} else {
							fields[key], _ = json.Marshal(metadata)
						}
						changed = true
					}
				}
				continue
			}
			cleaned, childChanged, err := stripThoughtSignaturesJSON(value, key)
			if err != nil {
				return nil, false, err
			}
			if childChanged {
				fields[key] = cleaned
				changed = true
			}
		}
		if !changed {
			return raw, false, nil
		}
		cleaned, err := json.Marshal(fields)
		return cleaned, true, err
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, false, err
		}
		changed := false
		for i, item := range items {
			cleaned, childChanged, err := stripThoughtSignaturesJSON(item, parent)
			if err != nil {
				return nil, false, err
			}
			if childChanged {
				items[i] = cleaned
				changed = true
			}
		}
		if !changed {
			return raw, false, nil
		}
		cleaned, err := json.Marshal(items)
		return cleaned, true, err
	default:
		return raw, false, nil
	}
}
