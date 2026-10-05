package openai

import (
	"bytes"
	"encoding/json"
)

// DropEmptyTools removes "tools": [] together with the fields that only make sense
// next to tools (tool_choice, parallel_tool_calls). Several OpenAI-compatible
// servers, vLLM among them, reject an empty tools array with a 400 although the
// request means exactly "no tools". Works on Chat Completions and Responses bodies
// alike. Returns body unchanged when tools is absent, non-empty or not an array.
func DropEmptyTools(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"tools"`)) {
		return body
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return body
	}
	var tools []json.RawMessage
	raw, ok := top["tools"]
	if !ok || json.Unmarshal(raw, &tools) != nil || tools == nil || len(tools) > 0 {
		return body
	}
	delete(top, "tools")
	delete(top, "tool_choice")
	delete(top, "parallel_tool_calls")
	out, err := json.Marshal(top)
	if err != nil {
		return body
	}
	return out
}

// MapReasoningEffort rewrites every reasoning effort the request carries through
// resolve: the Chat Completions reasoning_effort, the vLLM
// chat_template_kwargs.reasoning_effort and the Responses reasoning.effort. resolve
// returns the new value and whether it changed. Non-string values are left alone.
func MapReasoningEffort(body []byte, resolve func(string) (string, bool)) []byte {
	if resolve == nil || !bytes.Contains(body, []byte("effort")) {
		return body
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return body
	}
	changed := mapEffortField(top, "reasoning_effort", resolve)
	changed = mapNestedEffortField(top, "chat_template_kwargs", "reasoning_effort", resolve) || changed
	changed = mapNestedEffortField(top, "reasoning", "effort", resolve) || changed
	if !changed {
		return body
	}
	out, err := json.Marshal(top)
	if err != nil {
		return body
	}
	return out
}

func mapEffortField(obj map[string]json.RawMessage, key string, resolve func(string) (string, bool)) bool {
	raw, ok := obj[key]
	if !ok {
		return false
	}
	var value string
	if json.Unmarshal(raw, &value) != nil {
		return false
	}
	mapped, changed := resolve(value)
	if !changed {
		return false
	}
	encoded, err := json.Marshal(mapped)
	if err != nil {
		return false
	}
	obj[key] = encoded
	return true
}

func mapNestedEffortField(top map[string]json.RawMessage, parent, key string, resolve func(string) (string, bool)) bool {
	raw, ok := top[parent]
	if !ok {
		return false
	}
	var nested map[string]json.RawMessage
	if json.Unmarshal(raw, &nested) != nil || nested == nil {
		return false
	}
	if !mapEffortField(nested, key, resolve) {
		return false
	}
	encoded, err := json.Marshal(nested)
	if err != nil {
		return false
	}
	top[parent] = encoded
	return true
}
