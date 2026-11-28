package openai

import (
	"bytes"
	"encoding/json"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
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
	obj, ok := jsonscan.ParseObject(body)
	if !ok {
		return body
	}
	dropEmptyToolsFields(obj.Fields)
	if !obj.Changed() {
		return body
	}
	return obj.Encode()
}

// dropEmptyToolsFields is DropEmptyTools on a request's top-level members.
func dropEmptyToolsFields(top map[string]json.RawMessage) {
	var tools []json.RawMessage
	raw, ok := top["tools"]
	if !ok || json.Unmarshal(raw, &tools) != nil || tools == nil || len(tools) > 0 {
		return
	}
	delete(top, "tools")
	delete(top, "tool_choice")
	delete(top, "parallel_tool_calls")
}

// DropNullParams removes the listed top-level keys whose value is JSON null. A client
// that sends "max_tokens": null means "no limit set", but a key that is present
// counts as set for ApplyDefaultParams, so the default would never be filled in.
// Returns body unchanged when none of the keys is null or the body is not a JSON object.
func DropNullParams(body []byte, keys ...string) []byte {
	if !bytes.Contains(body, []byte("null")) {
		return body
	}
	obj, ok := jsonscan.ParseObject(body)
	if !ok {
		return body
	}
	top := obj.Fields
	changed := false
	for _, key := range keys {
		if raw, ok := top[key]; ok && bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			delete(top, key)
			changed = true
		}
	}
	if !changed {
		return body
	}
	return obj.Encode()
}

// MapReasoningEffort rewrites every reasoning effort the request carries through
// resolve: the Chat Completions reasoning_effort, the vLLM
// chat_template_kwargs.reasoning_effort and the Responses reasoning.effort. resolve
// returns the new value and whether it changed. Non-string values are left alone.
func MapReasoningEffort(body []byte, resolve func(string) (string, bool)) []byte {
	if resolve == nil || !bytes.Contains(body, []byte("effort")) {
		return body
	}
	obj, ok := jsonscan.ParseObject(body)
	if !ok {
		return body
	}
	mapReasoningEffortFields(obj.Fields, resolve)
	if !obj.Changed() {
		return body
	}
	return obj.Encode()
}

// mapReasoningEffortFields is MapReasoningEffort on a request's top-level
// members.
func mapReasoningEffortFields(top map[string]json.RawMessage, resolve func(string) (string, bool)) {
	if resolve == nil {
		return
	}
	mapEffortField(top, "reasoning_effort", resolve)
	mapNestedEffortField(top, "chat_template_kwargs", "reasoning_effort", resolve)
	mapNestedEffortField(top, "reasoning", "effort", resolve)
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
