package anthropic

import "encoding/json"

var cacheEphemeral = map[string]interface{}{"type": "ephemeral"}

// CacheControlOutcome reports what InjectCacheControlOutcome found/did, so
// callers can tell a client's own cache_control usage apart from the
// router's automatic injection when logging cache-eligibility signals.
type CacheControlOutcome string

const (
	// CacheControlNone means neither the client nor the router placed any
	// cache_control marker (body had no messages/system, or malformed JSON).
	CacheControlNone CacheControlOutcome = "none"
	// CacheControlClientSupplied means the body already carried a
	// cache_control marker before this call; it was left untouched.
	CacheControlClientSupplied CacheControlOutcome = "client_supplied"
	// CacheControlRouterInjected means this call added the marker(s) itself.
	CacheControlRouterInjected CacheControlOutcome = "router_injected"
)

// InjectCacheControl adds Anthropic prompt-caching markers to a request body,
// in either OpenAI-format (system prompt is a message with role "system"/"developer")
// or native Anthropic Messages API format (system prompt is a top-level "system" field).
//
// It marks two cache breakpoints (standard Anthropic multi-turn caching pattern):
//  1. Last content block of the system prompt (stable across turns)
//  2. Last content block of the second-to-last user message (history boundary)
//
// If the body already contains any cache_control field, it is returned unchanged.
// Non-JSON or structurally unexpected bodies are also returned unchanged.
func InjectCacheControl(body []byte) []byte {
	result, _ := injectCacheControl(body)
	return result
}

// InjectCacheControlOutcome behaves exactly like InjectCacheControl, but
// additionally reports the outcome -- distinguishing "the client already
// sent its own cache_control" from "the router added it" from "neither" --
// so a caller can log which signal, if any, drove this request's caching.
func InjectCacheControlOutcome(body []byte) ([]byte, CacheControlOutcome) {
	return injectCacheControl(body)
}

func injectCacheControl(body []byte) ([]byte, CacheControlOutcome) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, CacheControlNone
	}

	messages, _ := req["messages"].([]interface{})
	hasSystemField := req["system"] != nil
	if len(messages) == 0 && !hasSystemField {
		return body, CacheControlNone
	}

	if hasAnyCacheControl(messages) || contentHasCacheControl(req["system"]) {
		return body, CacheControlClientSupplied
	}

	modified := hasSystemField && markSystemField(req)

	for _, msg := range messages {
		m, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "system" || role == "developer" {
			if markLastContentBlock(m) {
				modified = true
			}
		}
	}

	// Collect user message indices; mark the second-to-last one (history boundary).
	var userIdxs []int
	for i, msg := range messages {
		m, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := m["role"].(string); role == "user" {
			userIdxs = append(userIdxs, i)
		}
	}
	if len(userIdxs) >= 2 {
		histMsg, ok := messages[userIdxs[len(userIdxs)-2]].(map[string]interface{})
		if ok && markLastContentBlock(histMsg) {
			modified = true
		}
	}

	if !modified {
		return body, CacheControlNone
	}

	result, err := json.Marshal(req)
	if err != nil {
		return body, CacheControlNone
	}
	return result, CacheControlRouterInjected
}

// hasAnyCacheControl reports whether any content block in messages already carries
// a cache_control field.
func hasAnyCacheControl(messages []interface{}) bool {
	for _, msg := range messages {
		m, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if contentHasCacheControl(m["content"]) {
			return true
		}
	}
	return false
}

func contentHasCacheControl(content interface{}) bool {
	blocks, ok := content.([]interface{})
	if !ok {
		return false
	}
	for _, block := range blocks {
		b, ok := block.(map[string]interface{})
		if ok && b["cache_control"] != nil {
			return true
		}
	}
	return false
}

// markLastContentBlock adds cache_control to the last block of a message's content.
// String content is promoted to a single-element text block array so the marker
// survives the OpenAI→Anthropic conversion.
func markLastContentBlock(msg map[string]interface{}) bool {
	newContent, ok := markLastBlockValue(msg["content"])
	if ok {
		msg["content"] = newContent
	}
	return ok
}

// markSystemField adds cache_control to the last block of a native Anthropic
// Messages API request's top-level "system" field (string or content-block array).
func markSystemField(req map[string]interface{}) bool {
	newSystem, ok := markLastBlockValue(req["system"])
	if ok {
		req["system"] = newSystem
	}
	return ok
}

// markLastBlockValue promotes a string content/system value to a single-element
// text block array carrying cache_control, or marks the last block of an
// existing content-block array. Returns the (possibly new) value and whether
// it was modified.
func markLastBlockValue(value interface{}) (interface{}, bool) {
	switch c := value.(type) {
	case string:
		if c == "" {
			return value, false
		}
		return []interface{}{
			map[string]interface{}{
				"type":          "text",
				"text":          c,
				"cache_control": cacheEphemeral,
			},
		}, true
	case []interface{}:
		if len(c) == 0 {
			return value, false
		}
		last, ok := c[len(c)-1].(map[string]interface{})
		if !ok {
			return value, false
		}
		last["cache_control"] = cacheEphemeral
		return value, true
	}
	return value, false
}
