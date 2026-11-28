package anthropic

import (
	"bytes"
	"encoding/json"
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

// injectCacheControl decides and applies the router's cache breakpoints
// without decoding the conversation.
//
// The decision reads very little: the role of each message, whether a
// message or a content block already carries a marker, and the shape of the
// system prompt. The change touches at most the system prompt, the
// system/developer messages and one user message. So the body is scanned,
// the messages array is scanned once more for its items and their roles, and
// only the messages that are inspected for an existing marker or that
// receive a new one are decoded -- by the same helpers the whole-body
// implementation (injectCacheControlDecode) uses, which remains the fallback
// and the reference in tests.
func injectCacheControl(body []byte) ([]byte, CacheControlOutcome) {
	obj, ok := jsonscan.ParseObject(body)
	if !ok {
		return injectCacheControlDecode(body)
	}
	top := obj.Top()

	systemRaw, hasSystemKey := obj.Fields["system"]
	hasSystemField := hasSystemKey && !jsonscan.IsNull(systemRaw)

	// messages that is not an array counts as no messages.
	messages, roles, isArray := jsonscan.ScanArrayMembers(obj.Fields["messages"], "role")
	if !isArray {
		messages, roles = nil, nil
	}
	if len(messages) == 0 && !hasSystemField {
		return body, CacheControlNone
	}

	// Did the client place a marker itself? A marker's key is the text
	// cache_control unless some string in the body uses an escape, so a
	// message that does not contain that text has none.
	mayHide := top.EscapedASCII()
	for i, raw := range messages {
		if len(raw) == 0 || raw[0] != '{' || (!mayHide && !bytes.Contains(raw, cacheControlKeyBytes)) {
			continue
		}
		supplied, decoded := messageHasCacheControl(raw, roles[i])
		if !decoded {
			return injectCacheControlDecode(body)
		}
		if supplied {
			return body, CacheControlClientSupplied
		}
	}
	if hasSystemField && (mayHide || bytes.Contains(systemRaw, cacheControlKeyBytes)) {
		var system interface{}
		if err := json.Unmarshal(systemRaw, &system); err != nil {
			return injectCacheControlDecode(body)
		}
		if contentHasCacheControl(system) {
			return body, CacheControlClientSupplied
		}
	}

	modified := false

	// Breakpoint 1: the system prompt, as the top-level field ...
	if hasSystemField {
		var system interface{}
		if err := json.Unmarshal(systemRaw, &system); err != nil {
			return injectCacheControlDecode(body)
		}
		if marked, ok := markLastBlockValue(system); ok {
			encoded, err := json.Marshal(marked)
			if err != nil {
				return body, CacheControlNone
			}
			obj.Fields["system"] = encoded
			modified = true
		}
	}

	// ... and as system/developer messages. Breakpoint 2: the second-to-last
	// user message. Each of these messages is decoded, marked and re-encoded
	// on its own; a message can be both (never in practice) and is then
	// marked once, as marking is idempotent.
	var toMark []int
	var userIdxs []int
	for i, raw := range messages {
		if len(raw) == 0 || raw[0] != '{' {
			continue
		}
		role, decoded := rawJSONStringValue(roles[i])
		if !decoded {
			return injectCacheControlDecode(body)
		}
		switch role {
		case "system", "developer":
			toMark = append(toMark, i)
		case "user":
			userIdxs = append(userIdxs, i)
		}
	}
	if len(userIdxs) >= 2 {
		toMark = append(toMark, userIdxs[len(userIdxs)-2])
	}
	messagesChanged := false
	for _, i := range toMark {
		var message map[string]interface{}
		if err := json.Unmarshal(messages[i], &message); err != nil {
			return injectCacheControlDecode(body)
		}
		if !markLastContentBlock(message) {
			continue
		}
		encoded, err := json.Marshal(message)
		if err != nil {
			return body, CacheControlNone
		}
		messages[i] = encoded
		messagesChanged = true
	}
	if messagesChanged {
		obj.Fields["messages"] = jsonscan.JoinArray(messages)
		modified = true
	}

	if !modified {
		return body, CacheControlNone
	}
	// A body that gets a marker used to be decoded and encoded again whole,
	// which also replaced any invalid UTF-8 in it; such a body still is.
	if !utf8.Valid(body) {
		return injectCacheControlDecode(body)
	}
	return obj.Encode(), CacheControlRouterInjected
}

// messageHasCacheControl is hasAnyCacheControl for one message given as raw
// JSON (an object) and its raw "role": a tool message carrying a marker
// itself, or any message with a marked content block. decoded=false means a
// value did not decode and the caller should decode the body instead.
func messageHasCacheControl(raw, roleRaw []byte) (supplied, decoded bool) {
	top, ok := jsonscan.ScanTop(raw)
	if !ok {
		return false, false
	}
	role, ok := rawJSONStringValue(roleRaw)
	if !ok {
		return false, false
	}
	if role == "tool" {
		if markerRaw, has := top.Raw("cache_control"); has {
			var marker interface{}
			if err := json.Unmarshal(markerRaw, &marker); err != nil {
				return false, false
			}
			if _, isMarker := asEphemeralCacheControl(marker); isMarker {
				return true, true
			}
		}
	}
	contentRaw, has := top.Raw("content")
	if !has || len(contentRaw) == 0 || contentRaw[0] != '[' {
		return false, true // contentHasCacheControl looks at block arrays only
	}
	_, markers, ok := jsonscan.ScanArrayMembers(contentRaw, "cache_control")
	if !ok {
		return false, false
	}
	for _, markerRaw := range markers {
		if markerRaw == nil {
			continue
		}
		var marker interface{}
		if err := json.Unmarshal(markerRaw, &marker); err != nil {
			return false, false
		}
		if _, isMarker := asEphemeralCacheControl(marker); isMarker {
			return true, true
		}
	}
	return false, true
}

// rawJSONStringValue returns the string a raw JSON value decodes to, "" when
// it is absent or not a string -- what `v, _ := m[key].(string)` yields on a
// decoded object. ok=false: the value does not decode.
func rawJSONStringValue(raw []byte) (value string, ok bool) {
	switch {
	case len(raw) == 0 || raw[0] != '"':
		return "", true
	case bytes.IndexByte(raw, '\\') < 0:
		return string(raw[1 : len(raw)-1]), true
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}
	return value, true
}
