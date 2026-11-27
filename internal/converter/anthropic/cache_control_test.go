package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func parseBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &m))
	return m
}

func TestInjectCacheControlOutcome_ToolMessageMarker(t *testing.T) {
	body := []byte(`{
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "tool", "tool_call_id": "call_1", "content": "result", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
		]
	}`)

	result, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlClientSupplied, outcome)
	assert.Equal(t, body, result)
}

func TestInjectCacheControl_SingleTurn(t *testing.T) {
	// Only one user message → no history to mark; system gets marked.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello"}
		]
	}`)

	result := InjectCacheControl(body)
	m := parseBody(t, result)
	messages := m["messages"].([]interface{})

	sys := messages[0].(map[string]interface{})
	sysContent := sys["content"].([]interface{})
	require.Len(t, sysContent, 1)
	block := sysContent[0].(map[string]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, block["cache_control"])
	assert.Equal(t, "You are helpful.", block["text"])

	// Only one user message → no second-to-last user, so user not marked.
	user := messages[1].(map[string]interface{})
	assert.Equal(t, "Hello", user["content"], "single user message content must stay as string")
}

func TestInjectCacheControl_MultiTurn(t *testing.T) {
	// Two user messages → second-to-last user (first) gets marked.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": "First question"},
			{"role": "assistant", "content": "First answer"},
			{"role": "user", "content": "Current question"}
		]
	}`)

	result := InjectCacheControl(body)
	m := parseBody(t, result)
	messages := m["messages"].([]interface{})

	// System marked
	sys := messages[0].(map[string]interface{})
	sysContent := sys["content"].([]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, sysContent[0].(map[string]interface{})["cache_control"])

	// First user message (history boundary) marked
	firstUser := messages[1].(map[string]interface{})
	firstUserContent := firstUser["content"].([]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, firstUserContent[0].(map[string]interface{})["cache_control"])

	// Current (last) user message NOT marked
	lastUser := messages[3].(map[string]interface{})
	assert.Equal(t, "Current question", lastUser["content"], "current user message must not be modified")

	// Assistant message NOT marked
	asst := messages[2].(map[string]interface{})
	assert.Equal(t, "First answer", asst["content"])
}

func TestInjectCacheControl_ArrayContent(t *testing.T) {
	// Multi-block user content: cache_control goes on LAST block only.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "Block 1"},
				{"type": "text", "text": "Block 2"}
			]},
			{"role": "assistant", "content": "ok"},
			{"role": "user", "content": "Current"}
		]
	}`)

	result := InjectCacheControl(body)
	m := parseBody(t, result)
	messages := m["messages"].([]interface{})

	firstUserBlocks := messages[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, firstUserBlocks, 2)
	assert.Nil(t, firstUserBlocks[0].(map[string]interface{})["cache_control"], "only last block marked")
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, firstUserBlocks[1].(map[string]interface{})["cache_control"])
}

func TestInjectCacheControl_AlreadyPresent(t *testing.T) {
	// Body with existing cache_control must be returned unchanged.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "Q", "cache_control": {"type": "ephemeral"}}
			]},
			{"role": "user", "content": "Current"}
		]
	}`)

	result := InjectCacheControl(body)
	// Must be identical bytes (same JSON).
	m := parseBody(t, result)
	messages := m["messages"].([]interface{})
	// Original cache_control untouched; no extra injection.
	block := messages[0].(map[string]interface{})["content"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, block["cache_control"])
	// Current user message must not have been marked.
	assert.Equal(t, "Current", messages[1].(map[string]interface{})["content"])
}

func TestInjectCacheControl_NoSystem(t *testing.T) {
	// No system message: only history user message marked.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "user", "content": "History"},
			{"role": "assistant", "content": "Reply"},
			{"role": "user", "content": "Current"}
		]
	}`)

	result := InjectCacheControl(body)
	m := parseBody(t, result)
	messages := m["messages"].([]interface{})

	histUser := messages[0].(map[string]interface{})["content"].([]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, histUser[0].(map[string]interface{})["cache_control"])

	assert.Equal(t, "Current", messages[2].(map[string]interface{})["content"])
}

func TestInjectCacheControl_NativeSystemField(t *testing.T) {
	// Native Anthropic Messages API requests carry the system prompt in a
	// top-level "system" field, not as a role=="system" message.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": "You are helpful.",
		"messages": [
			{"role": "user", "content": "First question"},
			{"role": "assistant", "content": "First answer"},
			{"role": "user", "content": "Second question"}
		]
	}`)

	result := InjectCacheControl(body)
	m := parseBody(t, result)

	sysContent := m["system"].([]interface{})
	require.Len(t, sysContent, 1)
	block := sysContent[0].(map[string]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, block["cache_control"])
	assert.Equal(t, "You are helpful.", block["text"])

	messages := m["messages"].([]interface{})
	histUser := messages[0].(map[string]interface{})["content"].([]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, histUser[0].(map[string]interface{})["cache_control"])
	assert.Equal(t, "Second question", messages[2].(map[string]interface{})["content"], "current turn must stay unmarked")
}

func TestInjectCacheControl_NativeSystemFieldAlreadyMarked(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": [{"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}],
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	assert.Equal(t, body, InjectCacheControl(body))
}

func TestInjectCacheControl_InvalidJSON(t *testing.T) {
	body := []byte(`not json`)
	assert.Equal(t, body, InjectCacheControl(body))
}

func TestInjectCacheControl_EmptyMessages(t *testing.T) {
	body := []byte(`{"model":"claude","messages":[]}`)
	assert.Equal(t, body, InjectCacheControl(body))
}

// Found in production: cache_control_outcome='client_supplied' on ~87% of a
// sample where CountCacheMarkers (cache_markers.go) independently counted
// zero markers anywhere in the same body -- this function's own detection
// was less strict than that diagnostic's, so a bogus "cache_control" field
// (an empty object some client/SDK round-trips on every block regardless of
// whether caching was requested, rather than omitting the key) made
// InjectCacheControl believe a marker was already there and skip injecting
// its own, silently losing caching for that request.
func TestInjectCacheControlOutcome_EmptyCacheControlObjectIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": "First question"},
			{"role": "assistant", "content": "First answer"},
			{"role": "user", "content": [
				{"type": "text", "text": "Current question", "cache_control": {}}
			]}
		]
	}`)

	result, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlRouterInjected, outcome,
		"an empty cache_control object is not a real marker: the router must still inject its own")

	m := parseBody(t, result)
	messages := m["messages"].([]interface{})
	sysContent := messages[0].(map[string]interface{})["content"].([]interface{})
	assert.Equal(t, map[string]interface{}{"type": "ephemeral"}, sysContent[0].(map[string]interface{})["cache_control"])
}

func TestInjectCacheControlOutcome_NullCacheControlIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": [
				{"type": "text", "text": "Hello", "cache_control": null}
			]}
		]
	}`)

	_, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlRouterInjected, outcome)
}

func TestInjectCacheControlOutcome_NonEphemeralCacheControlIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": [
				{"type": "text", "text": "Hello", "cache_control": {"type": "persistent"}}
			]}
		]
	}`)

	_, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlRouterInjected, outcome,
		"only type:ephemeral is a real Anthropic cache_control marker")
}

func TestInjectCacheControlOutcome_NativeSystemFieldEmptyCacheControlObjectIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": [{"type": "text", "text": "Be concise.", "cache_control": {}}],
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	_, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlRouterInjected, outcome)
}

func TestInjectCacheControlOutcome_RealMarkerStillDetectedAsClientSupplied(t *testing.T) {
	// Sanity check alongside the false-positive tests above: a genuine
	// ephemeral marker must still be recognized as client-supplied.
	body := []byte(`{
		"model": "claude-3-5-sonnet",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "Q", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
			]},
			{"role": "user", "content": "Current"}
		]
	}`)

	result, outcome := InjectCacheControlOutcome(body)
	assert.Equal(t, CacheControlClientSupplied, outcome)
	assert.Equal(t, body, result, "client_supplied must leave the body untouched")
}
