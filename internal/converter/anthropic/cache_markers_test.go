package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCountCacheMarkers_ToolsOnly(t *testing.T) {
	// The exact gap this was built for: InjectCacheControlOutcome reports
	// "none" for this body (no messages/system marker), but the client did
	// mark something -- a tool definition -- and it does reach the provider.
	body := []byte(`{
		"model": "claude-haiku-5-5",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "f", "parameters": {}}, "cache_control": {"type": "ephemeral"}}
		]
	}`)

	counts := CountCacheMarkers(body)
	assert.Equal(t, CacheMarkerCounts{System: 0, Messages: 0, Tools: 1, TTL1h: 0}, counts)
}

func TestCountCacheMarkers_SystemAndMessages(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"system": [{"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
			{"role": "assistant", "content": "b"},
			{"role": "user", "content": "c"}
		]
	}`)

	counts := CountCacheMarkers(body)
	assert.Equal(t, CacheMarkerCounts{System: 1, Messages: 1, Tools: 0, TTL1h: 1}, counts)
}

func TestCountCacheMarkers_MultipleToolsAndTTLMix(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "a"}, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
			{"type": "function", "function": {"name": "b"}, "cache_control": {"type": "ephemeral"}},
			{"type": "function", "function": {"name": "c"}}
		]
	}`)

	counts := CountCacheMarkers(body)
	assert.Equal(t, 2, counts.Tools)
	assert.Equal(t, 1, counts.TTL1h)
	assert.Equal(t, 2, counts.Total())
}

func TestCountCacheMarkers_ResponsesAPIShape(t *testing.T) {
	// Responses API: "instructions" instead of "system", "input" instead of
	// "messages" -- CountCacheMarkers must find the same markers there too.
	body := []byte(`{
		"model": "claude",
		"instructions": [{"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}],
		"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
		"tools": [{"type": "web_search", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
	}`)

	counts := CountCacheMarkers(body)
	assert.Equal(t, CacheMarkerCounts{System: 1, Messages: 1, Tools: 1, TTL1h: 1}, counts)
}

func TestCountCacheMarkers_NoMarkersAtAll(t *testing.T) {
	body := []byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, CacheMarkerCounts{}, CountCacheMarkers(body))
	assert.Equal(t, 0, CountCacheMarkers(body).Total())
}

func TestCountCacheMarkers_MalformedJSON(t *testing.T) {
	assert.Equal(t, CacheMarkerCounts{}, CountCacheMarkers([]byte("not json")))
}

func TestCountCacheMarkers_DoesNotRecurseIntoCacheControlValueItself(t *testing.T) {
	// A pathological/malicious body nesting a "cache_control"-looking key
	// inside the marker's own value must not be double-counted.
	body := []byte(`{
		"model": "claude",
		"system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h", "cache_control": {"ttl": "1h"}}}]
	}`)
	counts := CountCacheMarkers(body)
	assert.Equal(t, 1, counts.System)
	assert.Equal(t, 1, counts.TTL1h)
}

// TestCountCacheMarkers_SchemaPropertyNamedCacheControlIsNotAMarker guards
// against the exact false positive a code review caught: a tool's JSON
// Schema is free to declare a parameter literally called "cache_control",
// e.g. {"properties":{"cache_control":{"type":"string"}}} -- that is not an
// Anthropic cache_control marker (no "type":"ephemeral"), and must not be
// counted as one.
func TestCountCacheMarkers_SchemaPropertyNamedCacheControlIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "f", "parameters": {
				"type": "object",
				"properties": {"cache_control": {"type": "string"}}
			}}}
		]
	}`)
	assert.Equal(t, CacheMarkerCounts{}, CountCacheMarkers(body))
}

// TestCountCacheMarkers_NullCacheControlIsNotAMarker guards against the
// other false positive the same review caught: a present-but-null
// "cache_control" key (map lookup's ok is still true for a JSON null) must
// not be counted.
func TestCountCacheMarkers_NullCacheControlIsNotAMarker(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "f"}, "cache_control": null}
		]
	}`)
	assert.Equal(t, CacheMarkerCounts{}, CountCacheMarkers(body))
}
