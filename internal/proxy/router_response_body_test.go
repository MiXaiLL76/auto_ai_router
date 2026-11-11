package proxy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The shape a router answers with, as it reaches the client -- the example
// from the feature request.
const routerResponseExample = `{
  "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "The model's own text", "role": "assistant"}}],
  "created": 1794422795,
  "id": "chatcmpl-msg_0140m0Ta3fVULjc3iSmSEOiN",
  "model": "anthropic/claude-opus-4.8",
  "object": "chat.completion",
  "usage": {"completion_tokens": 1, "prompt_tokens": 75410, "total_tokens": 75411}
}`

// TestRedactRouterResponseBody_MasksTextKeepsStructure is the core of the
// feature: the model's own text is replaced, and everything else -- the
// model, the usage, the ids, the finish reason -- survives.
//
// The distinction matters. Redacting the whole body (as the raw-bodies path
// does for a provider's error text) would leave nothing to read, and the
// whole point of capturing a response is its shape: which model answered,
// how it ended, how many tokens it used.
func TestRedactRouterResponseBody_MasksTextKeepsStructure(t *testing.T) {
	masked, ok := redactRouterResponseBody([]byte(routerResponseExample))
	require.True(t, ok, "the body must be redactable")
	assert.NotContains(t, masked, "The model's own text",
		"the model's output must not survive redaction")
	assert.Contains(t, masked, routerResponseRedactedPlaceholder)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(masked), &parsed))
	assert.Equal(t, "anthropic/claude-opus-4.8", parsed["model"])
	assert.Equal(t, "chat.completion", parsed["object"])
	assert.Equal(t, "chatcmpl-msg_0140m0Ta3fVULjc3iSmSEOiN", parsed["id"])
	assert.NotNil(t, parsed["usage"], "the usage must survive redaction")

	choices, ok := parsed["choices"].([]any)
	require.True(t, ok)
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]any)
	assert.Equal(t, "stop", choice["finish_reason"], "the finish reason must survive redaction")
	message := choice["message"].(map[string]any)
	assert.Equal(t, routerResponseRedactedPlaceholder, message["content"])
	assert.Equal(t, "assistant", message["role"], "the role must survive redaction")
}

// TestRedactRouterResponseBody_MasksToolArguments pins that the arguments of
// a tool call are masked too: they are the model's own output, even though
// the call's shape (which tool, which parameter) is what an operator reads.
func TestRedactRouterResponseBody_MasksToolArguments(t *testing.T) {
	body := `{
	  "choices": [{"index": 0, "message": {"role": "assistant", "tool_calls": [
	    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Moscow\"}"}}
	  ]}}],
	  "model": "gpt-4o"
	}`
	masked, ok := redactRouterResponseBody([]byte(body))
	require.True(t, ok)
	assert.NotContains(t, masked, "Moscow", "the tool arguments are the model's output and must be masked")
	assert.Contains(t, masked, "get_weather", "the tool's name must survive redaction")
	assert.Contains(t, masked, "call_1", "the tool call's id must survive redaction")
}

// TestRedactRouterResponseBody_KeepsSchemaShape guards the opposite
// direction: a key named "content" that is NOT the answer (a JSON-Schema
// property named "content") must not be masked away -- masking it would
// destroy the structure without protecting anything.
func TestRedactRouterResponseBody_KeepsSchemaShape(t *testing.T) {
	body := `{
	  "choices": [{"index": 0, "message": {"role": "assistant", "content": "the answer"}}],
	  "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}}]
	}`
	masked, ok := redactRouterResponseBody([]byte(body))
	require.True(t, ok)
	assert.NotContains(t, masked, "the answer")
	assert.Contains(t, masked, `"type":"string"`,
		"a schema property named content must keep its type")
}

// TestRedactRouterResponseBody_FailsClosed pins that a body which is not
// valid JSON is not shipped as-is: it might be a plain-text answer, and
// shipping it unredacted would leak the model's output on the one path
// where the function could not tell output from structure.
func TestRedactRouterResponseBody_FailsClosed(t *testing.T) {
	_, ok := redactRouterResponseBody([]byte("just a plain text answer"))
	assert.False(t, ok, "an unparseable body must not be reported as redactable")
}

// TestRedactRouterResponseBody_ResponsesShape pins the Responses API shape,
// whose output parts carry the answer under "text" inside "output".
func TestRedactRouterResponseBody_ResponsesShape(t *testing.T) {
	body := `{
	  "id": "resp_1", "object": "response", "model": "gpt-5",
	  "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "the answer"}]}],
	  "usage": {"input_tokens": 10, "output_tokens": 5}
	}`
	masked, ok := redactRouterResponseBody([]byte(body))
	require.True(t, ok)
	assert.NotContains(t, masked, "the answer")
	assert.Contains(t, masked, "resp_1")
	assert.Contains(t, masked, "output_text", "the part's type must survive redaction")
}

// TestRedactRouterResponseBody_MasksStringArray pins that a text field
// holding a list of strings is masked as a whole: each element is the
// model's own text.
func TestRedactRouterResponseBody_MasksStringArray(t *testing.T) {
	body := `{"choices": [{"index": 0, "message": {"role": "assistant", "content": ["part one", "part two"]}}]}`
	masked, ok := redactRouterResponseBody([]byte(body))
	require.True(t, ok)
	assert.NotContains(t, masked, "part one")
	assert.NotContains(t, masked, "part two")
}

// TestCaptureRouterResponseBody_RespectsRedactToggle pins the toggle
// itself: with redaction off the body is captured verbatim, with it on the
// text is masked.
func TestCaptureRouterResponseBody_RespectsRedactToggle(t *testing.T) {
	raw := captureRouterResponseBody([]byte(routerResponseExample), false)
	assert.Contains(t, raw, "The model's own text",
		"with redaction off the body must be captured verbatim")

	masked := captureRouterResponseBody([]byte(routerResponseExample), true)
	assert.NotContains(t, masked, "The model's own text",
		"with redaction on the model's text must be masked")
	assert.Contains(t, masked, routerResponseRedactedPlaceholder)
}

// TestCaptureRouterResponseBody_FailsClosedOnUnparseable pins the same
// fail-closed rule at the capture level, where it matters most: the
// unparseable body must not reach the publish path in the clear.
func TestCaptureRouterResponseBody_FailsClosedOnUnparseable(t *testing.T) {
	captured := captureRouterResponseBody([]byte("a plain text answer"), true)
	assert.Equal(t, routerResponseRedactedPlaceholder, captured,
		"an unparseable body must not be captured in the clear")
}

// TestCaptureRouterResponseBody_CapsLength pins the size cap, and that the
// marker says the body was cut -- so a reader never believes a truncated
// JSON body is complete.
func TestCaptureRouterResponseBody_CapsLength(t *testing.T) {
	huge := make([]byte, maxRouterResponseBodyBytes+1000)
	for i := range huge {
		huge[i] = 'a'
	}
	captured := captureRouterResponseBody(huge, false)
	assert.Less(t, len(captured), maxRouterResponseBodyBytes+100,
		"the captured body must be capped")
	assert.Contains(t, captured, "truncated", "the truncation must be visible")
}

// TestCaptureRouterResponseBody_EmptyBody pins that an empty body is
// captured as empty (and not published at all -- see the publish gate).
func TestCaptureRouterResponseBody_EmptyBody(t *testing.T) {
	assert.Empty(t, captureRouterResponseBody(nil, true))
	assert.Empty(t, captureRouterResponseBody([]byte{}, false))
}
