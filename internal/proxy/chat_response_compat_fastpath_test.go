package proxy

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The SSE model rewrite is the hottest path in the router: it runs once per
// streamed chunk of every response. Two changes came out of measuring it:
//
//   - the gate that skips the decode when the chunk cannot carry a model no
//     longer looks for "chat.completion.chunk" or "text_completion". By the
//     OpenAI spec every such event carries "object":"chat.completion.chunk",
//     so those needles were in every chunk and the gate never skipped
//     anything -- it only added four scans per chunk to a path that then
//     decoded the chunk anyway;
//   - the shallow decode is a plain Unmarshal rather than a streaming
//     Decoder (the latter builds a per-call RuntimeContext).
//
// These tests pin the behaviour both changes must preserve: the same
// outcomes for the same payloads, and the fast path actually taken.

// TestNormalizeSSEDataLineModelGateSkipsOnlyUnusableChunks pins which
// chunks skip the decode. A chunk with no "model", no "object" and no
// "response" cannot gain an alias and is left byte-for-byte alone. A chunk
// with any of them is decoded.
func TestNormalizeSSEDataLineModelGateSkipsOnlyUnusableChunks(t *testing.T) {
	// No model, no object, no response: nothing to rewrite.
	skipped := []byte("data: {\"id\":\"c1\",\"choices\":[]}\n\n")
	assert.Equal(t, skipped, normalizeSSEDataLineModelWithOptions(skipped, "alias", true),
		"a chunk with no model, object or response must be left alone")

	// An "object" naming a chunk shape: the alias is added, so the chunk
	// must NOT be skipped.
	added := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\"}\n\n")
	got := string(normalizeSSEDataLineModelWithOptions(added, "alias", true))
	assert.Contains(t, got, `"model":"alias"`,
		"a chunk with an object gets the alias added -- the gate must not skip it")

	// A "model" of its own: rewritten.
	own := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"upstream\"}\n\n")
	got = string(normalizeSSEDataLineModelWithOptions(own, "alias", true))
	assert.Contains(t, got, `"model":"alias"`,
		"a chunk with its own model has it rewritten")

	// A nested "response" object (the /v1/responses shape).
	nested := []byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"inner\"}}\n\n")
	got = string(normalizeSSEDataLineModelWithOptions(nested, "alias", false))
	assert.Contains(t, got, `"model":"alias"`,
		"a nested response object has its model rewritten")
}

// TestDecodeShallowJSONObjectMatchesTheDecoder pins that the Unmarshal-based
// decode accepts and rejects exactly what the streaming Decoder did: the
// same objects, the same rejections, and the same "not an object" verdicts.
func TestDecodeShallowJSONObjectMatchesTheDecoder(t *testing.T) {
	accepted := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m"}`,
		`{"error":{"message":"x"}}`,
		`{"type":"error"}`,
		`{}`,
		`{"a":"x","b":[1,2,3],"c":{"d":"e"}}`,
	}
	for _, body := range accepted {
		object, err := decodeShallowJSONObject([]byte(body))
		require.NoError(t, err, "body %s", body)
		require.NotNil(t, object, "body %s", body)
	}

	rejected := []string{
		`not json`,
		`{"a":"x"} trailing`,
		`{"a":"x"}{"b":"y"}`,
		`[1,2,3]`,
		`"a string"`,
	}
	for _, body := range rejected {
		object, err := decodeShallowJSONObject([]byte(body))
		require.Error(t, err, "body %s must be rejected", body)
		require.Nil(t, object, "body %s must yield no object", body)
	}
}

// TestDecodeShallowJSONObjectKeepsValuesOpaque pins that the values stay
// raw bytes: a multi-MB base64 image delta must not be expanded into a
// tree, and a large number must survive byte-for-byte.
func TestDecodeShallowJSONObjectKeepsValuesOpaque(t *testing.T) {
	big := strings.Repeat("A", 1<<20)
	body := `{"delta":{"image":"data:image/png;base64,` + big + `"},"n":9007199254740993}`
	object, err := decodeShallowJSONObject([]byte(body))
	require.NoError(t, err)
	raw, ok := object["n"]
	require.True(t, ok)
	assert.Equal(t, "9007199254740993", string(raw),
		"a large number must survive the shallow decode byte-for-byte")
}

// TestNormalizeSSEDataLineModelKeepsErrorEventsAndFraming pins the framing
// and the error-event pass-through the fast path must not disturb.
func TestNormalizeSSEDataLineModelKeepsErrorEventsAndFraming(t *testing.T) {
	for _, line := range []string{
		"data: {\"type\":\"error\",\"error\":{\"message\":\"x\"}}\n\n",
		"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"r\"}}\n\n",
		"data: [DONE]\n\n",
		"event: message\ndata: {\"id\":\"c1\"}\n\n",
	} {
		assert.Equal(t, line, string(normalizeSSEDataLineModelWithOptions([]byte(line), "alias", true)),
			"line %q must be retained byte-for-byte", line)
	}
}

// TestDecodeShallowJSONObjectRejectsTrailingContent is the specific
// difference between the two decoders: the Decoder reported "multiple JSON
// values", Unmarshal reports "invalid character after top-level value".
// Both reject; only the wording differs, and nothing depends on it.
func TestDecodeShallowJSONObjectRejectsTrailingContent(t *testing.T) {
	_, err := decodeShallowJSONObject([]byte(`{"a":"x"}{"b":"y"}`))
	require.Error(t, err)
	assert.NotContains(t, err.Error(), "multiple JSON values")
}

// TestNormalizeSSEDataLineModelRewritesAnEscapedModelKey pins the case the
// gate's backslash check exists for: a chunk whose "model" key is written
// with a unicode escape ("model" is "model") does not contain
// those bytes, so a gate that only looked for the literal key would skip
// it and the client would keep seeing the upstream's model name.
func TestNormalizeSSEDataLineModelRewritesAnEscapedModelKey(t *testing.T) {
	line := []byte("data: {\"\\u006dodel\":\"upstream-name\"}\n\n")
	got := string(normalizeSSEDataLineModelWithOptions(line, "public-alias", true))
	assert.NotEqual(t, string(line), got, "the escaped model key must be rewritten")
	assert.Contains(t, got, `"model":"public-alias"`,
		"the escaped model key must be rewritten to the client's alias")
}
