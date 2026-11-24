package proxy

import (
	"bytes"
	"errors"
	"io"
	"math/rand"
	"strings"
	"testing"

	goccyjson "github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The rewrite of the model name on a streamed chunk runs once per chunk of
// every response, so the two things it does -- the gate that skips the
// decode, and the shallow decode itself -- must decide the same thing as
// before, whatever the provider sends.
//
// These tests pin that equivalence against the previous implementation, so
// a later change to either half cannot silently alter what a client sees.

// previousDecode is the shallow decode as it was before it became an
// Unmarshal: a streaming Decoder plus a second read that rejects trailing
// content.
func previousDecode(data []byte) (map[string]goccyjson.RawMessage, error) {
	decoder := goccyjson.NewDecoder(bytes.NewReader(data))
	var object map[string]goccyjson.RawMessage
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	var trailing interface{}
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return object, nil
}

// previousGate is the gate as it was: it looked for the VALUES of the
// "object" field ("chat.completion.chunk", "text_completion") rather than
// for the field itself, and so never skipped anything.
func previousGate(payload []byte) bool {
	return !bytes.Contains(payload, []byte(`"model"`)) &&
		!bytes.Contains(payload, []byte(`"response"`)) &&
		!bytes.Contains(payload, []byte("chat.completion.chunk")) &&
		!bytes.Contains(payload, []byte("text_completion")) &&
		bytes.IndexByte(payload, '\\') < 0
}

// TestShallowDecodeDecidesAsThePreviousOne pins that the Unmarshal-based
// decode accepts and rejects exactly what the streaming decoder did.
func TestShallowDecodeDecidesAsThePreviousOne(t *testing.T) {
	accepted := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m"}`,
		`{"error":{"message":"x"}}`,
		`{"type":"error"}`,
		`{"type":"response.failed"}`,
		`{"error":null}`,
		`{"model":"a","model":"b"}`,
		`{}`,
		`{"a":"x","b":[1,2,3],"c":{"d":"e"}}`,
	}
	for _, body := range accepted {
		previous, previousErr := previousDecode([]byte(body))
		current, currentErr := decodeShallowJSONObject([]byte(body))
		require.NoError(t, previousErr, "body %s", body)
		require.NoError(t, currentErr, "body %s", body)
		require.NotNil(t, previous)
		require.NotNil(t, current)
		require.Len(t, current, len(previous), "body %s", body)
		for key, value := range previous {
			require.Equal(t, string(value), string(current[key]), "body %s key %q", body, key)
		}
	}

	rejected := []string{
		`not json`,
		`{"a":"x"} trailing`,
		`{"a":"x"}{"b":"y"}`,
		`[1,2,3]`,
		`"a string"`,
	}
	for _, body := range rejected {
		previous, previousErr := previousDecode([]byte(body))
		current, currentErr := decodeShallowJSONObject([]byte(body))
		require.Error(t, previousErr, "body %s", body)
		require.Error(t, currentErr, "body %s", body)
		require.Nil(t, previous)
		require.Nil(t, current)
	}

	// A JSON null decodes to no map and no error in both -- the caller
	// treats a nil map as "nothing here" either way.
	previous, previousErr := previousDecode([]byte("null"))
	current, currentErr := decodeShallowJSONObject([]byte("null"))
	require.NoError(t, previousErr)
	require.NoError(t, currentErr)
	require.Nil(t, previous)
	require.Nil(t, current)
}

// TestModelRewriteGateDecidesAsThePreviousOne pins that the gate still
// reaches the same verdict. The two gates differ on one class of payload --
// a chunk whose TEXT mentions "chat.completion.chunk" without carrying an
// "object" field -- and there the previous gate decoded the chunk for
// nothing while the new one skips it. Both leave the chunk untouched, so
// the outcome is the same.
func TestModelRewriteGateDecidesAsThePreviousOne(t *testing.T) {
	payloads := []string{
		`{"id":"c1","object":"chat.completion.chunk","created":1}`,
		`{"id":"c1","object":"text_completion","created":1}`,
		`{"id":"c1","object":"something.else"}`,
		`{"id":"c1","choices":[]}`,
		`{"id":"c1"}`,
		`{}`,
		`{"model":"m","choices":[]}`,
		`{"response":{"id":"r"}}`,
		`{"note":"see chat.completion.chunk docs"}`,
		`{"note":"a\\b"}`,
	}
	for _, payload := range payloads {
		previous, current := previousGate([]byte(payload)), currentGate([]byte(payload))
		if previous != current {
			// The only permitted difference: the previous gate decoded a
			// chunk that had nothing to rewrite. The output must be the
			// same either way, which the loop below checks.
			line := []byte("data: " + payload + "\n\n")
			require.Equal(t, string(normalizeSSEDataLineModelWithOptions(line, "alias", true)),
				string(previousNormalize(line, "alias", true)),
				"payload %s", payload)
			continue
		}
		assert.Equal(t, previous, current, "payload %s", payload)
	}
}

// TestModelRewriteOutputIsUnchangedForRandomChunks compares the whole
// function's output, byte for byte, against the previous implementation
// over randomly assembled chunks.
func TestModelRewriteOutputIsUnchangedForRandomChunks(t *testing.T) {
	pieces := []string{
		`"model":"m"`, `"object":"chat.completion.chunk"`, `"object":"text_completion"`,
		`"object":"other"`, `"response":{"id":"r","model":"inner"}`, `"error":{"message":"x"}`,
		`"type":"error"`, `"type":"response.failed"`, `"error":null`, `"choices":[]`,
		`"id":"c1"`, `"delta":{"content":"hi"}`, `"model":"alias"`,
		`"note":"chat.completion.chunk"`, `"x":"a\\b"`, `"usage":{"prompt_tokens":1}`,
		`"model":"m","model":"n"`, `9007199254740993`, `"caf\u00e9"`, `"<tag>"`,
	}
	rng := rand.New(rand.NewSource(20261123))
	for i := 0; i < 5000; i++ {
		body := "{"
		for j := 0; j < rng.Intn(5)+1; j++ {
			if j > 0 {
				body += ","
			}
			body += pieces[rng.Intn(len(pieces))]
		}
		body += "}"
		line := append([]byte("data: "), body...)
		line = append(line, '\n', '\n')
		for _, preserveNested := range []bool{true, false} {
			require.Equal(t, string(previousNormalize(line, "public-alias", preserveNested)),
				string(normalizeSSEDataLineModelWithOptions(line, "public-alias", preserveNested)),
				"chunk %s", body)
		}
	}
}

// TestModelRewriteOutputIsUnchangedForLargeChunks covers the shapes the
// shallow decode exists for: a multi-MB base64 delta, deep nesting, huge
// numbers and escapes.
func TestModelRewriteOutputIsUnchangedForLargeChunks(t *testing.T) {
	shapes := []string{
		`{"model":"m","delta":{"image":"data:image/png;base64,` + strings.Repeat("A", 1<<20) + `"}}`,
		`{"model":"m","choices":[{"a":{"b":{"c":{"d":{"e":[1,2,3]}}}}}]}`,
		`{"response":{"response":{"model":"deep"}}}`,
		`{"model":"m","u":` + strings.Repeat("1", 100000) + `}`,
		`{"model":"m","s":"` + strings.Repeat("é", 50000) + `"}`,
		`{"model":"m","esc":"` + strings.Repeat(`\n`, 10000) + `"}`,
	}
	for _, shape := range shapes {
		line := append([]byte("data: "), shape...)
		line = append(line, '\n', '\n')
		require.Equal(t, string(previousNormalize(line, "public-alias", true)),
			string(normalizeSSEDataLineModelWithOptions(line, "public-alias", true)),
			"shape of length %d", len(shape))
	}
}

// previousNormalize is the whole function as it was before the change: the
// previous gate, the previous decode, and the same rewrite on top.
func previousNormalize(line []byte, publicModel string, preserveNestedResponseModel bool) []byte {
	newline := []byte(nil)
	body := line
	switch {
	case bytes.HasSuffix(body, []byte("\r\n")):
		newline = []byte("\r\n")
		body = body[:len(body)-2]
	case bytes.HasSuffix(body, []byte("\n")):
		newline = []byte("\n")
		body = body[:len(body)-1]
	}
	if !bytes.HasPrefix(body, []byte("data:")) {
		return line
	}
	rest := body[len("data:"):]
	payload := bytes.TrimSpace(rest)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}
	if previousGate(payload) {
		return line
	}
	event, err := previousDecode(payload)
	if err != nil || event == nil {
		return line
	}
	rawError, hasError := event["error"]
	eventType := rawMessageString(event["type"])
	if isStreamErrorEvent(eventType, hasError && !isJSONNull(rawError)) {
		return line
	}
	changed := false
	if _, exists := event["model"]; exists {
		if rawMessageString(event["model"]) != publicModel {
			event["model"] = marshalJSONString(publicModel)
			changed = true
		}
	} else if object := rawMessageString(event["object"]); object == "chat.completion.chunk" || object == "text_completion" {
		event["model"] = marshalJSONString(publicModel)
		changed = true
	}
	if rawResponse, ok := event["response"]; ok && !preserveNestedResponseModel {
		if updated, nestedChanged := rewriteNestedResponseModel(rawResponse, publicModel); nestedChanged {
			event["response"] = updated
			changed = true
		}
	}
	if !changed {
		return line
	}
	normalized, err := goccyjson.Marshal(event)
	if err != nil {
		return line
	}
	prefixLen := len(rest) - len(bytes.TrimLeft(rest, " \t"))
	result := make([]byte, 0, len(body)+len(newline))
	result = append(result, []byte("data:")...)
	result = append(result, rest[:prefixLen]...)
	result = append(result, normalized...)
	result = append(result, newline...)
	return result
}

// currentGate is the gate as it is now: the field names followed by their
// closing quote.
func currentGate(payload []byte) bool {
	return !bytes.Contains(payload, []byte(`model"`)) &&
		!bytes.Contains(payload, []byte(`object"`)) &&
		!bytes.Contains(payload, []byte(`response"`)) &&
		bytes.IndexByte(payload, '\\') < 0
}

// TestModelRewriteSkipsResponsesDeltaChunks pins the case that a Responses
// API delta must be skipped: the router generates thousands of them per
// response ("type":"response.output_text.delta"), and the word "response"
// sits in the VALUE of "type" -- so a gate that looked for the bare word
// would decode every one of them for nothing, at ~600 ns and 15
// allocations each, while the rewrite still did not happen.
func TestModelRewriteSkipsResponsesDeltaChunks(t *testing.T) {
	deltas := []string{
		`data: {"type":"response.output_text.delta","item_id":"msg_01Ab","output_index":0,"content_index":0,"delta":"Hello"}`,
		`data: {"type":"response.refusal.delta","item_id":"msg_01Ab","output_index":0,"content_index":0,"delta":"No"}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{}"}`,
	}
	for _, line := range deltas {
		payload := bytes.TrimSpace([]byte(line)[len("data: "):])
		require.True(t, currentGate(payload),
			"a Responses delta must be skipped, not decoded: %s", line)
		require.Equal(t, line+"\n\n",
			string(normalizeSSEDataLineModelWithOptions([]byte(line+"\n\n"), "public-alias", true)),
			"a Responses delta must be left byte-for-byte alone: %s", line)
	}
}

// And the events that DO carry a model must still be rewritten.
func TestModelRewriteRewritesResponsesLifecycleEvents(t *testing.T) {
	for _, line := range []string{
		`data: {"type":"response.created","sequence_number":1,"response":{"id":"r1","model":"upstream-model","status":"in_progress"}}`,
		`data: {"type":"response.completed","response":{"id":"r1","model":"upstream-model","status":"completed"}}`,
	} {
		out := string(normalizeSSEDataLineModelWithOptions([]byte(line+"\n\n"), "public-alias", false))
		require.Contains(t, out, `"model":"public-alias"`,
			"the nested response model must be rewritten: %s", line)
	}
}
