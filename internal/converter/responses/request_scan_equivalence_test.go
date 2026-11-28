package responses

import (
	"encoding/json"
	"reflect"
	"testing"
)

// The functions below are the implementations IsResponsesAPI and
// ExtractResponsesMetadata had before they moved to jsonscan (one full decode
// of the body each). They stay here as the reference the new ones are held to.

func legacyIsResponsesAPI(body []byte) bool {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return false
	}
	_, hasInput := raw["input"]
	_, hasMessages := raw["messages"]
	return hasInput && !hasMessages
}

func legacyExtractResponsesMetadata(body []byte) ResponsesMetadata {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return ResponsesMetadata{}
	}
	var meta ResponsesMetadata
	if v, ok := raw["store"].(bool); ok {
		meta.Store = v
	}
	if v, ok := raw["previous_response_id"].(string); ok {
		meta.PreviousResponseID = v
	}
	if v, ok := raw["metadata"].(map[string]interface{}); ok {
		meta.Metadata = make(map[string]string, len(v))
		for k, val := range v {
			if s, ok := val.(string); ok {
				meta.Metadata[k] = s
			}
		}
	}
	if v, ok := raw["ttl"].(float64); ok {
		meta.TTL = int(v)
	}
	if text, ok := raw["text"].(map[string]interface{}); ok {
		if format, ok := text["format"].(map[string]interface{}); ok && isWellFormedTextFormat(format) {
			meta.Text = map[string]interface{}{"format": format}
		}
	}
	return meta
}

var scanEquivalenceBodies = []string{
	``, `null`, `[]`, `{`, `{"input":[]`, `{"input":[]} trailing`,
	`{}`,
	`{"model":"m","input":"hi"}`,
	`{"model":"m","input":[{"role":"user","content":"line\nbreak \"quoted\""}]}`,
	`{"model":"m","messages":[{"role":"user","content":"mentions \"input\" in text"}]}`,
	`{"model":"m","input":[],"messages":[]}`,
	`{"model":"m","input":null}`,
	`{"model":"m","tools":[{"name":"t","input_schema":{"properties":{"input":{"type":"string"}}}}],"messages":[]}`,
	`{"input":[1]}`,
	`{"input":[1],"messages":[]}`,
	`{"input":[1],"input":[2],"store":true,"store":false}`,
	`{"input":[],"store":true,"previous_response_id":"resp_1","metadata":{"a":"b","n":1,"z":null},"ttl":3600}`,
	`{"input":[],"store":"yes","previous_response_id":7,"metadata":[1],"ttl":"60"}`,
	`{"input":[],"ttl":12.9,"metadata":{}}`,
	`{"input":[],"metadata":{"a":"1"},"metadata":{"b":"2"}}`,
	`{"input":[],"text":{"format":{"type":"text"},"verbosity":"low"}}`,
	`{"input":[],"text":{"format":{"type":"json_object"}}}`,
	`{"input":[],"text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object"},"strict":true}}}`,
	`{"input":[],"text":{"format":{"type":"json_schema","schema":{}}}}`,
	`{"input":[],"text":{"format":{"type":"bogus"}}}`,
	`{"input":[],"text":"plain"}`,
	`{"input":[],"store":true,"text":null,"metadata":null,"ttl":null}`,
	"{\"input\":[\"\xff\xfe\"],\"store\":true}",
}

func TestScanBasedRequestHelpersMatchFullDecode(t *testing.T) {
	for _, body := range scanEquivalenceBodies {
		b := []byte(body)
		if got, want := IsResponsesAPI(b), legacyIsResponsesAPI(b); got != want {
			t.Errorf("IsResponsesAPI(%q) = %v, full decode says %v", body, got, want)
		}
		if got, want := ExtractResponsesMetadata(b), legacyExtractResponsesMetadata(b); !reflect.DeepEqual(got, want) {
			t.Errorf("ExtractResponsesMetadata(%q) = %#v, full decode says %#v", body, got, want)
		}
	}
}

// The one input class where the scan and a full decode disagree: a number
// literal outside the float64 range somewhere in the body. A full decode into
// interface{} fails on it and the old helpers therefore answered "not a
// Responses request" / "no metadata"; the scan does not convert numbers it
// was not asked about. Pinned here so the difference stays deliberate.
func TestScanBasedRequestHelpersOutOfRangeNumber(t *testing.T) {
	unrelated := []byte(`{"input":[],"store":true,"temperature":1e999}`)
	if legacyIsResponsesAPI(unrelated) {
		t.Fatal("reference changed: a full decode used to fail on 1e999")
	}
	if !IsResponsesAPI(unrelated) {
		t.Error("IsResponsesAPI must not depend on a number it does not read")
	}
	if !ExtractResponsesMetadata(unrelated).Store {
		t.Error("ExtractResponsesMetadata must not depend on a number it does not read")
	}
	// In a field it does read, the value still fails to decode: no metadata.
	own := []byte(`{"input":[],"store":true,"ttl":1e999}`)
	if got := ExtractResponsesMetadata(own); !reflect.DeepEqual(got, ResponsesMetadata{}) {
		t.Errorf("ExtractResponsesMetadata with ttl=1e999 = %#v, want zero value", got)
	}
}

func FuzzScanBasedRequestHelpers(f *testing.F) {
	for _, body := range scanEquivalenceBodies {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var probe map[string]interface{}
		if json.Unmarshal(body, &probe) != nil && json.Valid(body) {
			// Valid JSON a full decode rejects: non-object documents (both
			// sides agree, checked below) or an out-of-range number (the
			// documented difference).
			var asRaw map[string]json.RawMessage
			if json.Unmarshal(body, &asRaw) == nil && asRaw != nil {
				t.Skip("out-of-range number")
			}
		}
		if got, want := IsResponsesAPI(body), legacyIsResponsesAPI(body); got != want {
			t.Fatalf("IsResponsesAPI(%q) = %v, full decode says %v", body, got, want)
		}
		if got, want := ExtractResponsesMetadata(body), legacyExtractResponsesMetadata(body); !reflect.DeepEqual(got, want) {
			t.Fatalf("ExtractResponsesMetadata(%q) = %#v, full decode says %#v", body, got, want)
		}
	})
}
