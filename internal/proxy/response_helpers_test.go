package proxy

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestInjectStreamOptions_AddsIncludeUsage(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
}

func TestInjectStreamOptions_UpdatesExisting(t *testing.T) {
	body := []byte(`{"stream_options":{"include_usage":false,"foo":1}}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
	if streamOptions["foo"] != float64(1) {
		t.Fatalf("expected foo to be preserved, got %v", streamOptions["foo"])
	}
}

func TestInjectStreamOptions_ReplacesNonMap(t *testing.T) {
	body := []byte(`{"stream_options":"bad"}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
}

func TestInjectStreamOptions_InvalidJSON(t *testing.T) {
	body := []byte(`{"stream_options":`)
	modified := injectStreamOptions(body)
	if !bytes.Equal(modified, body) {
		t.Fatalf("expected invalid json to be returned as-is")
	}
}
