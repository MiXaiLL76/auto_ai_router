package security

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRedactProviderResponseBodyKeepsStructureAndUsage(t *testing.T) {
	body := `{"id":"chatcmpl-abc","object":"chat.completion","model":"anthropic/claude-opus-4.8",
"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"the model answer"}}],
"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,
"prompt_tokens_details":{"cached_tokens":40,"cache_creation_tokens":15}}}`

	got := RedactProviderResponseBody(body)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("redacted body must stay JSON: %v (%s)", err, got)
	}
	if parsed["model"] != "anthropic/claude-opus-4.8" {
		t.Fatalf("model must survive, got %v", parsed["model"])
	}
	if parsed["id"] != "chatcmpl-abc" {
		t.Fatalf("id must survive, got %v", parsed["id"])
	}
	if _, ok := parsed["usage"]; !ok {
		t.Fatal("usage must survive")
	}
	usage := parsed["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 100 {
		t.Fatalf("prompt_tokens must survive, got %v", usage["prompt_tokens"])
	}
	details := usage["prompt_tokens_details"].(map[string]any)
	if details["cache_creation_tokens"].(float64) != 15 {
		t.Fatalf("cache_creation_tokens must survive, got %v", details["cache_creation_tokens"])
	}
	if details["cached_tokens"].(float64) != 40 {
		t.Fatalf("cached_tokens must survive, got %v", details["cached_tokens"])
	}
	choices := parsed["choices"].([]any)
	choice := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason must survive, got %v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]any)
	if message["content"] != "[REDACTED]" {
		t.Fatalf("the model's own text must be masked, got %v", message["content"])
	}
	if message["role"] != "assistant" {
		t.Fatalf("role must survive, got %v", message["role"])
	}
}

func TestRedactProviderResponseBodyMasksTheProvidersProse(t *testing.T) {
	body := `{"model":"gpt-5.4","error":{"message":"invalid api key sk-secret-value","type":"invalid_request_error"},
"detail":"upstream said no"}`

	got := RedactProviderResponseBody(body)

	if strings.Contains(got, "sk-secret-value") {
		t.Fatalf("a credential in the provider's prose must not survive: %s", got)
	}
	if strings.Contains(got, "upstream said no") {
		t.Fatalf("the provider's free text must be masked: %s", got)
	}
	// The shape survives: the error object is still an object, and the
	// numeric-free fields an operator reads are still there.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("redacted body must stay JSON: %v (%s)", err, got)
	}
	if parsed["model"] != "gpt-5.4" {
		t.Fatalf("model must survive, got %v", parsed["model"])
	}
	errObj, ok := parsed["error"].(map[string]any)
	if !ok {
		t.Fatalf("the error object must keep its shape, got %T", parsed["error"])
	}
	if errObj["message"] != "[REDACTED]" {
		t.Fatalf("the error message must be masked, got %v", errObj["message"])
	}
	if errObj["type"] != "invalid_request_error" {
		t.Fatalf("the error type must survive, got %v", errObj["type"])
	}
}

func TestRedactProviderResponseBodyKeepsNumbers(t *testing.T) {
	body := `{"model":"x","usage":{"prompt_tokens":7,"completion_tokens":3},"retry_after":30,"attempt":2}`

	got := RedactProviderResponseBody(body)

	for _, want := range []string{`"prompt_tokens":7`, `"completion_tokens":3`, `"retry_after":30`, `"attempt":2`} {
		if !strings.Contains(got, want) {
			t.Fatalf("%s must survive: %s", want, got)
		}
	}
}

func TestRedactProviderResponseBodyFailsClosed(t *testing.T) {
	// Not JSON: the whole-body placeholder, never the original text.
	body := "this is not json and it contains sk-secret-value"

	got := RedactProviderResponseBody(body)

	if !strings.HasPrefix(got, "[REDACTED body bytes=") {
		t.Fatalf("an unparseable body must fail closed, got %s", got)
	}
	if strings.Contains(got, "sk-secret-value") {
		t.Fatalf("the original text must not survive: %s", got)
	}
}

func TestRedactProviderResponseBodyEmpty(t *testing.T) {
	if got := RedactProviderResponseBody(""); got != "" {
		t.Fatalf("an empty body must stay empty, got %q", got)
	}
}

func TestRedactProviderResponseBodyKeepsTheStructureOfEveryShape(t *testing.T) {
	// The Anthropic-native shape: the top level IS the message.
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4.8",
"content":[{"type":"text","text":"the answer"}],
"stop_reason":"end_turn",
"usage":{"input_tokens":10,"output_tokens":3,"cache_creation_input_tokens":7,"cache_read_input_tokens":2}}`

	got := RedactProviderResponseBody(body)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("redacted body must stay JSON: %v (%s)", err, got)
	}
	if parsed["model"] != "claude-opus-4.8" {
		t.Fatalf("model must survive, got %v", parsed["model"])
	}
	if parsed["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason must survive, got %v", parsed["stop_reason"])
	}
	blocks := parsed["content"].([]any)
	block := blocks[0].(map[string]any)
	if block["type"] != "text" {
		t.Fatalf("the part type must survive, got %v", block["type"])
	}
	if block["text"] != "[REDACTED]" {
		t.Fatalf("the text must be masked, got %v", block["text"])
	}
	usage := parsed["usage"].(map[string]any)
	if usage["cache_creation_input_tokens"].(float64) != 7 {
		t.Fatalf("cache_creation_input_tokens must survive, got %v", usage["cache_creation_input_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 2 {
		t.Fatalf("cache_read_input_tokens must survive, got %v", usage["cache_read_input_tokens"])
	}
}

func TestRedactProviderResponseBodyMasksToolArguments(t *testing.T) {
	body := `{"model":"x","choices":[{"message":{"role":"assistant","tool_calls":[
{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"query\":\"secret text\"}"}}]}}]}`

	got := RedactProviderResponseBody(body)

	if strings.Contains(got, "secret text") {
		t.Fatalf("tool arguments must be masked: %s", got)
	}
}
