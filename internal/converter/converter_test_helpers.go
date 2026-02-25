package converter

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

func mustJSON(t *testing.T, v interface{}) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func mustUnmarshal[T any](t *testing.T, data []byte) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return v
}

func minimalOpenAIChatRequest() openai.OpenAIRequest {
	return openai.OpenAIRequest{
		Model: "gpt-4",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "hi"},
		},
	}
}
