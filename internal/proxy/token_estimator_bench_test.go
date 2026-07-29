package proxy

import (
	"bytes"
	stdjson "encoding/json"
	"strings"
	"testing"
)

// --- stdlib-only baselines ---
// appendChatCompletionDeltaText/appendResponsesDeltaText in token_estimator.go
// now use goccy, so these copies keep a pure-stdlib version around to isolate
// each change's own contribution (substring filter vs JSON engine swap).

func appendChatCompletionDeltaTextStdlib(b *strings.Builder, payload []byte) {
	var data struct {
		Choices []struct {
			Delta struct {
				Content      interface{} `json:"content"`
				Refusal      string      `json:"refusal"`
				FunctionCall *struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function_call,omitempty"`
				ToolCalls []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := stdjson.Unmarshal(payload, &data); err != nil {
		return
	}
	for _, choice := range data.Choices {
		appendDeltaValueText(b, choice.Delta.Content)
		b.WriteString(choice.Delta.Refusal)
		if choice.Delta.FunctionCall != nil {
			b.WriteString(choice.Delta.FunctionCall.Name)
			b.WriteString(choice.Delta.FunctionCall.Arguments)
		}
		for _, call := range choice.Delta.ToolCalls {
			b.WriteString(call.Function.Name)
			b.WriteString(call.Function.Arguments)
		}
	}
}

func appendResponsesDeltaTextStdlib(b *strings.Builder, payload []byte) {
	var event struct {
		Type  string      `json:"type"`
		Delta interface{} `json:"delta"`
	}
	if err := stdjson.Unmarshal(payload, &event); err != nil || event.Type == "" {
		return
	}
	switch event.Type {
	case "response.output_text.delta",
		"response.refusal.delta",
		"response.reasoning_text.delta",
		"response.reasoning_summary_text.delta",
		"response.function_call_arguments.delta",
		"response.mcp_call_arguments.delta",
		"response.custom_tool_call_input.delta",
		"response.code_interpreter_call_code.delta":
		appendDeltaValueText(b, event.Delta)
	}
}

// oldAppendDeltaText reproduces the original (pre-fix) extractCompletionDeltaText:
// both shape decoders run unconditionally on every payload via stdlib json, so
// one of them always does a fully wasted unmarshal.
func oldAppendDeltaText(chunk []byte) string {
	payloads := extractJSONPayloadsFromStreamChunk(chunk)
	var b strings.Builder
	for _, payload := range payloads {
		appendChatCompletionDeltaTextStdlib(&b, payload)
		appendResponsesDeltaTextStdlib(&b, payload)
	}
	return b.String()
}

// filteredAppendDeltaTextStdlib reproduces the substring-gated fix but stays on
// stdlib json, isolating the filter's own win from the later goccy swap.
func filteredAppendDeltaTextStdlib(chunk []byte) string {
	payloads := extractJSONPayloadsFromStreamChunk(chunk)
	var b strings.Builder
	for _, payload := range payloads {
		if bytes.Contains(payload, []byte(`"choices"`)) {
			appendChatCompletionDeltaTextStdlib(&b, payload)
		}
		if bytes.Contains(payload, []byte(`"type"`)) {
			appendResponsesDeltaTextStdlib(&b, payload)
		}
	}
	return b.String()
}

var (
	chatDeltaPayload = []byte(`data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hello world, this is a streamed token chunk"},"finish_reason":null}]}` + "\n\n")

	responsesDeltaPayload = []byte(`data: {"type":"response.output_text.delta","item_id":"item_abc123","output_index":0,"content_index":0,"delta":"hello world, this is a streamed token chunk"}` + "\n\n")
)

// BenchmarkExtractCompletionDeltaText_Old: original code, both decoders always
// run, stdlib json.
func BenchmarkExtractCompletionDeltaText_Old(b *testing.B) {
	b.Run("chat-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			oldAppendDeltaText(chatDeltaPayload)
		}
	})
	b.Run("responses-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			oldAppendDeltaText(responsesDeltaPayload)
		}
	})
}

// BenchmarkExtractCompletionDeltaText_FilteredStdlib: substring shape filter
// added, still stdlib json — isolates the filter's own contribution.
func BenchmarkExtractCompletionDeltaText_FilteredStdlib(b *testing.B) {
	b.Run("chat-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			filteredAppendDeltaTextStdlib(chatDeltaPayload)
		}
	})
	b.Run("responses-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			filteredAppendDeltaTextStdlib(responsesDeltaPayload)
		}
	})
}

// BenchmarkExtractCompletionDeltaText_FilteredGoccy: current production code —
// substring shape filter + goccy/go-json. Isolates the JSON engine swap's own
// contribution on top of the filter.
func BenchmarkExtractCompletionDeltaText_FilteredGoccy(b *testing.B) {
	b.Run("chat-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extractCompletionDeltaText(chatDeltaPayload)
		}
	})
	b.Run("responses-shaped", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extractCompletionDeltaText(responsesDeltaPayload)
		}
	})
}
