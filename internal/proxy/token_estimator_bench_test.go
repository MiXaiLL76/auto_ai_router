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
	payloads := splitSSEPayloads(chunk, nil)
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
	payloads := splitSSEPayloads(chunk, nil)
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

// --- Fix A: local tiktoken prompt estimator, eager vs lazy-armed ---

// chatBody5632Tokens mirrors the real production request shape used by this
// team's load generator (ai-gateway-bench's common/payloads.py: {"model", "stream",
// "messages":[{"role":"user","content": "token " * N}]}), sized to roughly the real
// monthly chat average of ~5632 prompt tokens documented in pprof_plan_test.md.
var chatBody5632Tokens = buildChatBody5632Tokens()

func buildChatBody5632Tokens() []byte {
	body := map[string]interface{}{
		"model":  "gpt-4o-mini",
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": strings.Repeat("token ", 5632)},
		},
	}
	data, err := stdjson.Marshal(body)
	if err != nil {
		panic(err)
	}
	return data
}

// BenchmarkEstimatePromptTokensForModel measures the full cost fix A removes from
// the unconditional streaming entry path: a stdlib json.Unmarshal of the whole
// request body into map[string]interface{} (boxing every element of messages) plus
// a full tiktoken BPE pass over the prompt text. Before the fix, this ran on every
// single streaming request regardless of whether the result was ever used.
func BenchmarkEstimatePromptTokensForModel(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		estimatePromptTokensForModel(chatBody5632Tokens, "gpt-4o-mini")
	}
}

// BenchmarkSetPromptTokensEstimate_ArmedNeverCalled measures the "after" cost on
// the providerUsage=true path (the common case for OpenAI/Vertex/Bedrock/Anthropic):
// arming the lazy closure over the same 5632-token body without ever invoking it.
// Should be orders of magnitude cheaper than BenchmarkEstimatePromptTokensForModel
// above — no json decode, no tokenization, just a closure allocation capturing the
// already-live body slice header.
func BenchmarkSetPromptTokensEstimate_ArmedNeverCalled(b *testing.B) {
	prx := NewTestProxyBuilder().Build()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logCtx := &RequestLogContext{}
		prx.setPromptTokensEstimate(logCtx, chatBody5632Tokens, "gpt-4o-mini")
	}
}
