package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// TestConvertVertexUsageMetadata_WithThoughtsTokens verifies that when
// ThoughtsTokenCount > 0, CompletionTokens includes both candidates + thoughts,
// and TotalTokens is consistent (PromptTokens + CompletionTokens).
func TestConvertVertexUsageMetadata_WithThoughtsTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     100,
		CandidatesTokenCount: 50,
		ThoughtsTokenCount:   200,
	}

	usage := convertVertexUsageMetadata(meta)

	// CompletionTokens should be CandidatesTokenCount + ThoughtsTokenCount
	expectedCompletion := 50 + 200
	if usage.CompletionTokens != expectedCompletion {
		t.Fatalf("expected CompletionTokens = %d, got %d", expectedCompletion, usage.CompletionTokens)
	}

	// PromptTokens should be PromptTokenCount (no ToolUsePromptTokenCount here)
	if usage.PromptTokens != 100 {
		t.Fatalf("expected PromptTokens = 100, got %d", usage.PromptTokens)
	}

	// TotalTokens = PromptTokens + CompletionTokens
	expectedTotal := 100 + expectedCompletion
	if usage.TotalTokens != expectedTotal {
		t.Fatalf("expected TotalTokens = %d, got %d", expectedTotal, usage.TotalTokens)
	}

	// CompletionTokensDetails should have ReasoningTokens set
	if usage.CompletionTokensDetails == nil {
		t.Fatalf("expected CompletionTokensDetails to be set")
	}
	if usage.CompletionTokensDetails.ReasoningTokens != 200 {
		t.Fatalf("expected ReasoningTokens = 200, got %d", usage.CompletionTokensDetails.ReasoningTokens)
	}
}

// TestConvertVertexUsageMetadata_NoThoughts verifies that when ThoughtsTokenCount
// is 0, CompletionTokens equals CandidatesTokenCount only, and TotalTokens is
// PromptTokens + CompletionTokens.
func TestConvertVertexUsageMetadata_NoThoughts(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     80,
		CandidatesTokenCount: 40,
		ThoughtsTokenCount:   0,
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.CompletionTokens != 40 {
		t.Fatalf("expected CompletionTokens = 40, got %d", usage.CompletionTokens)
	}
	if usage.PromptTokens != 80 {
		t.Fatalf("expected PromptTokens = 80, got %d", usage.PromptTokens)
	}
	if usage.TotalTokens != 120 {
		t.Fatalf("expected TotalTokens = 120, got %d", usage.TotalTokens)
	}

	// No thinking tokens means CompletionTokensDetails should be nil (unless other details set it)
	if usage.CompletionTokensDetails != nil {
		t.Fatalf("expected CompletionTokensDetails to be nil when no thoughts, got %+v", usage.CompletionTokensDetails)
	}
}

// TestConvertVertexUsageMetadata_WithToolUsePromptTokens verifies that
// ToolUsePromptTokenCount is added to PromptTokens.
func TestConvertVertexUsageMetadata_WithToolUsePromptTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        100,
		ToolUsePromptTokenCount: 25,
		CandidatesTokenCount:    60,
		ThoughtsTokenCount:      0,
	}

	usage := convertVertexUsageMetadata(meta)

	// PromptTokens = PromptTokenCount + ToolUsePromptTokenCount
	if usage.PromptTokens != 125 {
		t.Fatalf("expected PromptTokens = 125, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 60 {
		t.Fatalf("expected CompletionTokens = 60, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 185 {
		t.Fatalf("expected TotalTokens = 185, got %d", usage.TotalTokens)
	}
}

// TestConvertVertexUsageMetadata_CachedContentTokens verifies that
// CachedContentTokenCount is mapped to PromptTokensDetails.CachedTokens.
func TestConvertVertexUsageMetadata_CachedContentTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        200,
		CandidatesTokenCount:    30,
		CachedContentTokenCount: 50,
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil {
		t.Fatalf("expected PromptTokensDetails to be set for cached tokens")
	}
	if usage.PromptTokensDetails.CachedTokens != 50 {
		t.Fatalf("expected CachedTokens = 50, got %d", usage.PromptTokensDetails.CachedTokens)
	}
}

// TestMapFinishReason verifies Vertex finish reason mapping to OpenAI format.
func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"RECITATION", "content_filter"},
		{"TOOL_CALL", "tool_calls"},
		{"UNKNOWN_REASON", "stop"},
		{"", "stop"},
	}

	for _, tt := range tests {
		got := mapFinishReason(tt.input)
		if got != tt.want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// TestVertexToOpenAI_FinishReasonOverrideWithToolCalls verifies that when Vertex
// returns finishReason="STOP" but the response contains FunctionCall parts,
// the finish_reason is overridden to "tool_calls" for OpenAI compatibility.
// This is a Gemini 3+ behavior where Vertex uses STOP even for tool call responses.
func TestVertexToOpenAI_FinishReasonOverrideWithToolCalls(t *testing.T) {
	// Simulate Vertex response with STOP finish reason but containing a FunctionCall
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "multisearch",
								Args: map[string]interface{}{
									"query": "test",
								},
							},
						},
					},
				},
				FinishReason: genai.FinishReasonStop, // STOP, not TOOL_CALL
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 20,
		},
	}

	vertexBytes, err := json.Marshal(vertexResp)
	if err != nil {
		t.Fatalf("marshal vertex response: %v", err)
	}

	resultBytes, err := VertexToOpenAI(vertexBytes, "gemini-3-flash-preview")
	if err != nil {
		t.Fatalf("VertexToOpenAI error: %v", err)
	}

	var openAIResp openai.OpenAIResponse
	if err := json.Unmarshal(resultBytes, &openAIResp); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}

	if len(openAIResp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(openAIResp.Choices))
	}

	choice := openAIResp.Choices[0]

	// finish_reason must be "tool_calls", not "stop"
	if choice.FinishReason != "tool_calls" {
		t.Fatalf("expected finish_reason = %q, got %q", "tool_calls", choice.FinishReason)
	}

	// tool_calls must be present
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool_call, got %d", len(choice.Message.ToolCalls))
	}
	if choice.Message.ToolCalls[0].Function.Name != "multisearch" {
		t.Fatalf("expected tool_call function name = %q, got %q", "multisearch", choice.Message.ToolCalls[0].Function.Name)
	}
}

// TestVertexToOpenAI_FinishReasonNoOverrideWithoutToolCalls verifies that
// finish_reason is NOT overridden when there are no tool_calls in the response.
func TestVertexToOpenAI_FinishReasonNoOverrideWithoutToolCalls(t *testing.T) {
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Hello, world!"},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     50,
			CandidatesTokenCount: 10,
		},
	}

	vertexBytes, err := json.Marshal(vertexResp)
	if err != nil {
		t.Fatalf("marshal vertex response: %v", err)
	}

	resultBytes, err := VertexToOpenAI(vertexBytes, "gemini-3-flash-preview")
	if err != nil {
		t.Fatalf("VertexToOpenAI error: %v", err)
	}

	var openAIResp openai.OpenAIResponse
	if err := json.Unmarshal(resultBytes, &openAIResp); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}

	// finish_reason must remain "stop" — no tool_calls to override
	if openAIResp.Choices[0].FinishReason != "stop" {
		t.Fatalf("expected finish_reason = %q, got %q", "stop", openAIResp.Choices[0].FinishReason)
	}
}
