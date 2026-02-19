package vertex

import (
	"testing"

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
