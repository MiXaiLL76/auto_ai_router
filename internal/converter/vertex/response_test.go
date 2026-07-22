package vertex

import (
	"encoding/json"
	"strings"
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
		return
	}
	if usage.CompletionTokensDetails.ReasoningTokens != 200 {
		t.Fatalf("expected ReasoningTokens = 200, got %d", usage.CompletionTokensDetails.ReasoningTokens)
	}
}

func TestVertexToOpenAI_ImageUsesChatImageURLShape(t *testing.T) {
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "created"},
						{InlineData: &genai.Blob{Data: []byte("img"), MIMEType: "image/png"}},
					},
				},
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     12,
			CandidatesTokenCount: 1183,
			ThoughtsTokenCount:   12,
		},
	}

	vertexBytes, err := json.Marshal(vertexResp)
	if err != nil {
		t.Fatalf("marshal vertex response: %v", err)
	}

	resultBytes, err := VertexToOpenAI(vertexBytes, "gemini-3-pro-image-preview")
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
	images := openAIResp.Choices[0].Message.Images
	if len(images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(images))
	}
	image := images[0]
	if image.Type != "image_url" {
		t.Fatalf("expected image type image_url, got %q", image.Type)
	}
	if image.Index == nil || *image.Index != 0 {
		t.Fatalf("expected image index 0, got %v", image.Index)
	}
	if image.ImageURL == nil || image.ImageURL.URL != "data:image/png;base64,aW1n" {
		t.Fatalf("unexpected image_url: %+v", image.ImageURL)
	}
	if image.B64JSON != "" {
		t.Fatalf("chat images must not use b64_json, got %q", image.B64JSON)
	}
	if strings.Contains(string(resultBytes), `"b64_json"`) {
		t.Fatalf("chat response must not contain b64_json: %s", string(resultBytes))
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
		return
	}
	if usage.PromptTokensDetails.CachedTokens != 50 {
		t.Fatalf("expected CachedTokens = 50, got %d", usage.PromptTokensDetails.CachedTokens)
	}
}

// TestConvertVertexUsageMetadata_AudioInputTokens verifies that audio tokens
// from PromptTokensDetails are mapped to PromptTokensDetails.AudioTokens.
func TestConvertVertexUsageMetadata_AudioInputTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     100,
		CandidatesTokenCount: 30,
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: 70},
			{Modality: genai.MediaModalityAudio, TokenCount: 30},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil {
		t.Fatal("expected PromptTokensDetails to be non-nil")
		return
	}
	if usage.PromptTokensDetails.AudioTokens != 30 {
		t.Fatalf("expected AudioTokens = 30, got %d", usage.PromptTokensDetails.AudioTokens)
	}
	// Text modality has no dedicated OpenAI field — only audio is extracted
	if usage.PromptTokens != 100 {
		t.Fatalf("expected PromptTokens = 100, got %d", usage.PromptTokens)
	}
}

// TestConvertVertexUsageMetadata_AudioOutputTokens verifies that audio tokens
// from CandidatesTokensDetails are mapped to CompletionTokensDetails.AudioTokens.
func TestConvertVertexUsageMetadata_AudioOutputTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     50,
		CandidatesTokenCount: 80,
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: 30},
			{Modality: genai.MediaModalityAudio, TokenCount: 50},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.CompletionTokensDetails == nil {
		t.Fatal("expected CompletionTokensDetails to be non-nil")
		return
	}
	if usage.CompletionTokensDetails.AudioTokens != 50 {
		t.Fatalf("expected AudioTokens = 50, got %d", usage.CompletionTokensDetails.AudioTokens)
	}
	if usage.CompletionTokens != 80 {
		t.Fatalf("expected CompletionTokens = 80, got %d", usage.CompletionTokens)
	}
}

// TestConvertVertexUsageMetadata_AudioInputAndOutput verifies both input and output
// audio tokens are correctly mapped when present simultaneously.
func TestConvertVertexUsageMetadata_AudioInputAndOutput(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     100,
		CandidatesTokenCount: 60,
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 40},
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 20},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.AudioTokens != 40 {
		t.Fatalf("expected input AudioTokens = 40, got %v", usage.PromptTokensDetails)
	}
	if usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.AudioTokens != 20 {
		t.Fatalf("expected output AudioTokens = 20, got %v", usage.CompletionTokensDetails)
	}
}

// TestConvertVertexUsageMetadata_ToolUseAudioTokens verifies that audio tokens
// from ToolUsePromptTokensDetails are added to PromptTokensDetails.AudioTokens.
func TestConvertVertexUsageMetadata_ToolUseAudioTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        100,
		ToolUsePromptTokenCount: 50,
		CandidatesTokenCount:    30,
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 20},
		},
		ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 10},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokens != 150 {
		t.Fatalf("expected PromptTokens = 150, got %d", usage.PromptTokens)
	}
	if usage.PromptTokensDetails == nil {
		t.Fatal("expected PromptTokensDetails to be non-nil")
		return
	}
	// Both prompt and tool-use audio tokens accumulate
	if usage.PromptTokensDetails.AudioTokens != 30 {
		t.Fatalf("expected AudioTokens = 30 (20+10), got %d", usage.PromptTokensDetails.AudioTokens)
	}
}

// TestConvertVertexUsageMetadata_CachedAudioSubtraction verifies the billing-correct
// behavior: cached audio tokens are subtracted from audio_tokens so that
// CalculateTokenCosts can bill them at the cached rate (not the audio rate).
//
// NOTE: This means audio_tokens in the response reflects NON-CACHED audio only.
// From OpenAI spec, audio_tokens should be total audio — but keeping it non-cached
// is necessary for correct billing since CalculateTokenCosts formula is:
//
//	regularInputTokens = PromptTokens - AudioInputTokens - CachedInputTokens
//
// If AudioInputTokens included cached audio, cached audio would be double-subtracted.
func TestConvertVertexUsageMetadata_CachedAudioSubtraction(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        200,
		CandidatesTokenCount:    30,
		CachedContentTokenCount: 80, // 40 text cached + 40 audio cached
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 100}, // 60 non-cached + 40 cached
		},
		CacheTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 40},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil {
		t.Fatal("expected PromptTokensDetails to be non-nil")
		return
	}
	// audio_tokens = total(100) - cached(40) = 60 non-cached audio
	// This enables correct billing: 60 at audio rate + 80 at cached rate
	if usage.PromptTokensDetails.AudioTokens != 60 {
		t.Fatalf("expected AudioTokens = 60 (non-cached only), got %d", usage.PromptTokensDetails.AudioTokens)
	}
	if usage.PromptTokensDetails.CachedTokens != 80 {
		t.Fatalf("expected CachedTokens = 80, got %d", usage.PromptTokensDetails.CachedTokens)
	}

	// Verify billing math: regularInputTokens = 200 - 60 - 80 = 60 (text tokens)
	// This is the invariant that CalculateTokenCosts depends on.
	regularInputTokens := usage.PromptTokens - usage.PromptTokensDetails.AudioTokens - usage.PromptTokensDetails.CachedTokens
	if regularInputTokens != 60 {
		t.Fatalf("billing invariant broken: regularInputTokens = %d, want 60", regularInputTokens)
	}
}

// TestConvertVertexUsageMetadata_AllAudioCached verifies that when all audio is cached,
// audio_tokens becomes 0. This is billing-correct (no audio billed at audio rate),
// but users may see audio_tokens=0 even though audio was sent (all from cache).
func TestConvertVertexUsageMetadata_AllAudioCached(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        100,
		CandidatesTokenCount:    20,
		CachedContentTokenCount: 60, // 60 cached audio
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 60},
		},
		CacheTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityAudio, TokenCount: 60},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil {
		t.Fatal("expected PromptTokensDetails to be non-nil (CachedTokens must be set)")
		return
	}
	// All audio is cached → audio_tokens = 0 (billed at cached rate only)
	if usage.PromptTokensDetails.AudioTokens != 0 {
		t.Fatalf("expected AudioTokens = 0 (all cached), got %d", usage.PromptTokensDetails.AudioTokens)
	}
	if usage.PromptTokensDetails.CachedTokens != 60 {
		t.Fatalf("expected CachedTokens = 60, got %d", usage.PromptTokensDetails.CachedTokens)
	}
}

// TestConvertVertexUsageMetadata_NilModalityEntries verifies that nil entries
// in modality detail slices are safely skipped without panicking.
func TestConvertVertexUsageMetadata_NilModalityEntries(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     100,
		CandidatesTokenCount: 50,
		PromptTokensDetails: []*genai.ModalityTokenCount{
			nil,
			{Modality: genai.MediaModalityAudio, TokenCount: 25},
			nil,
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			nil,
			{Modality: genai.MediaModalityAudio, TokenCount: 15},
		},
	}

	// Must not panic
	usage := convertVertexUsageMetadata(meta)

	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.AudioTokens != 25 {
		t.Fatalf("expected input AudioTokens = 25, got %v", usage.PromptTokensDetails)
	}
	if usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.AudioTokens != 15 {
		t.Fatalf("expected output AudioTokens = 15, got %v", usage.CompletionTokensDetails)
	}
}

// TestConvertVertexUsageMetadata_ImageVideoTokens verifies that generated media tokens
// are exposed through LiteLLM's completion_tokens_details.image_tokens extension.
func TestConvertVertexUsageMetadata_ImageVideoTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     200,
		CandidatesTokenCount: 50,
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityImage, TokenCount: 100},
			{Modality: genai.MediaModalityVideo, TokenCount: 50},
			{Modality: genai.MediaModalityText, TokenCount: 50},
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityImage, TokenCount: 30},
			{Modality: genai.MediaModalityVideo, TokenCount: 20},
		},
	}

	usage := convertVertexUsageMetadata(meta)

	// Input image/video details remain part of PromptTokens; this converter only
	// exposes the generated media breakdown needed for output billing.
	if usage.PromptTokensDetails != nil && usage.PromptTokensDetails.AudioTokens != 0 {
		t.Fatalf("expected no audio tokens from image/video, got %d", usage.PromptTokensDetails.AudioTokens)
	}
	if usage.CompletionTokensDetails == nil {
		t.Fatal("expected completion token details for generated media")
	}
	if usage.CompletionTokensDetails.ImageTokens != 50 {
		t.Fatalf("expected 50 image/video output tokens, got %d", usage.CompletionTokensDetails.ImageTokens)
	}
	// Total tokens still correct
	if usage.PromptTokens != 200 || usage.CompletionTokens != 50 {
		t.Fatalf("token totals wrong: prompt=%d completion=%d", usage.PromptTokens, usage.CompletionTokens)
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
