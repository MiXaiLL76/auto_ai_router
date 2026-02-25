package converter

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"google.golang.org/genai"
)

func TestProviderConverter_RequestFrom_Passthrough(t *testing.T) {
	c := New(config.ProviderTypeOpenAI, RequestMode{})
	body := []byte(`{"test":true}`)
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("expected passthrough body, got %s", string(got))
	}
}

func TestProviderConverter_RequestFrom_Anthropic(t *testing.T) {
	body := mustJSON(t, minimalOpenAIChatRequest())

	c := New(config.ProviderTypeAnthropic, RequestMode{ModelID: "claude-test"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}

	req := mustUnmarshal[anthropic.AnthropicRequest](t, got)
	if req.Model != "claude-test" {
		t.Fatalf("expected model claude-test, got %q", req.Model)
	}
	if req.MaxTokens != 4096 {
		t.Fatalf("expected default max_tokens 4096, got %d", req.MaxTokens)
	}
}

func TestProviderConverter_RequestFrom_AnthropicImageNotSupported(t *testing.T) {
	c := New(config.ProviderTypeAnthropic, RequestMode{IsImageGeneration: true})
	_, err := c.RequestFrom([]byte(`{"model":"gpt-4"}`))
	if err == nil {
		t.Fatalf("expected error for image generation")
	}
	if !strings.Contains(err.Error(), "does not support image generation") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProviderConverter_RequestFrom_VertexImageGeneration_Imagen(t *testing.T) {
	n := 2
	imgReq := openai.OpenAIImageRequest{
		Model:   "imagen-3",
		Prompt:  "make image",
		N:       &n,
		Size:    "1792x1024",
		Quality: "hd",
	}
	body := mustJSON(t, imgReq)

	c := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: true, ModelID: "imagen-3"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}

	req := mustUnmarshal[vertex.VertexImageRequest](t, got)
	if len(req.Instances) != 1 || req.Instances[0].Prompt != "make image" {
		t.Fatalf("unexpected instances: %+v", req.Instances)
	}
	if req.Parameters.SampleCount != 2 {
		t.Fatalf("expected sampleCount 2, got %d", req.Parameters.SampleCount)
	}
	if req.Parameters.AspectRatio != "16:9" {
		t.Fatalf("expected aspectRatio 16:9, got %q", req.Parameters.AspectRatio)
	}
	if req.Parameters.SafetyFilterLevel != "block_few" {
		t.Fatalf("expected safety block_few, got %q", req.Parameters.SafetyFilterLevel)
	}
}

func TestProviderConverter_RequestFrom_VertexChat(t *testing.T) {
	body := mustJSON(t, minimalOpenAIChatRequest())
	c := New(config.ProviderTypeVertexAI, RequestMode{ModelID: "gemini-1.5-flash"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}

	req := mustUnmarshal[vertex.VertexRequest](t, got)
	if len(req.Contents) != 1 {
		t.Fatalf("expected 1 content item, got %d", len(req.Contents))
	}
	if req.Contents[0].Role != "user" {
		t.Fatalf("expected role user, got %q", req.Contents[0].Role)
	}
}

func TestProviderConverter_ResponseTo_Passthrough(t *testing.T) {
	c := New(config.ProviderTypeProxy, RequestMode{})
	body := []byte(`{"ok":1}`)
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("expected passthrough body, got %s", string(got))
	}
}

func TestProviderConverter_ResponseTo_VertexImage_Imagen(t *testing.T) {
	vertexResp := vertex.VertexImageResponse{
		Predictions: []vertex.VertexImagePrediction{{BytesBase64Encoded: "aGVsbG8="}},
	}
	body := mustJSON(t, vertexResp)

	c := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: true, ModelID: "imagen-3"})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIImageResponse](t, got)
	if len(resp.Data) != 1 || resp.Data[0].B64JSON != "aGVsbG8=" {
		t.Fatalf("unexpected image response: %+v", resp.Data)
	}
}

func TestProviderConverter_ResponseTo_VertexImage_Gemini(t *testing.T) {
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{Content: &genai.Content{Parts: []*genai.Part{
				{InlineData: &genai.Blob{Data: []byte("img"), MIMEType: "image/png"}},
			}}},
		},
	}
	body := mustJSON(t, vertexResp)

	c := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: true, ModelID: "gemini-2.0-flash"})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIImageResponse](t, got)
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Data))
	}
	if resp.Data[0].B64JSON != base64.StdEncoding.EncodeToString([]byte("img")) {
		t.Fatalf("unexpected image b64: %q", resp.Data[0].B64JSON)
	}
}

func TestProviderConverter_ResponseTo_VertexChat(t *testing.T) {
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{Content: &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}},
		},
	}
	body := mustJSON(t, vertexResp)

	c := New(config.ProviderTypeVertexAI, RequestMode{ModelID: "gemini-1.5-flash"})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIResponse](t, got)
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "hello" {
		t.Fatalf("unexpected response: %+v", resp.Choices)
	}
}

func TestProviderConverter_ResponseTo_Anthropic(t *testing.T) {
	anthropicResp := anthropic.AnthropicResponse{
		ID:         "msg_1",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude",
		StopReason: "end_turn",
		Usage: anthropic.AnthropicUsage{
			InputTokens:          5,
			OutputTokens:         7,
			CacheReadInputTokens: 2,
		},
		Content: []anthropic.ContentBlock{
			{Type: "text", Text: "hello"},
			{Type: "thinking", Thinking: "hmm"},
			{Type: "tool_use", ID: "tool1", Name: "calc", Input: map[string]interface{}{"x": 1}},
		},
	}
	body := mustJSON(t, anthropicResp)

	c := New(config.ProviderTypeAnthropic, RequestMode{ModelID: "claude"})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIResponse](t, got)
	if resp.ID != "msg_1" {
		t.Fatalf("expected id msg_1, got %q", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	msg := resp.Choices[0].Message
	if msg.Content != "hello" || msg.ReasoningContent != "hmm" || len(msg.ToolCalls) != 1 {
		t.Fatalf("unexpected message: %+v", msg)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 5 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 12 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}
	if resp.Usage.PromptTokensDetails == nil || resp.Usage.PromptTokensDetails.CachedTokens != 2 {
		t.Fatalf("unexpected prompt token details: %+v", resp.Usage.PromptTokensDetails)
	}
}

func TestProviderConverter_StreamTo(t *testing.T) {
	{
		c := New(config.ProviderTypeOpenAI, RequestMode{})
		input := strings.NewReader("abc")
		var out bytes.Buffer
		if err := c.StreamTo(input, &out); err != nil {
			t.Fatalf("StreamTo error: %v", err)
		}
		if out.String() != "abc" {
			t.Fatalf("expected passthrough output, got %q", out.String())
		}
	}

	{
		c := New(config.ProviderTypeVertexAI, RequestMode{ModelID: "gemini-1.5-flash"})
		input := strings.NewReader("data: [DONE]\n\n")
		var out bytes.Buffer
		if err := c.StreamTo(input, &out); err != nil {
			t.Fatalf("StreamTo error: %v", err)
		}
		if out.String() != "data: [DONE]\n\n" {
			t.Fatalf("unexpected output: %q", out.String())
		}
	}

	{
		c := New(config.ProviderTypeAnthropic, RequestMode{ModelID: "claude"})
		input := strings.NewReader("data: {\"type\":\"message_stop\"}\n")
		var out bytes.Buffer
		if err := c.StreamTo(input, &out); err != nil {
			t.Fatalf("StreamTo error: %v", err)
		}
		if out.String() != "data: [DONE]\n\n" {
			t.Fatalf("unexpected output: %q", out.String())
		}
	}
}

func TestProviderConverter_BuildURL(t *testing.T) {
	cred := &config.CredentialConfig{
		ProjectID: "proj",
		Location:  "us-central1",
		BaseURL:   "https://example.com/",
	}

	cVertexImage := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: true, ModelID: "imagen-3"})
	got := cVertexImage.BuildURL(cred)
	want := vertex.BuildVertexImageURL(cred, "imagen-3")
	if got != want {
		t.Fatalf("vertex image url mismatch: got %q want %q", got, want)
	}

	cVertexChat := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: false, ModelID: "gemini-1.5"})
	got = cVertexChat.BuildURL(cred)
	want = vertex.BuildVertexURL(cred, "gemini-1.5", false)
	if got != want {
		t.Fatalf("vertex url mismatch: got %q want %q", got, want)
	}

	cAnthropic := New(config.ProviderTypeAnthropic, RequestMode{})
	got = cAnthropic.BuildURL(cred)
	if got != "https://example.com/v1/messages" {
		t.Fatalf("unexpected anthropic url: %q", got)
	}

	cOpenAI := New(config.ProviderTypeOpenAI, RequestMode{})
	if got = cOpenAI.BuildURL(cred); got != "" {
		t.Fatalf("expected empty url for openai, got %q", got)
	}
}

func TestProviderConverter_IsPassthrough(t *testing.T) {
	if !New(config.ProviderTypeOpenAI, RequestMode{}).IsPassthrough() {
		t.Fatalf("openai should be passthrough")
	}
	if !New(config.ProviderTypeProxy, RequestMode{}).IsPassthrough() {
		t.Fatalf("proxy should be passthrough")
	}
	if New(config.ProviderTypeVertexAI, RequestMode{}).IsPassthrough() {
		t.Fatalf("vertex should not be passthrough")
	}
}

func TestProviderConverter_UsageFromResponse(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":3,"completion_tokens":4}}`)
	c := New(config.ProviderTypeOpenAI, RequestMode{})
	usage := c.UsageFromResponse(body)
	if usage == nil || usage.PromptTokens != 3 || usage.CompletionTokens != 4 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestAnthropicUsageToTokenUsage(t *testing.T) {
	usage := AnthropicUsageToTokenUsage(1, 2, 3)
	if usage.PromptTokens != 1 || usage.CompletionTokens != 2 || usage.CachedInputTokens != 3 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestExtractTokenUsage(t *testing.T) {
	if got := ExtractTokenUsage(nil); got != nil {
		t.Fatalf("expected nil for empty body")
	}
	if got := ExtractTokenUsage([]byte("not-json")); got != nil {
		t.Fatalf("expected nil for invalid json")
	}

	chatBody := []byte(`{"usage":{"prompt_tokens":5,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":2,"audio_tokens":1},"completion_tokens_details":{"accepted_prediction_tokens":3,"rejected_prediction_tokens":1,"audio_tokens":4,"reasoning_tokens":6}}}`)
	usage := ExtractTokenUsage(chatBody)
	if usage == nil {
		t.Fatalf("expected usage for chat format")
	}
	if usage.PromptTokens != 5 || usage.CompletionTokens != 7 {
		t.Fatalf("unexpected chat token counts: %+v", usage)
	}
	if usage.CachedInputTokens != 2 || usage.AudioInputTokens != 1 || usage.AudioOutputTokens != 4 || usage.ReasoningTokens != 6 {
		t.Fatalf("unexpected details: %+v", usage)
	}
	if usage.AcceptedPredictionTokens != 3 || usage.RejectedPredictionTokens != 1 {
		t.Fatalf("unexpected prediction tokens: %+v", usage)
	}

	imageBody := []byte(`{"usage":{"input_tokens":9,"output_tokens":10,"input_tokens_details":{"image_tokens":8}}}`)
	usage = ExtractTokenUsage(imageBody)
	if usage == nil {
		t.Fatalf("expected usage for image format")
	}
	if usage.PromptTokens != 9 || usage.CompletionTokens != 10 || usage.ImageTokens != 8 {
		t.Fatalf("unexpected image token counts: %+v", usage)
	}

	zeroBody := []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0}}`)
	if got := ExtractTokenUsage(zeroBody); got != nil {
		t.Fatalf("expected nil for zero usage")
	}
}

func TestExtractTokenUsage_ImageFallback(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0,"input_tokens":2,"output_tokens":3}}`)
	usage := ExtractTokenUsage(body)
	if usage == nil || usage.PromptTokens != 2 || usage.CompletionTokens != 3 {
		t.Fatalf("unexpected fallback usage: %+v", usage)
	}
}

func TestProviderConverter_ResponseTo_VertexImage_Gemini_JSONRoundTrip(t *testing.T) {
	vertexResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{Content: &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{Data: []byte("x"), MIMEType: "image/png"}}}}},
		},
	}
	b, err := json.Marshal(vertexResp)
	if err != nil {
		t.Fatalf("marshal genai: %v", err)
	}

	c := New(config.ProviderTypeVertexAI, RequestMode{IsImageGeneration: true, ModelID: "gemini-2.0"})
	got, err := c.ResponseTo(b)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIImageResponse](t, got)
	if len(resp.Data) != 1 || resp.Data[0].B64JSON == "" {
		t.Fatalf("unexpected gemini image response: %+v", resp.Data)
	}
}
