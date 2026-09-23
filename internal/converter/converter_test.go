package converter

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
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

// TestProviderConverter_RequestFrom_StripsCacheSaltForOpenAICompatible covers
// the "default" (OpenAI-compatible) branch of RequestFrom: cache_salt is a
// LiteLLM/router-level convention, not part of OpenAI's own Chat Completions
// API -- genuine api.openai.com rejects it outright with a 400 ("Unknown
// parameter: 'cache_salt'.", confirmed directly against api.openai.com), same
// as every other server sharing this default bucket, so there's no "real
// OpenAI" exception to carve out here.
func TestProviderConverter_RequestFrom_StripsCacheSaltForOpenAICompatible(t *testing.T) {
	body := []byte(`{"model":"gpt-5-mini","cache_salt":"partition-1","messages":[]}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		ModelID: "gpt-5-mini",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["cache_salt"]; present {
		t.Fatalf("expected cache_salt to be stripped, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_PreservesCacheSaltForVLLM covers the vLLM
// exception to shouldStripCacheSalt: self-hosted vLLM is confirmed to support
// cache_salt for its own prefix-cache partitioning, unlike other non-OpenAI
// servers sharing the same default RequestFrom bucket.
func TestProviderConverter_RequestFrom_PreservesCacheSaltForVLLM(t *testing.T) {
	body := []byte(`{"model":"qwen3-32b","cache_salt":"partition-1","messages":[]}`)

	c := New(config.ProviderTypeVLLM, RequestMode{
		ModelID: "qwen3-32b",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if v, present := m["cache_salt"]; !present || v != "partition-1" {
		t.Fatalf("expected cache_salt to be preserved for vLLM, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsCacheSaltForAnthropicMessagesPassthrough
// covers the MessagesPassthrough branch: a client can still send an
// OpenAI-only field like cache_salt on a /v1/messages request that's
// forwarded natively to Anthropic (or CometAPI/ProMan in Anthropic-protocol
// mode) without going through OpenAIToAnthropic at all.
func TestProviderConverter_RequestFrom_StripsCacheSaltForAnthropicMessagesPassthrough(t *testing.T) {
	body := []byte(`{"model":"claude-test","cache_salt":"partition-1","messages":[]}`)

	c := New(config.ProviderTypeAnthropic, RequestMode{
		ModelID:             "claude-test",
		MessagesPassthrough: true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["cache_salt"]; present {
		t.Fatalf("expected cache_salt to be stripped for Anthropic messages passthrough, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsCacheSaltForBedrockOpenAICompatible
// covers the Bedrock non-Anthropic branch (OpenAI-compatible passthrough,
// e.g. GLM/Llama): body forwards mostly as-is, but a stray cache_salt must
// still be removed since Bedrock's OpenAI-compatible layer rejects it.
func TestProviderConverter_RequestFrom_StripsCacheSaltForBedrockOpenAICompatible(t *testing.T) {
	body := []byte(`{"model":"zai.glm-4.7-flash","cache_salt":"partition-1","messages":[]}`)

	c := New(config.ProviderTypeBedrock, RequestMode{ModelID: "zai.glm-4.7-flash"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["cache_salt"]; present {
		t.Fatalf("expected cache_salt to be stripped for Bedrock OpenAI-compatible passthrough, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsCacheSaltForEmbeddings covers the
// IsEmbeddings default branch (OpenAI/Proxy/AIR/etc. embeddings passthrough),
// which previously bypassed cache_salt stripping entirely.
func TestProviderConverter_RequestFrom_StripsCacheSaltForEmbeddings(t *testing.T) {
	body := []byte(`{"model":"text-embedding-3-small","cache_salt":"partition-1","input":"hi"}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		IsEmbeddings: true,
		ModelID:      "text-embedding-3-small",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["cache_salt"]; present {
		t.Fatalf("expected cache_salt to be stripped for embeddings, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsStreamOptionsExtrasForOpenAI covers
// the "default" (OpenAI-compatible) branch: real api.openai.com doesn't
// understand vLLM's stream_options.continuous_usage_stats extension key
// (rejects it outright with a 400), so it must be stripped down to just
// include_usage for everyone except vLLM.
func TestProviderConverter_RequestFrom_StripsStreamOptionsExtrasForOpenAI(t *testing.T) {
	body := []byte(`{"model":"gpt-5-mini","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[]}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		ModelID:     "gpt-5-mini",
		IsStreaming: true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	streamOptions, ok := m["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options map, got %T", m["stream_options"])
	}
	if len(streamOptions) != 1 || streamOptions["include_usage"] != true {
		t.Fatalf("expected stream_options to contain only include_usage=true, got %v", streamOptions)
	}
}

// TestProviderConverter_RequestFrom_PreservesStreamOptionsExtrasForVLLM covers
// the vLLM exception: self-hosted vLLM understands continuous_usage_stats
// (its own streaming-usage extension), so RequestFrom must leave a client's
// stream_options object untouched instead of stripping it down.
func TestProviderConverter_RequestFrom_PreservesStreamOptionsExtrasForVLLM(t *testing.T) {
	body := []byte(`{"model":"qwen3-32b","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[]}`)

	c := New(config.ProviderTypeVLLM, RequestMode{
		ModelID:     "qwen3-32b",
		IsStreaming: true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	streamOptions, ok := m["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options map, got %T", m["stream_options"])
	}
	if streamOptions["continuous_usage_stats"] != true {
		t.Fatalf("expected continuous_usage_stats to be preserved for vLLM, got %v", streamOptions)
	}
}

// TestProviderConverter_RequestFrom_StripsStreamOptionsExtrasForBedrockOpenAICompatible
// covers the Bedrock non-Anthropic branch (OpenAI-compatible passthrough,
// e.g. GLM/Llama): same rule as the default branch applies here too.
func TestProviderConverter_RequestFrom_StripsStreamOptionsExtrasForBedrockOpenAICompatible(t *testing.T) {
	body := []byte(`{"model":"zai.glm-4.7-flash","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[]}`)

	c := New(config.ProviderTypeBedrock, RequestMode{
		ModelID:     "zai.glm-4.7-flash",
		IsStreaming: true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	streamOptions, ok := m["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options map, got %T", m["stream_options"])
	}
	if len(streamOptions) != 1 || streamOptions["include_usage"] != true {
		t.Fatalf("expected stream_options to contain only include_usage=true, got %v", streamOptions)
	}
}

// TestProviderConverter_RequestFrom_LeavesStreamOptionsAloneWhenNotStreaming
// guards against RebuildStreamOptionsIncludeUsageOnly running on a
// non-streaming request: IsStreaming gates the check, so a stray
// stream_options-shaped field on a non-streaming body (unusual, but not
// impossible) is left untouched.
func TestProviderConverter_RequestFrom_LeavesStreamOptionsAloneWhenNotStreaming(t *testing.T) {
	body := []byte(`{"model":"gpt-5-mini","stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[]}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		ModelID: "gpt-5-mini",
		// IsStreaming intentionally left false.
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	streamOptions, ok := m["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options map, got %T", m["stream_options"])
	}
	if streamOptions["continuous_usage_stats"] != true {
		t.Fatalf("expected stream_options to be left untouched for a non-streaming request, got %v", streamOptions)
	}
}

// TestProviderConverter_RequestFrom_StripsStreamOptionsForAnthropicMessagesPassthrough
// covers the MessagesPassthrough branch: native api.anthropic.com has no
// stream_options concept at all (rejects the whole field, not just
// unrecognized keys inside it), and a client can still send it directly on a
// /v1/messages request since ingress sanitization only skips *injecting*
// stream_options for isMessagesAPI, it doesn't strip one the client sent.
func TestProviderConverter_RequestFrom_StripsStreamOptionsForAnthropicMessagesPassthrough(t *testing.T) {
	body := []byte(`{"model":"claude-test","stream":true,"stream_options":{"include_usage":true},"messages":[]}`)

	c := New(config.ProviderTypeAnthropic, RequestMode{
		ModelID:             "claude-test",
		MessagesPassthrough: true,
		IsStreaming:         true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["stream_options"]; present {
		t.Fatalf("expected stream_options to be stripped for Anthropic messages passthrough, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForOpenAICompatible
// covers the "default" (OpenAI-compatible) branch: plugins/provider are
// genuine OpenRouter-only features that no other destination sharing this
// bucket (aggregators, genuine api.openai.com, ...) is confirmed to
// understand.
func TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForOpenAICompatible(t *testing.T) {
	body := []byte(`{"model":"gpt-5-mini","plugins":[{"id":"web"}],"provider":{"order":["openai"]},"messages":[]}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		ModelID: "gpt-5-mini",
		BaseURL: "https://api.cometapi.com/v1",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["plugins"]; present {
		t.Fatalf("expected plugins to be stripped, got %s", string(got))
	}
	if _, present := m["provider"]; present {
		t.Fatalf("expected provider to be stripped, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_PreservesOpenRouterOnlyFieldsForGenuineOpenRouter
// covers the OpenRouter exception to shouldStripOpenRouterOnlyFields: the one
// destination actually confirmed to support plugins (paid web search) and
// provider (vendor routing preference).
func TestProviderConverter_RequestFrom_PreservesOpenRouterOnlyFieldsForGenuineOpenRouter(t *testing.T) {
	body := []byte(`{"model":"gpt-5-mini","plugins":[{"id":"web"}],"provider":{"order":["openai"]},"messages":[]}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		ModelID: "gpt-5-mini",
		BaseURL: "https://openrouter.ai/api/v1",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["plugins"]; !present {
		t.Fatalf("expected plugins to be preserved for genuine OpenRouter, got %s", string(got))
	}
	if _, present := m["provider"]; !present {
		t.Fatalf("expected provider to be preserved for genuine OpenRouter, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForAnthropicMessagesPassthrough
// covers the MessagesPassthrough branch: a client can still send OpenRouter
// fields on a /v1/messages request forwarded natively to an Anthropic-wire
// provider, which never understands them.
func TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForAnthropicMessagesPassthrough(t *testing.T) {
	body := []byte(`{"model":"claude-test","plugins":[{"id":"web"}],"messages":[]}`)

	c := New(config.ProviderTypeAnthropic, RequestMode{
		ModelID:             "claude-test",
		MessagesPassthrough: true,
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["plugins"]; present {
		t.Fatalf("expected plugins to be stripped for Anthropic messages passthrough, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForBedrockOpenAICompatible
// covers the Bedrock non-Anthropic branch (OpenAI-compatible passthrough,
// e.g. GLM/Llama): same rule as the default branch applies here too.
func TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForBedrockOpenAICompatible(t *testing.T) {
	body := []byte(`{"model":"zai.glm-4.7-flash","plugins":[{"id":"web"}],"messages":[]}`)

	c := New(config.ProviderTypeBedrock, RequestMode{ModelID: "zai.glm-4.7-flash"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["plugins"]; present {
		t.Fatalf("expected plugins to be stripped for Bedrock OpenAI-compatible passthrough, got %s", string(got))
	}
}

// TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForEmbeddings
// covers the IsEmbeddings default branch.
func TestProviderConverter_RequestFrom_StripsOpenRouterOnlyFieldsForEmbeddings(t *testing.T) {
	body := []byte(`{"model":"text-embedding-3-small","plugins":[{"id":"web"}],"input":"hi"}`)

	c := New(config.ProviderTypeOpenAI, RequestMode{
		IsEmbeddings: true,
		ModelID:      "text-embedding-3-small",
	})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	m := mustUnmarshal[map[string]any](t, got)
	if _, present := m["plugins"]; present {
		t.Fatalf("expected plugins to be stripped for embeddings, got %s", string(got))
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

func TestProviderConverter_RequestFrom_BedrockAnthropicGlobal(t *testing.T) {
	body := mustJSON(t, minimalOpenAIChatRequest())

	c := New(config.ProviderTypeBedrock, RequestMode{ModelID: "global.anthropic.claude-opus-4-7"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}

	var req map[string]any
	if err := json.Unmarshal(got, &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if req["anthropic_version"] != "bedrock-2023-05-31" {
		t.Fatalf("expected bedrock anthropic version, got %#v", req["anthropic_version"])
	}
	if _, ok := req["model"]; ok {
		t.Fatalf("bedrock request body must not contain model field: %s", string(got))
	}
	if req["max_tokens"] != float64(4096) {
		t.Fatalf("expected default max_tokens 4096, got %#v", req["max_tokens"])
	}
}

func TestProviderConverter_RequestFrom_BedrockOpenAICompatiblePassthrough(t *testing.T) {
	body := mustJSON(t, minimalOpenAIChatRequest())

	c := New(config.ProviderTypeBedrock, RequestMode{ModelID: "zai.glm-4.7-flash"})
	got, err := c.RequestFrom(body)
	if err != nil {
		t.Fatalf("RequestFrom error: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("expected passthrough body, got %s", string(got))
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

func TestProviderConverter_RequestFrom_GeminiImageGeneration_Size(t *testing.T) {
	tests := []struct {
		size        string
		aspectRatio string
		imageSize   string
	}{
		{size: "1024x1024", aspectRatio: "1:1", imageSize: "1K"},
		{size: "1x1", aspectRatio: "1:1", imageSize: "1K"},
		{size: "3:4", aspectRatio: "3:4", imageSize: "1K"},
		{size: "1792x1024", aspectRatio: "16:9", imageSize: "1K"},
		{size: "1792x2400", aspectRatio: "3:4", imageSize: "2K"},
		{size: "896x1152", aspectRatio: "4:5", imageSize: "1K"},
		{size: "640x1024", aspectRatio: "2:3", imageSize: "1K"},
		{size: "792x168", aspectRatio: "21:9", imageSize: "512"},
	}

	for _, tt := range tests {
		t.Run(tt.size, func(t *testing.T) {
			body := mustJSON(t, openai.OpenAIImageRequest{
				Model:  "gemini-3.1-flash-image-preview",
				Prompt: "make image",
				Size:   tt.size,
			})
			c := New(config.ProviderTypeGemini, RequestMode{
				IsImageGeneration: true,
				ModelID:           "gemini-3.1-flash-image-preview",
			})

			got, err := c.RequestFrom(body)
			if err != nil {
				t.Fatalf("RequestFrom error: %v", err)
			}

			req := mustUnmarshal[vertex.VertexRequest](t, got)
			if req.GenerationConfig == nil || req.GenerationConfig.ImageConfig == nil {
				t.Fatalf("missing generationConfig.imageConfig in %s", got)
			}
			if req.GenerationConfig.ImageConfig.AspectRatio != tt.aspectRatio {
				t.Fatalf("expected aspectRatio %q, got %q", tt.aspectRatio, req.GenerationConfig.ImageConfig.AspectRatio)
			}
			if req.GenerationConfig.ImageConfig.ImageSize != tt.imageSize {
				t.Fatalf("expected imageSize %q, got %q", tt.imageSize, req.GenerationConfig.ImageConfig.ImageSize)
			}
		})
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
		Usage: &anthropic.AnthropicUsage{
			InputTokens:              5,
			OutputTokens:             7,
			CacheReadInputTokens:     2,
			CacheCreationInputTokens: 3,
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
	if resp.ID != "chatcmpl-msg_1" {
		t.Fatalf("expected id chatcmpl-msg_1, got %q", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	msg := resp.Choices[0].Message
	if msg.Content != "hello" || msg.ReasoningContent != "hmm" || len(msg.ToolCalls) != 1 {
		t.Fatalf("unexpected message: %+v", msg)
	}
	// PromptTokens = InputTokens + CacheReadInputTokens + CacheCreationInputTokens = 5 + 2 + 3 = 10
	if resp.Usage == nil || resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 17 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}
	if resp.Usage.PromptTokensDetails == nil ||
		resp.Usage.PromptTokensDetails.CachedTokens != 2 ||
		resp.Usage.PromptTokensDetails.CacheCreationTokens != 3 {
		t.Fatalf("unexpected prompt token details: %+v", resp.Usage.PromptTokensDetails)
	}
}

func TestProviderConverter_ResponseTo_BedrockAnthropicGlobalAlias(t *testing.T) {
	anthropicResp := anthropic.AnthropicResponse{
		ID:         "msg_1",
		Type:       "message",
		Role:       "assistant",
		Model:      "global.anthropic.claude-opus-4-7",
		StopReason: "end_turn",
		Usage: &anthropic.AnthropicUsage{
			InputTokens:  5,
			OutputTokens: 7,
		},
		Content: []anthropic.ContentBlock{
			{Type: "text", Text: "hello"},
		},
	}
	body := mustJSON(t, anthropicResp)

	c := New(config.ProviderTypeBedrock, RequestMode{
		ModelID:        "global.anthropic.claude-opus-4-7",
		DisplayModelID: "anthropic/claude-opus-4.7",
	})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}

	resp := mustUnmarshal[openai.OpenAIResponse](t, got)
	if resp.Model != "anthropic/claude-opus-4.7" {
		t.Fatalf("expected alias model in response, got %q", resp.Model)
	}
}

func TestProviderConverter_ResponseTo_BedrockOpenAICompatibleAlias(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","model":"zai.glm-4.7-flash","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`)

	c := New(config.ProviderTypeBedrock, RequestMode{
		ModelID:        "zai.glm-4.7-flash",
		DisplayModelID: "z-ai/glm-4.7-flash",
	})
	got, err := c.ResponseTo(body)
	if err != nil {
		t.Fatalf("ResponseTo error: %v", err)
	}
	if !strings.Contains(string(got), `"model":"z-ai/glm-4.7-flash"`) {
		t.Fatalf("expected alias model in response body, got %s", string(got))
	}
	if strings.Contains(string(got), `"model":"zai.glm-4.7-flash"`) {
		t.Fatalf("expected real model id to be replaced, got %s", string(got))
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

	{
		c := New(config.ProviderTypeBedrock, RequestMode{
			ModelID:        "zai.glm-4.7-flash",
			DisplayModelID: "z-ai/glm-4.7-flash",
			IsStreaming:    true,
		})
		frame := buildBedrockEventStreamFrame(t, `{"id":"chatcmpl-1","model":"zai.glm-4.7-flash","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`)
		var out bytes.Buffer
		if err := c.StreamTo(bytes.NewReader(frame), &out); err != nil {
			t.Fatalf("StreamTo error: %v", err)
		}
		if !strings.Contains(out.String(), `"model":"z-ai/glm-4.7-flash"`) {
			t.Fatalf("expected alias model in stream, got %q", out.String())
		}
		if strings.Contains(out.String(), `"model":"zai.glm-4.7-flash"`) {
			t.Fatalf("expected real model id to be replaced, got %q", out.String())
		}
	}
}

func TestResponseModelAliasWriter_SplitModelToken(t *testing.T) {
	var out bytes.Buffer
	w := &responseModelAliasWriter{
		w:        &out,
		oldModel: "zai.glm-4.7-flash",
		newModel: "z-ai/glm-4.7-flash",
	}

	first := []byte("data: {\"model\":\"zai.glm-")
	second := []byte("4.7-flash\",\"choices\":[]}\n\n")

	if _, err := w.Write(first); err != nil {
		t.Fatalf("Write first chunk error: %v", err)
	}
	if _, err := w.Write(second); err != nil {
		t.Fatalf("Write second chunk error: %v", err)
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush error: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, `"model":"z-ai/glm-4.7-flash"`) {
		t.Fatalf("expected alias model after split writes, got %q", got)
	}
	if strings.Contains(got, `"model":"zai.glm-4.7-flash"`) {
		t.Fatalf("expected real model id to be replaced, got %q", got)
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

	cred.BaseURL = "https://example.com/v1"
	got = cAnthropic.BuildURL(cred)
	if got != "https://example.com/v1/messages" {
		t.Fatalf("unexpected anthropic versioned url: %q", got)
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
	if !New(config.ProviderTypeAIR, RequestMode{}).IsPassthrough() {
		t.Fatalf("air should be passthrough")
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

func TestProviderConverter_UsageFromResponseProxyDefaultsToOpenAISemantics(t *testing.T) {
	body := []byte(`{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}`)

	proxyUsage := New(config.ProviderTypeProxy, RequestMode{}).UsageFromResponse(body)
	if proxyUsage == nil {
		t.Fatal("expected proxy usage")
		return
	}
	if proxyUsage.AudioInputTokens != 60 {
		t.Fatalf("generic proxy should default to raw OpenAI-compatible usage, got %+v", proxyUsage)
	}

	openAIUsage := New(config.ProviderTypeOpenAI, RequestMode{}).UsageFromResponse(body)
	if openAIUsage == nil {
		t.Fatal("expected OpenAI usage")
		return
	}
	if openAIUsage.AudioInputTokens != 60 {
		t.Fatalf("OpenAI-compatible raw usage should subtract cached audio, got %+v", openAIUsage)
	}
}

func TestExtractTokenUsage(t *testing.T) {
	if got := ExtractTokenUsage(nil); got != nil {
		t.Fatalf("expected nil for empty body")
	}
	if got := ExtractTokenUsage([]byte("not-json")); got != nil {
		t.Fatalf("expected nil for invalid json")
	}

	chatBody := []byte(`{"usage":{"prompt_tokens":15,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":5,"cached_audio_tokens":2,"cache_creation_tokens":4,"cache_creation_token_details":{"ephemeral_5m_input_tokens":1,"ephemeral_1h_input_tokens":3},"audio_tokens":3},"completion_tokens_details":{"accepted_prediction_tokens":3,"rejected_prediction_tokens":1,"audio_tokens":4,"cached_tokens":2,"reasoning_tokens":6,"image_tokens":2,"text_tokens":5}}}`)
	usage := ExtractTokenUsage(chatBody)
	if usage == nil {
		t.Fatalf("expected usage for chat format")
		return
	}
	if usage.PromptTokens != 15 || usage.CompletionTokens != 7 {
		t.Fatalf("unexpected chat token counts: %+v", usage)
	}
	if usage.CachedInputTokens != 5 || usage.CachedAudioInputTokens != 2 || usage.CachedOutputTokens != 2 || usage.CacheCreationTokens != 4 || usage.CacheCreation5mTokens != 1 || usage.CacheCreation1hTokens != 3 || usage.AudioInputTokens != 1 || usage.AudioOutputTokens != 4 || usage.OutputTextTokens != 5 || usage.ReasoningTokens != 6 {
		t.Fatalf("unexpected details: %+v", usage)
	}
	if usage.AcceptedPredictionTokens != 3 || usage.RejectedPredictionTokens != 1 {
		t.Fatalf("unexpected prediction tokens: %+v", usage)
	}
	if usage.OutputImageTokens != 2 {
		t.Fatalf("expected output image tokens, got %+v", usage)
	}

	imageBody := []byte(`{"usage":{"input_tokens":9,"output_tokens":10,"input_tokens_details":{"image_tokens":8},"output_tokens_details":{"image_tokens":6,"text_tokens":4}}}`)
	usage = ExtractTokenUsage(imageBody)
	if usage == nil {
		t.Fatalf("expected usage for image format")
		return
	}
	if usage.PromptTokens != 9 || usage.CompletionTokens != 10 || usage.ImageTokens != 8 || usage.OutputImageTokens != 6 || usage.OutputTextTokens != 4 {
		t.Fatalf("unexpected image token counts: %+v", usage)
	}

	zeroBody := []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0}}`)
	if got := ExtractTokenUsage(zeroBody); got != nil {
		t.Fatalf("expected nil for zero usage")
	}
}

func TestExtractTokenUsageFromConvertedGeminiImage(t *testing.T) {
	providerBody := []byte(`{
		"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]}}],
		"usageMetadata":{"promptTokenCount":1128,"candidatesTokenCount":1357,"totalTokenCount":2485}
	}`)
	imageConverter := New(config.ProviderTypeGemini, RequestMode{
		IsImageGeneration: true,
		ModelID:           "gemini-3.1-flash-lite-image",
	})
	convertedBody, err := imageConverter.ResponseTo(providerBody)
	if err != nil {
		t.Fatalf("convert Gemini image response: %v", err)
	}

	usage := imageConverter.UsageFromResponse(convertedBody)
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.PromptTokens != 1128 || usage.CompletionTokens != 1357 {
		t.Fatalf("unexpected totals: %+v", usage)
	}
	if usage.OutputImageTokens != 1357 {
		t.Fatalf("expected all 1357 generated tokens to be image tokens, got %d", usage.OutputImageTokens)
	}
}

func TestExtractTokenUsage_AnthropicFlatCacheFields(t *testing.T) {
	// Anthropic's input_tokens is EXCLUSIVE of cache tokens (unlike OpenAI's inclusive
	// prompt_tokens), so PromptTokens must be corrected back to the inclusive total
	// (70+25+5=100) here — CalculateTokenCosts subtracts CachedInputTokens/CacheCreationTokens
	// from PromptTokens assuming it is always the inclusive total.
	body := []byte(`{"usage":{"input_tokens":70,"output_tokens":20,"cache_read_input_tokens":25,"cache_creation_input_tokens":5}}`)

	usage := ExtractTokenUsage(body)

	if usage == nil {
		t.Fatal("expected Anthropic usage")
		return
	}
	if usage.PromptTokens != 100 || usage.CompletionTokens != 20 {
		t.Fatalf("unexpected token counts: %+v", usage)
	}
	if usage.CachedInputTokens != 25 || usage.CacheCreationTokens != 5 {
		t.Fatalf("unexpected Anthropic cache usage: %+v", usage)
	}
}

func TestExtractTokenUsage_CacheCreationDetailsProvideMissingAggregate(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "responses shape",
			body: `{"usage":{"input_tokens":10,"output_tokens":1,"input_tokens_details":{"cache_creation_token_details":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":7}}}}`,
		},
		{
			name: "chat completions shape",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cache_creation_token_details":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":7}}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(tt.body))

			if usage == nil || usage.CacheCreationTokens != 10 || usage.CacheCreation5mTokens != 3 || usage.CacheCreation1hTokens != 7 {
				t.Fatalf("unexpected cache creation usage: %+v", usage)
			}
		})
	}
}

func TestExtractTokenUsage_ImageFallback(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0,"input_tokens":2,"output_tokens":3}}`)
	usage := ExtractTokenUsage(body)
	if usage == nil || usage.PromptTokens != 2 || usage.CompletionTokens != 3 {
		t.Fatalf("unexpected fallback usage: %+v", usage)
	}
}

func TestExtractTokenUsage_ResponsesAPI(t *testing.T) {
	// Responses API format (GPT-5, /v1/responses) uses input_tokens/output_tokens
	// with output_tokens_details instead of completion_tokens_details
	body := []byte(`{"usage":{"input_tokens":150,"output_tokens":80,"total_tokens":230,"input_tokens_details":{"cached_tokens":30,"audio_tokens":10},"output_tokens_details":{"reasoning_tokens":25,"audio_tokens":5,"image_tokens":40,"text_tokens":10}}}`)
	usage := ExtractTokenUsage(body)
	if usage == nil {
		t.Fatalf("expected usage for Responses API format")
		return
	}
	if usage.PromptTokens != 150 || usage.CompletionTokens != 80 {
		t.Fatalf("unexpected token counts: prompt=%d completion=%d", usage.PromptTokens, usage.CompletionTokens)
	}
	if usage.CachedInputTokens != 30 {
		t.Fatalf("expected cached_tokens=30, got %d", usage.CachedInputTokens)
	}
	if usage.AudioInputTokens != 10 {
		t.Fatalf("expected audio_input=10, got %d", usage.AudioInputTokens)
	}
	if usage.ReasoningTokens != 25 {
		t.Fatalf("expected reasoning_tokens=25, got %d", usage.ReasoningTokens)
	}
	if usage.AudioOutputTokens != 5 {
		t.Fatalf("expected audio_output=5, got %d", usage.AudioOutputTokens)
	}
	if usage.OutputImageTokens != 40 {
		t.Fatalf("expected output_image_tokens=40, got %d", usage.OutputImageTokens)
	}
	if usage.OutputTextTokens != 10 {
		t.Fatalf("expected output_text_tokens=10, got %d", usage.OutputTextTokens)
	}
}

func TestExtractTokenUsage_CachedAudioIsExcludedFromAudioInput(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "chat completions",
			body: `{"usage":{"prompt_tokens":200,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}`,
		},
		{
			name: "responses API",
			body: `{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}`,
		},
		{
			name: "responses streaming event",
			body: `{"type":"response.completed","response":{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(tt.body))

			if usage == nil {
				t.Fatal("expected usage")
				return
			}
			if usage.CachedAudioInputTokens != 40 {
				t.Fatalf("expected cached audio=40, got %+v", usage)
			}
			if usage.AudioInputTokens != 60 {
				t.Fatalf("expected non-cached audio=60, got %+v", usage)
			}
		})
	}
}

func TestExtractTokenUsage_PreservesAlreadyNormalizedAudioInput(t *testing.T) {
	body := []byte(`{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":60}}}`)

	usage := ExtractTokenUsageWithOptions(body, TokenUsageExtractionOptions{})

	if usage == nil {
		t.Fatal("expected usage")
		return
	}
	if usage.AudioInputTokens != 60 {
		t.Fatalf("expected already-normalized audio=60, got %+v", usage)
	}
}

func TestExtractTokenUsage_NegativeCachedTokensDoNotIncreaseAudioInput(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "chat completions",
			body: `{"usage":{"prompt_tokens":200,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}`,
		},
		{
			name: "responses API",
			body: `{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}`,
		},
		{
			name: "responses streaming event",
			body: `{"type":"response.completed","response":{"usage":{"input_tokens":200,"output_tokens":1,"input_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(tt.body))

			if usage == nil {
				t.Fatal("expected usage")
				return
			}
			if usage.CachedInputTokens != 0 || usage.CachedAudioInputTokens != 0 {
				t.Fatalf("expected cached fields to be sanitized, got %+v", usage)
			}
			if usage.AudioInputTokens != 100 {
				t.Fatalf("expected audio input to stay 100, got %+v", usage)
			}
		})
	}
}

func TestExtractTokenUsage_CacheWriteTokens(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "chat completions alias",
			body: `{"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cache_write_tokens":60}}}`,
		},
		{
			name: "responses API canonical field",
			body: `{"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cache_creation_tokens":60}}}`,
		},
		{
			name: "responses API streaming alias",
			body: `{"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cache_write_tokens":60}}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(tt.body))
			if usage == nil || usage.CacheCreationTokens != 60 {
				t.Fatalf("unexpected cache creation usage: %+v", usage)
			}
		})
	}
}

func TestExtractTokenUsage_ResponsesAPIStreamingEvent(t *testing.T) {
	// Responses API streaming event format: response.completed SSE event
	// Usage is nested inside response.usage, not at top level
	body := []byte(`{"type":"response.completed","response":{"id":"resp_123","object":"response","status":"completed","model":"qwen3","output":[],"usage":{"input_tokens":16,"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0,"image_tokens":2},"input_tokens_details":{"cached_tokens":5},"total_tokens":18}}}`)
	usage := ExtractTokenUsage(body)
	if usage == nil {
		t.Fatalf("expected usage for Responses API streaming event format")
		return
	}
	if usage.PromptTokens != 16 {
		t.Fatalf("expected prompt_tokens=16, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 2 {
		t.Fatalf("expected completion_tokens=2, got %d", usage.CompletionTokens)
	}
	if usage.OutputImageTokens != 2 {
		t.Fatalf("expected output_image_tokens=2, got %d", usage.OutputImageTokens)
	}
	if usage.CachedInputTokens != 5 {
		t.Fatalf("expected cached_tokens=5, got %d", usage.CachedInputTokens)
	}
}

func TestExtractTokenUsage_WebSearchRequests(t *testing.T) {
	tests := []struct {
		name string
		body string
		want int
	}{
		{
			name: "server_tool_use",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"server_tool_use":{"web_search_requests":3}}}`,
			want: 3,
		},
		{
			name: "responses output items",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12},"output":[{"type":"web_search_call","status":"completed"},{"type":"web_search_call","status":"completed"},{"type":"message","status":"completed"}]}`,
			want: 2,
		},
		{
			name: "nested streaming response output items without token usage",
			body: `{"type":"response.completed","response":{"output":[{"type":"web_search_call","status":"completed"}]}}`,
			want: 1,
		},
		{
			name: "chat annotations fallback",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"choices":[{"message":{"annotations":[{"type":"url_citation"}]}}]}`,
			want: 1,
		},
		{
			name: "non web annotation is not billed",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"choices":[{"message":{"annotations":[{"type":"file_citation"}]}}]}`,
			want: 0,
		},
		{
			name: "x_tools is not added to output items",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_tools":{"web_search":{"count":1}}},"output":[{"type":"web_search_call","status":"completed"},{"type":"message","status":"completed"}]}`,
			want: 1,
		},
		{
			name: "x_tools wins over output items",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_tools":{"web_search":{"count":2}}},"output":[{"type":"web_search_call","status":"completed"}]}`,
			want: 2,
		},
		{
			name: "x_tools and x_details are not summed",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_details":[{"x_billing_type":"response_api","plugins":{"web_search":{"count":1}}}],"x_tools":{"web_search":{"count":1}}}}`,
			want: 1,
		},
		{
			name: "x_details without x_tools",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_details":[{"plugins":{"web_search":{"count":2}}}]},"output":[{"type":"web_search_call","status":"completed"}]}`,
			want: 2,
		},
		{
			name: "chat plugins.search",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"plugins":{"search":{"count":1,"strategy":"agent"}}}}`,
			want: 1,
		},
		{
			name: "server_tool_use wins over extensions",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"server_tool_use":{"web_search_requests":3},"plugins":{"search":{"count":1}}}}`,
			want: 3,
		},
		{
			name: "nested completed response x_tools is not added to output items",
			body: `{"type":"response.completed","response":{"output":[{"type":"web_search_call","status":"completed"}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_details":[{"plugins":{"web_search":{"count":1}}}],"x_tools":{"web_search":{"count":1}}}}}`,
			want: 1,
		},
		{
			name: "nested completed response x_tools wins over output items",
			body: `{"type":"response.completed","response":{"output":[{"type":"web_search_call","status":"completed"}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_tools":{"web_search":{"count":3}}}}}`,
			want: 3,
		},
		{
			name: "unexpected extension shape falls back to output items",
			body: `{"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"x_tools":"n/a","plugins":["search"]},"output":[{"type":"web_search_call","status":"completed"}]}`,
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(tt.body))
			if usage == nil {
				t.Fatalf("expected usage")
				return
			}
			if usage.WebSearchRequests != tt.want {
				t.Fatalf("expected web_search_requests=%d, got %d", tt.want, usage.WebSearchRequests)
			}
		})
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

func buildBedrockEventStreamFrame(t *testing.T, innerJSON string) []byte {
	t.Helper()

	innerBase64 := base64.StdEncoding.EncodeToString([]byte(innerJSON))
	payload := []byte(`{"bytes":"` + innerBase64 + `","p":""}`)
	totalLength := uint32(12 + len(payload) + 4)

	frame := make([]byte, 12+len(payload)+4)
	binary.BigEndian.PutUint32(frame[0:4], totalLength)
	binary.BigEndian.PutUint32(frame[4:8], 0)
	copy(frame[12:], payload)
	return frame
}
