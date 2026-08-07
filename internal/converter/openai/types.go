package openai

// Request types

// OpenAIRequest represents the OpenAI API request format
type OpenAIRequest struct {
	Model                string                 `json:"model"`
	Messages             []OpenAIMessage        `json:"messages"`
	Temperature          *float64               `json:"temperature,omitempty"`
	MaxTokens            *int                   `json:"max_tokens,omitempty"`
	MaxCompletionTokens  *int                   `json:"max_completion_tokens,omitempty"`
	Stream               bool                   `json:"stream,omitempty"`
	TopP                 *float64               `json:"top_p,omitempty"`
	Stop                 interface{}            `json:"stop,omitempty"`
	N                    *int                   `json:"n,omitempty"`
	FrequencyPenalty     *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty      *float64               `json:"presence_penalty,omitempty"`
	LogitBias            map[string]int         `json:"logit_bias,omitempty"`
	Logprobs             *bool                  `json:"logprobs,omitempty"`
	TopLogprobs          *int                   `json:"top_logprobs,omitempty"`
	Seed                 *int64                 `json:"seed,omitempty"`
	User                 string                 `json:"user,omitempty"`
	ResponseFormat       interface{}            `json:"response_format,omitempty"`
	Tools                []interface{}          `json:"tools,omitempty"`
	ToolChoice           interface{}            `json:"tool_choice,omitempty"`
	Store                *bool                  `json:"store,omitempty"`
	ParallelToolCalls    *bool                  `json:"parallel_tool_calls,omitempty"`
	PromptCacheKey       string                 `json:"prompt_cache_key,omitempty"`
	SafetyIdentifier     string                 `json:"safety_identifier,omitempty"`
	Metadata             map[string]string      `json:"metadata,omitempty"`
	Modalities           []string               `json:"modalities,omitempty"`
	PromptCacheRetention string                 `json:"prompt_cache_retention,omitempty"`
	ReasoningEffort      string                 `json:"reasoning_effort,omitempty"`
	Reasoning            interface{}            `json:"reasoning,omitempty"`       // {"effort":"high","generate_summary":"auto"}
	Thinking             interface{}            `json:"thinking,omitempty"`        // Anthropic-style thinking param: {"type":"enabled","budget_tokens":N}
	ThinkingBudget       interface{}            `json:"thinking_budget,omitempty"` // Gemini-style thinking budget: int (tokens) or -1 (dynamic)
	ThinkingLevel        string                 `json:"thinking_level,omitempty"`  // Gemini-style thinking level: "low"/"medium"/"high"
	StreamOptions        interface{}            `json:"stream_options,omitempty"`
	Verbosity            string                 `json:"verbosity,omitempty"`
	Prediction           interface{}            `json:"prediction,omitempty"`
	WebSearchOptions     interface{}            `json:"web_search_options,omitempty"`
	ImageConfig          map[string]interface{} `json:"imageConfig,omitempty"`
	ImageConfigSnake     map[string]interface{} `json:"image_config,omitempty"`
	AspectRatio          string                 `json:"aspectRatio,omitempty"`
	AspectRatioSnake     string                 `json:"aspect_ratio,omitempty"`
	ImageSize            string                 `json:"imageSize,omitempty"`
	ImageSizeSnake       string                 `json:"image_size,omitempty"`
	ExtraBody            map[string]interface{} `json:"extra_body,omitempty"`
}

type OpenAIMessage struct {
	Role             string                `json:"role"`
	Content          interface{}           `json:"content"`
	Name             string                `json:"name,omitempty"`
	ToolCallID       string                `json:"tool_call_id,omitempty"`
	ToolCalls        []interface{}         `json:"tool_calls,omitempty"`
	ThinkingBlocks   []OpenAIThinkingBlock `json:"thinking_blocks,omitempty"`
	Refusal          string                `json:"refusal,omitempty"`
	ReasoningContent string                `json:"reasoning_content,omitempty"`
}

type OpenAIThinkingBlock struct {
	Type         string      `json:"type"`
	Thinking     string      `json:"thinking,omitempty"`
	Signature    string      `json:"signature,omitempty"`
	Data         string      `json:"data,omitempty"`
	CacheControl interface{} `json:"cache_control,omitempty"`
}

type ContentBlock struct {
	Type       string     `json:"type"`
	Text       string     `json:"text,omitempty"`
	ImageURL   *ImageURL  `json:"image_url,omitempty"`
	InputAudio *AudioData `json:"input_audio,omitempty"`
	VideoURL   *VideoURL  `json:"video_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type AudioData struct {
	Data   string `json:"data"`             // base64-encoded audio data
	Format string `json:"format,omitempty"` // e.g., "wav", "mp3", "ogg"
}

type VideoURL struct {
	URL string `json:"url"`
}

// Response types

// OpenAIResponse represents OpenAI response format
type OpenAIResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
	ServiceTier       string         `json:"service_tier,omitempty"`
	Choices           []OpenAIChoice `json:"choices"`
	Usage             *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int                   `json:"index"`
	Message      OpenAIResponseMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type OpenAIResponseMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
	Refusal          string           `json:"refusal,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	Images           []ImageData      `json:"images,omitempty"` // custom extension for Gemini image responses
}

type OpenAIToolCall struct {
	ID                     string                 `json:"id"`
	Type                   string                 `json:"type"`
	Function               OpenAIToolFunction     `json:"function"`
	ProviderSpecificFields map[string]interface{} `json:"provider_specific_fields,omitempty"`
}

type OpenAIToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ImageData struct {
	Type     string    `json:"type,omitempty"`
	Index    *int      `json:"index,omitempty"`
	B64JSON  string    `json:"b64_json,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type TokenDetails struct {
	CachedTokens              int                        `json:"cached_tokens,omitempty"`
	CachedAudioTokens         int                        `json:"cached_audio_tokens,omitempty"`
	CacheCreationTokens       int                        `json:"cache_creation_tokens,omitempty"`
	CacheCreationTokenDetails *CacheCreationTokenDetails `json:"cache_creation_token_details,omitempty"`
	CacheWriteTokens          int                        `json:"cache_write_tokens,omitempty"`
	AudioTokens               int                        `json:"audio_tokens,omitempty"`
}

// CacheCreationTokenDetails preserves Anthropic's cache-write TTL breakdown.
// The enclosing CacheCreationTokens remains the aggregate for compatibility.
type CacheCreationTokenDetails struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
}

type CompletionTokenDetails struct {
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	ImageTokens              int `json:"image_tokens,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens"`
	PromptTokensDetails     *TokenDetails           `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokenDetails `json:"completion_tokens_details,omitempty"`
	ServerToolUse           *ServerToolUseDetails   `json:"server_tool_use,omitempty"`
}

type ServerToolUseDetails struct {
	WebSearchRequests int `json:"web_search_requests,omitempty"`
}

// Streaming types

// OpenAIStreamingChunk represents OpenAI streaming response format
type OpenAIStreamingChunk struct {
	ID                string                  `json:"id"`
	Object            string                  `json:"object"`
	Created           int64                   `json:"created"`
	Model             string                  `json:"model"`
	SystemFingerprint string                  `json:"system_fingerprint,omitempty"`
	ServiceTier       string                  `json:"service_tier,omitempty"`
	Choices           []OpenAIStreamingChoice `json:"choices"`
	Usage             *OpenAIUsage            `json:"usage,omitempty"`
}

type OpenAIStreamingChoice struct {
	Index        int                  `json:"index"`
	Delta        OpenAIStreamingDelta `json:"delta"`
	FinishReason *string              `json:"finish_reason"`
}

type OpenAIStreamingDelta struct {
	Role             string                    `json:"role,omitempty"`
	Content          string                    `json:"content,omitempty"`
	ToolCalls        []OpenAIStreamingToolCall `json:"tool_calls,omitempty"`
	Refusal          string                    `json:"refusal,omitempty"`
	ReasoningContent string                    `json:"reasoning_content,omitempty"`
}

type OpenAIStreamingToolCall struct {
	Index                  int                          `json:"index"`
	ID                     string                       `json:"id,omitempty"`
	Type                   string                       `json:"type,omitempty"`
	Function               *OpenAIStreamingToolFunction `json:"function,omitempty"`
	ProviderSpecificFields map[string]interface{}       `json:"provider_specific_fields,omitempty"`
}

type OpenAIStreamingToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// Image generation types

// OpenAIImageRequest represents OpenAI image generation request
type OpenAIImageRequest struct {
	Model             string   `json:"model"`
	Prompt            string   `json:"prompt"`
	N                 *int     `json:"n,omitempty"`
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	Seed              *int64   `json:"seed,omitempty"`
	Size              string   `json:"size,omitempty"`
	Quality           string   `json:"quality,omitempty"`
	ResponseFormat    string   `json:"response_format,omitempty"`
	Style             string   `json:"style,omitempty"`
	User              string   `json:"user,omitempty"`
	Background        string   `json:"background,omitempty"`         // gpt-image-1
	Moderation        string   `json:"moderation,omitempty"`         // gpt-image-1
	OutputCompression int      `json:"output_compression,omitempty"` // gpt-image-1
	OutputFormat      string   `json:"output_format,omitempty"`      // gpt-image-1
}

type OpenAIImageData struct {
	B64JSON       string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// OpenAIImageInputTokenDetails breaks down input token counts for image API responses.
type OpenAIImageInputTokenDetails struct {
	TextTokens  int `json:"text_tokens"`
	ImageTokens int `json:"image_tokens"`
}

// OpenAIImageOutputTokenDetails breaks down output token counts for image API
// responses. Not part of OpenAI's own gpt-image-1 schema (whose output is
// always 100% image), but needed for Gemini-backed models, which can return
// a text caption alongside the generated image within the same output token
// count. Mirrors the same field the generic usage extractor
// (converter.ExtractTokenUsageWithOptions) already reads from
// output_tokens_details.image_tokens for the Responses API shape.
type OpenAIImageOutputTokenDetails struct {
	TextTokens  int `json:"text_tokens,omitempty"`
	ImageTokens int `json:"image_tokens,omitempty"`
}

// OpenAIImageUsage is the usage format for the images API (gpt-image-1 style),
// distinct from the chat-completions OpenAIUsage.
type OpenAIImageUsage struct {
	InputTokens         int                            `json:"input_tokens"`
	InputTokensDetails  OpenAIImageInputTokenDetails   `json:"input_tokens_details"`
	OutputTokens        int                            `json:"output_tokens"`
	OutputTokensDetails *OpenAIImageOutputTokenDetails `json:"output_tokens_details,omitempty"`
	TotalTokens         int                            `json:"total_tokens"`
}

// OpenAIImageResponse represents OpenAI image response
type OpenAIImageResponse struct {
	Created int64             `json:"created"`
	Data    []OpenAIImageData `json:"data"`
	Usage   *OpenAIImageUsage `json:"usage,omitempty"`
}

// Embedding types

// OpenAIEmbeddingRequest represents OpenAI embeddings request
type OpenAIEmbeddingRequest struct {
	Input          interface{} `json:"input"` // string | []string | []int | [][]int
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"` // "float" | "base64"
	Dimensions     *int        `json:"dimensions,omitempty"`
	User           string      `json:"user,omitempty"`
}

// OpenAIEmbeddingResponse represents OpenAI embeddings response
type OpenAIEmbeddingResponse struct {
	Object string                `json:"object"` // "list"
	Data   []OpenAIEmbeddingData `json:"data"`
	Model  string                `json:"model"`
	Usage  OpenAIEmbeddingUsage  `json:"usage"`
}

type OpenAIEmbeddingData struct {
	Object    string    `json:"object"` // "embedding"
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type OpenAIEmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
