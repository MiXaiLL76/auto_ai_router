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
	Thinking             interface{}            `json:"thinking,omitempty"` // Anthropic-style thinking param: {"type":"enabled","budget_tokens":N}
	ServiceTier          string                 `json:"service_tier,omitempty"`
	StreamOptions        interface{}            `json:"stream_options,omitempty"`
	Verbosity            string                 `json:"verbosity,omitempty"`
	Prediction           interface{}            `json:"prediction,omitempty"`
	WebSearchOptions     interface{}            `json:"web_search_options,omitempty"`
	ExtraBody            map[string]interface{} `json:"extra_body,omitempty"`
}

type OpenAIMessage struct {
	Role             string        `json:"role"`
	Content          interface{}   `json:"content"`
	Name             string        `json:"name,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
	ToolCalls        []interface{} `json:"tool_calls,omitempty"`
	Refusal          string        `json:"refusal,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
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
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
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
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ImageData struct {
	B64JSON  string    `json:"b64_json,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type TokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

type CompletionTokenDetails struct {
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens"`
	PromptTokensDetails     *TokenDetails           `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokenDetails `json:"completion_tokens_details,omitempty"`
}

// Streaming types

// OpenAIStreamingChunk represents OpenAI streaming response format
type OpenAIStreamingChunk struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Choices []OpenAIStreamingChoice `json:"choices"`
	Usage   *OpenAIUsage            `json:"usage,omitempty"`
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
	Index    int                          `json:"index"`
	ID       string                       `json:"id,omitempty"`
	Type     string                       `json:"type,omitempty"`
	Function *OpenAIStreamingToolFunction `json:"function,omitempty"`
}

type OpenAIStreamingToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// Image generation types

// OpenAIImageRequest represents OpenAI image generation request
type OpenAIImageRequest struct {
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	N                 *int   `json:"n,omitempty"`
	Size              string `json:"size,omitempty"`
	Quality           string `json:"quality,omitempty"`
	ResponseFormat    string `json:"response_format,omitempty"`
	Style             string `json:"style,omitempty"`
	User              string `json:"user,omitempty"`
	Background        string `json:"background,omitempty"`         // gpt-image-1
	Moderation        string `json:"moderation,omitempty"`         // gpt-image-1
	OutputCompression int    `json:"output_compression,omitempty"` // gpt-image-1
	OutputFormat      string `json:"output_format,omitempty"`      // gpt-image-1
}

type OpenAIImageData struct {
	B64JSON       string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// OpenAIImageResponse represents OpenAI image response
type OpenAIImageResponse struct {
	Created int64             `json:"created"`
	Data    []OpenAIImageData `json:"data"`
}
