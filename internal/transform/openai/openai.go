package openai

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// Request types

// OpenAIRequest represents the OpenAI API request format
type OpenAIRequest struct {
	Model            string                 `json:"model"`
	Messages         []OpenAIMessage        `json:"messages"`
	Temperature      *float64               `json:"temperature,omitempty"`
	MaxTokens        *int                   `json:"max_tokens,omitempty"`
	Stream           bool                   `json:"stream,omitempty"`
	TopP             *float64               `json:"top_p,omitempty"`
	Stop             interface{}            `json:"stop,omitempty"`
	N                *int                   `json:"n,omitempty"`
	FrequencyPenalty *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64               `json:"presence_penalty,omitempty"`
	LogitBias        map[string]float64     `json:"logit_bias,omitempty"`
	Logprobs         *bool                  `json:"logprobs,omitempty"`
	TopLogprobs      *int                   `json:"top_logprobs,omitempty"`
	Seed             *int                   `json:"seed,omitempty"`
	User             string                 `json:"user,omitempty"`
	ResponseFormat   interface{}            `json:"response_format,omitempty"`
	Tools            []interface{}          `json:"tools,omitempty"`
	ToolChoice       interface{}            `json:"tool_choice,omitempty"`
	ExtraBody        map[string]interface{} `json:"extra_body,omitempty"`
}

type OpenAIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ContentBlock struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
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
	Role    string      `json:"role"`
	Content string      `json:"content"`
	Images  []ImageData `json:"images,omitempty"`
}

type ImageData struct {
	B64JSON  string    `json:"b64_json,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// Helper functions

// GenerateID generates a unique chat completion ID
func GenerateID() string {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	return "chatcmpl-" + hex.EncodeToString(bytes)[:20]
}

// GetCurrentTimestamp returns the current Unix timestamp
func GetCurrentTimestamp() int64 {
	return time.Now().Unix()
}
