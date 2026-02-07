package anthropic

import (
	"github.com/mixaill76/auto_ai_router/internal/transform"
)

// CacheCreationDetails contains breakdown of cached tokens by TTL
type CacheCreationDetails struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
}

// ServerToolUsageDetails contains count of server tool requests
type ServerToolUsageDetails struct {
	WebSearchRequests int `json:"web_search_requests,omitempty"`
}

// AnthropicUsage represents token usage from Anthropic API
// See: https://platform.claude.com/docs/en/api/messages
type AnthropicUsage struct {
	InputTokens              int                     `json:"input_tokens"`
	OutputTokens             int                     `json:"output_tokens"`
	CacheCreationInputTokens int                     `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                     `json:"cache_read_input_tokens,omitempty"`
	CacheCreation            *CacheCreationDetails   `json:"cache_creation,omitempty"`
	ServerToolUse            *ServerToolUsageDetails `json:"server_tool_use,omitempty"`
	ServiceTier              string                  `json:"service_tier,omitempty"` // "standard", "priority", or "batch"
	InferenceGeo             string                  `json:"inference_geo,omitempty"`
}

// ToTokenUsage converts Anthropic usage to universal TokenUsage format
// Note: Total input tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
func (u *AnthropicUsage) ToTokenUsage() *transform.TokenUsage {
	if u == nil {
		return nil
	}

	return &transform.TokenUsage{
		PromptTokens:      u.InputTokens,
		CompletionTokens:  u.OutputTokens,
		CachedInputTokens: u.CacheCreationInputTokens + u.CacheReadInputTokens,
	}
}
