package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// enableMirrorCacheWriteTokens turns the flag on for the duration of one test
// and restores it to off (the default) afterward, since it is process-global.
func enableMirrorCacheWriteTokens(t *testing.T) {
	t.Helper()
	converterutil.SetMirrorCacheWriteTokens(true)
	t.Cleanup(func() { converterutil.SetMirrorCacheWriteTokens(false) })
}

// An Anthropic-compatible provider reports the cache write under its own
// name (cache_creation_input_tokens). The downstream usage metering reads
// cache_write_tokens in preference to cache_creation_tokens, so the write
// has to be visible under both: a customer reading only cache_write_tokens
// otherwise never sees it. Mirroring is opt-in (MirrorCacheWriteTokens,
// default false) — enabled here to exercise it.
func TestAnthropicToOpenAIReportsCacheWriteTokens(t *testing.T) {
	enableMirrorCacheWriteTokens(t)
	body := `{
		"id":"msg_cache_write",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":12,
			"cache_creation_input_tokens":2,
			"cache_read_input_tokens":3,
			"output_tokens":7
		}
	}`

	converted, err := AnthropicToOpenAI([]byte(body), "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	if response.Usage == nil || response.Usage.PromptTokensDetails == nil {
		t.Fatal("usage.prompt_tokens_details must be present")
	}
	details := response.Usage.PromptTokensDetails
	if details.CacheWriteTokens != 2 {
		t.Fatalf("cache_write_tokens must carry the cache creation count, got %d", details.CacheWriteTokens)
	}
	if details.CacheCreationTokens != 2 {
		t.Fatalf("cache_creation_tokens must stay as it was, got %d", details.CacheCreationTokens)
	}
	if details.CachedTokens != 3 {
		t.Fatalf("cached_tokens must stay as it was, got %d", details.CachedTokens)
	}

	// Both names carry the same count, so the write is counted once: the
	// metering takes cache_write_tokens when present and drops
	// cache_creation_tokens from the sum, and TokenDetails.CacheWrite() takes
	// cache_creation_tokens first and falls back to cache_write_tokens.
	if total, _, _ := details.CacheWrite(); total != 2 {
		t.Fatalf("CacheWrite() must report the write once, got %d", total)
	}

	var raw struct {
		Usage struct {
			PromptTokensDetails struct {
				CacheWriteTokens int `json:"cache_write_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(converted, &raw); err != nil {
		t.Fatalf("unmarshal raw OpenAI response: %v", err)
	}
	if raw.Usage.PromptTokensDetails.CacheWriteTokens != 2 {
		t.Fatalf("cache_write_tokens must be in the body, got %s", converted)
	}
}

// With the flag at its default (off), a cache write stays visible only under
// cache_creation_tokens, as before PR #355's mirroring: cache_write_tokens
// must not appear at all, not even as zero.
func TestAnthropicToOpenAIOmitsCacheWriteTokensWhenMirroringDisabled(t *testing.T) {
	body := `{
		"id":"msg_cache_write_disabled",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":12,
			"cache_creation_input_tokens":2,
			"cache_read_input_tokens":3,
			"output_tokens":7
		}
	}`

	converted, err := AnthropicToOpenAI([]byte(body), "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	details := response.Usage.PromptTokensDetails
	if details == nil {
		t.Fatal("usage.prompt_tokens_details must be present")
	}
	if details.CacheCreationTokens != 2 {
		t.Fatalf("cache_creation_tokens must still carry the write, got %d", details.CacheCreationTokens)
	}
	if details.CacheWriteTokens != 0 {
		t.Fatalf("cache_write_tokens must stay unset while mirroring is disabled, got %d", details.CacheWriteTokens)
	}

	var raw struct {
		Usage struct {
			PromptTokensDetails map[string]json.RawMessage `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(converted, &raw); err != nil {
		t.Fatalf("unmarshal raw OpenAI response: %v", err)
	}
	if _, exists := raw.Usage.PromptTokensDetails["cache_write_tokens"]; exists {
		t.Fatalf("cache_write_tokens must be absent from the body while mirroring is disabled, got %s", converted)
	}
}

// A response without any cache write must not grow an empty cache_write_tokens
// field: it would read as "a write of zero tokens" instead of "no write".
func TestAnthropicToOpenIROmitsCacheWriteTokensWithoutCacheWrite(t *testing.T) {
	body := `{
		"id":"msg_no_cache",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":12,"output_tokens":7}
	}`

	converted, err := AnthropicToOpenAI([]byte(body), "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var raw struct {
		Usage struct {
			PromptTokensDetails map[string]json.RawMessage `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(converted, &raw); err != nil {
		t.Fatalf("unmarshal raw OpenAI response: %v", err)
	}
	if _, exists := raw.Usage.PromptTokensDetails["cache_write_tokens"]; exists {
		t.Fatalf("cache_write_tokens must be absent without a cache write, got %s", converted)
	}
}

// The TTL split must stay under cache_creation_token_details while the
// aggregate appears under both names.
func TestAnthropicToOpenAIReportsCacheWriteWithTTLSplit(t *testing.T) {
	enableMirrorCacheWriteTokens(t)
	body := `{
		"id":"msg_cache_ttl",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":12,
			"cache_creation":{
				"ephemeral_5m_input_tokens":100,
				"ephemeral_1h_input_tokens":20
			},
			"cache_read_input_tokens":3,
			"output_tokens":7
		}
	}`

	converted, err := AnthropicToOpenAI([]byte(body), "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	details := response.Usage.PromptTokensDetails
	if details == nil {
		t.Fatal("usage.prompt_tokens_details must be present")
	}
	if details.CacheWriteTokens != 120 {
		t.Fatalf("cache_write_tokens must be the TTL total, got %d", details.CacheWriteTokens)
	}
	if details.CacheCreationTokenDetails == nil {
		t.Fatal("cache_creation_token_details must be present")
	}
	if details.CacheCreationTokenDetails.Ephemeral5mInputTokens != 100 ||
		details.CacheCreationTokenDetails.Ephemeral1hInputTokens != 20 {
		t.Fatalf("the TTL split must stay as it was, got %+v", details.CacheCreationTokenDetails)
	}
	// Same split again under Alibaba/Qwen's unsuffixed spelling
	// (cache_creation.ephemeral_*), for a client whose accounting reads that
	// name instead of cache_creation_token_details.
	if details.AlibabaCacheCreation == nil {
		t.Fatal("cache_creation (Alibaba spelling) must be present")
	}
	if details.AlibabaCacheCreation.Ephemeral5mInputTokens != 100 ||
		details.AlibabaCacheCreation.Ephemeral1hInputTokens != 20 {
		t.Fatalf("the Alibaba-spelled TTL split must match, got %+v", details.AlibabaCacheCreation)
	}

	var raw struct {
		Usage struct {
			PromptTokensDetails struct {
				CacheCreation struct {
					Ephemeral5m int `json:"ephemeral_5m_input_tokens"`
					Ephemeral1h int `json:"ephemeral_1h_input_tokens"`
				} `json:"cache_creation"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(converted, &raw); err != nil {
		t.Fatalf("unmarshal raw OpenAI response: %v", err)
	}
	if raw.Usage.PromptTokensDetails.CacheCreation.Ephemeral5m != 100 ||
		raw.Usage.PromptTokensDetails.CacheCreation.Ephemeral1h != 20 {
		t.Fatalf("cache_creation.ephemeral_* must be in the body, got %s", converted)
	}
}

// With the flag at its default (off), the TTL split stays only under
// cache_creation_token_details — no Alibaba-spelled cache_creation field.
func TestAnthropicToOpenAIOmitsAlibabaCacheCreationWhenMirroringDisabled(t *testing.T) {
	body := `{
		"id":"msg_cache_ttl_disabled",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":12,
			"cache_creation":{
				"ephemeral_5m_input_tokens":100,
				"ephemeral_1h_input_tokens":20
			},
			"cache_read_input_tokens":3,
			"output_tokens":7
		}
	}`

	converted, err := AnthropicToOpenAI([]byte(body), "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	details := response.Usage.PromptTokensDetails
	if details == nil {
		t.Fatal("usage.prompt_tokens_details must be present")
	}
	if details.AlibabaCacheCreation != nil {
		t.Fatalf("cache_creation (Alibaba spelling) must be absent while mirroring is disabled, got %+v", details.AlibabaCacheCreation)
	}

	var raw struct {
		Usage struct {
			PromptTokensDetails map[string]json.RawMessage `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(converted, &raw); err != nil {
		t.Fatalf("unmarshal raw OpenAI response: %v", err)
	}
	if _, exists := raw.Usage.PromptTokensDetails["cache_creation"]; exists {
		t.Fatalf("cache_creation must be absent from the body while mirroring is disabled, got %s", converted)
	}
}
