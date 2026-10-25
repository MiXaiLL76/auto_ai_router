package anthropic

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const nativeAdaptiveModel = "MiniMax-M3.1-Flash"

var nativeAdaptiveEffortLevels = []string{"low", "medium", "high", "xhigh", "max"}

func TestIsNativeAdaptiveModel(t *testing.T) {
	for _, model := range []string{"MiniMax-M3.1-Flash", "minimax-m3.1-flash", "minimax/minimax-m3.1-flash", "MiniMax-M3.1-Flash-Preview"} {
		assert.True(t, isNativeAdaptiveModel(model), model)
		assert.True(t, isAdaptiveThinkingModel(model), model)
		assert.Equal(t, thinkingAlwaysOn, thinkingOffModeFor(model), model)
		assert.Nil(t, thinkingOff(model, ""), model)
	}
	for _, model := range []string{"MiniMax-M3", "minimax-m2.7", "claude-sonnet-5.5", "gpt-6-luna"} {
		assert.False(t, isNativeAdaptiveModel(model), model)
	}
}

func TestValidateThinkingControls(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		api       RequestAPI
		body      string
		wantParam string // "" = accepted
	}{
		{"chat no controls", nativeAdaptiveModel, ChatAPI, `{}`, ""},
		{"chat minimal", nativeAdaptiveModel, ChatAPI, `{"reasoning_effort":"minimal"}`, ""},
		{"chat adaptive", nativeAdaptiveModel, ChatAPI, `{"thinking":{"type":"adaptive"},"reasoning_effort":"high"}`, ""},
		{"chat none", nativeAdaptiveModel, ChatAPI, `{"reasoning_effort":"none"}`, "reasoning_effort"},
		{"chat disable", nativeAdaptiveModel, ChatAPI, `{"reasoning_effort":"disable"}`, "reasoning_effort"},
		{"chat nested none", nativeAdaptiveModel, ChatAPI, `{"reasoning":{"effort":"none"}}`, "reasoning_effort"},
		{"chat extra_body nested none", nativeAdaptiveModel, ChatAPI, `{"extra_body":{"reasoning":{"effort":"none"}}}`, "reasoning_effort"},
		{"chat unknown effort", nativeAdaptiveModel, ChatAPI, `{"reasoning_effort":"ultra"}`, "reasoning_effort"},
		{"chat disabled", nativeAdaptiveModel, ChatAPI, `{"thinking":{"type":"disabled"}}`, "thinking.type"},
		{"chat extra_body disabled", nativeAdaptiveModel, ChatAPI, `{"extra_body":{"thinking":{"type":"disabled"}}}`, "thinking.type"},
		{"chat top-level adaptive wins over extra_body", nativeAdaptiveModel, ChatAPI, `{"thinking":{"type":"adaptive"},"extra_body":{"thinking":{"type":"disabled"}}}`, ""},
		{"chat enabled budget", nativeAdaptiveModel, ChatAPI, `{"thinking":{"type":"enabled","budget_tokens":2048}}`, "thinking.type"},
		{"chat effort none inside thinking", nativeAdaptiveModel, ChatAPI, `{"thinking":{"type":"adaptive","effort":"none"},"reasoning_effort":"high"}`, "reasoning_effort"},
		{"messages adaptive max", nativeAdaptiveModel, MessagesAPI, `{"thinking":{"type":"adaptive"},"output_config":{"effort":"max"}}`, ""},
		{"messages effort without thinking", nativeAdaptiveModel, MessagesAPI, `{"output_config":{"effort":"low"}}`, ""},
		{"messages thinking effort", nativeAdaptiveModel, MessagesAPI, `{"thinking":{"type":"adaptive","effort":"xhigh"}}`, ""},
		{"messages disabled", nativeAdaptiveModel, MessagesAPI, `{"thinking":{"type":"disabled"},"output_config":{"effort":"low"}}`, "thinking.type"},
		{"messages enabled", nativeAdaptiveModel, MessagesAPI, `{"thinking":{"type":"enabled","budget_tokens":4096}}`, "thinking.type"},
		{"messages between_tools", nativeAdaptiveModel, MessagesAPI, `{"thinking":{"type":"between_tools"}}`, "thinking.type"},
		{"messages none", nativeAdaptiveModel, MessagesAPI, `{"output_config":{"effort":"none"}}`, "output_config.effort"},
		{"messages minimal", nativeAdaptiveModel, MessagesAPI, `{"output_config":{"effort":"minimal"}}`, "output_config.effort"},
		{"responses xhigh", nativeAdaptiveModel, ResponsesAPI, `{"reasoning":{"effort":"xhigh"}}`, ""},
		{"responses minimal", nativeAdaptiveModel, ResponsesAPI, `{"reasoning":{"effort":"minimal","summary":"auto"}}`, ""},
		{"responses none", nativeAdaptiveModel, ResponsesAPI, `{"reasoning":{"effort":"none"}}`, "reasoning.effort"},
		{"public model id", "minimax/minimax-m3.1-flash", MessagesAPI, `{"thinking":{"type":"disabled"}}`, "thinking.type"},
		{"malformed body", nativeAdaptiveModel, ChatAPI, `{"reasoning_effort":`, ""},
		// Other models keep their own behaviour.
		{"other minimax disabled", "MiniMax-M3", MessagesAPI, `{"thinking":{"type":"disabled"}}`, ""},
		{"other minimax none", "MiniMax-M3", ChatAPI, `{"reasoning_effort":"none"}`, ""},
		{"claude disabled", "claude-opus-4-6", MessagesAPI, `{"thinking":{"type":"disabled"}}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateThinkingControls([]byte(tt.body), tt.model, tt.api)
			if tt.wantParam == "" {
				assert.NoError(t, err)
				return
			}
			var validationErr *converterutil.RequestValidationError
			require.True(t, errors.As(err, &validationErr), "want a RequestValidationError, got %v", err)
			assert.Equal(t, tt.wantParam, validationErr.Param)
		})
	}
	for _, effort := range nativeAdaptiveEffortLevels {
		for _, api := range []RequestAPI{ChatAPI, MessagesAPI, ResponsesAPI} {
			body := map[RequestAPI]string{
				ChatAPI:      `{"reasoning_effort":"` + effort + `"}`,
				MessagesAPI:  `{"thinking":{"type":"adaptive"},"output_config":{"effort":"` + effort + `"}}`,
				ResponsesAPI: `{"reasoning":{"effort":"` + effort + `"}}`,
			}[api]
			assert.NoError(t, ValidateThinkingControls([]byte(body), nativeAdaptiveModel, api), "%s on api %d", effort, api)
		}
	}
}

func TestValidateServiceTier(t *testing.T) {
	for _, tier := range []string{"", "auto", "default", "standard", "standard_only", "Auto"} {
		assert.NoError(t, ValidateServiceTier(tier, nativeAdaptiveModel), tier)
	}
	for _, tier := range []string{"priority", "flex", "scale"} {
		err := ValidateServiceTier(tier, "minimax/minimax-m3.1-flash")
		var validationErr *converterutil.RequestValidationError
		require.True(t, errors.As(err, &validationErr), tier)
		assert.Equal(t, "service_tier", validationErr.Param)
	}
	assert.NoError(t, ValidateServiceTier("priority", "MiniMax-M3"))
	assert.NoError(t, ValidateServiceTier("priority", "gpt-6-luna"))
}

func decodeAnthropicRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &request))
	return request
}

// Every effort reaches the model unchanged as adaptive thinking, with no budget and no
// raised max_tokens; without one the request carries no thinking controls at all, so
// the model keeps its default effort.
func TestOpenAIToAnthropic_NativeAdaptiveEffort(t *testing.T) {
	for _, effort := range nativeAdaptiveEffortLevels {
		t.Run(effort, func(t *testing.T) {
			body := `{"model":"minimax-m3.1-flash","max_tokens":1000,"reasoning_effort":"` + effort + `","messages":[{"role":"user","content":"hi"}]}`
			converted, err := OpenAIToAnthropic([]byte(body), nativeAdaptiveModel, true)
			require.NoError(t, err)
			request := decodeAnthropicRequest(t, converted)
			assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"])
			assert.Equal(t, map[string]interface{}{"effort": effort}, request["output_config"])
			assert.Equal(t, float64(1000), request["max_tokens"])
		})
	}

	t.Run("minimal maps to low", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"reasoning_effort":"minimal","messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), nativeAdaptiveModel, true)
		require.NoError(t, err)
		assert.Equal(t, map[string]interface{}{"effort": "low"}, decodeAnthropicRequest(t, converted)["output_config"])
	})

	t.Run("no effort keeps the model default", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), nativeAdaptiveModel, true)
		require.NoError(t, err)
		request := decodeAnthropicRequest(t, converted)
		assert.NotContains(t, request, "thinking")
		assert.NotContains(t, request, "output_config")
	})

	t.Run("no effort behind a gateway does not turn thinking off", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), nativeAdaptiveModel, false)
		require.NoError(t, err)
		assert.NotContains(t, decodeAnthropicRequest(t, converted), "thinking")
	})

	t.Run("adaptive thinking without effort", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), nativeAdaptiveModel, true)
		require.NoError(t, err)
		request := decodeAnthropicRequest(t, converted)
		assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"])
		assert.NotContains(t, request, "output_config")
		assert.Equal(t, float64(1000), request["max_tokens"])
	})

	// Regression guard: models outside the rule keep the budget mapping.
	t.Run("other minimax model keeps budget thinking", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), "MiniMax-M3", true)
		require.NoError(t, err)
		request := decodeAnthropicRequest(t, converted)
		assert.Equal(t, map[string]interface{}{"type": "enabled", "budget_tokens": float64(30000)}, request["thinking"])
		assert.Equal(t, float64(31024), request["max_tokens"])
	})

	t.Run("claude keeps summarized display", func(t *testing.T) {
		body := `{"model":"m","max_tokens":1000,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`
		converted, err := OpenAIToAnthropic([]byte(body), "claude-opus-4-7", true)
		require.NoError(t, err)
		assert.Equal(t, map[string]interface{}{"type": "adaptive", "display": "summarized"}, decodeAnthropicRequest(t, converted)["thinking"])
	})
}

func TestMessagesToChat_NativeAdaptiveEffort(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		controls   string
		wantEffort interface{} // nil = no reasoning_effort in the Chat body
	}{
		{"adaptive with output_config effort", "minimax-m3.1-flash", `"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}`, "xhigh"},
		{"output_config effort without thinking", "minimax-m3.1-flash", `"output_config":{"effort":"low"}`, "low"},
		{"effort inside thinking", "minimax-m3.1-flash", `"thinking":{"type":"adaptive","effort":"max"}`, "max"},
		{"adaptive without effort keeps the default", "minimax-m3.1-flash", `"thinking":{"type":"adaptive"}`, nil},
		{"nothing", "minimax-m3.1-flash", ``, nil},
		// Regression guard: other non-Claude models keep the old adaptive -> medium mapping.
		{"other model adaptive", "minimax-m3", `"thinking":{"type":"adaptive"}`, "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controls := tt.controls
			if controls != "" {
				controls += ","
			}
			body := `{"model":"` + tt.model + `","max_tokens":1000,` + controls + `"messages":[{"role":"user","content":"hi"}]}`
			chatBody, _, err := MessagesToChat([]byte(body))
			require.NoError(t, err)
			chat := decodeAnthropicRequest(t, chatBody)
			assert.NotContains(t, chat, "thinking")
			if tt.wantEffort == nil {
				assert.NotContains(t, chat, "reasoning_effort")
			} else {
				assert.Equal(t, tt.wantEffort, chat["reasoning_effort"])
			}
		})
	}

	t.Run("round trip back to adaptive", func(t *testing.T) {
		body := `{"model":"minimax-m3.1-flash","max_tokens":1000,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
		chatBody, _, err := MessagesToChat([]byte(body))
		require.NoError(t, err)
		converted, err := OpenAIToAnthropic(chatBody, nativeAdaptiveModel, true)
		require.NoError(t, err)
		request := decodeAnthropicRequest(t, converted)
		assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"])
		assert.Equal(t, map[string]interface{}{"effort": "high"}, request["output_config"])
		assert.Equal(t, float64(1000), request["max_tokens"])
	})
}

func TestNormalizeMessagesForPassthrough_NativeAdaptive(t *testing.T) {
	t.Run("adaptive stays adaptive", func(t *testing.T) {
		for _, effort := range nativeAdaptiveEffortLevels {
			body := `{"model":"MiniMax-M3.1-Flash","max_tokens":1000,"thinking":{"type":"adaptive"},"output_config":{"effort":"` + effort + `"},"messages":[]}`
			normalized, err := NormalizeMessagesForPassthrough([]byte(body), nativeAdaptiveModel, true)
			require.NoError(t, err)
			request := decodeAnthropicRequest(t, normalized)
			assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"], effort)
			assert.Equal(t, map[string]interface{}{"effort": effort}, request["output_config"], effort)
			assert.Equal(t, float64(1000), request["max_tokens"], effort)
			assert.NotContains(t, request, "anthropic_beta", effort)
		}
	})

	t.Run("effort inside thinking moves to output_config", func(t *testing.T) {
		body := `{"model":"MiniMax-M3.1-Flash","max_tokens":1000,"thinking":{"type":"adaptive","effort":"high"},"messages":[]}`
		normalized, err := NormalizeMessagesForPassthrough([]byte(body), nativeAdaptiveModel, true)
		require.NoError(t, err)
		request := decodeAnthropicRequest(t, normalized)
		assert.Equal(t, map[string]interface{}{"type": "adaptive"}, request["thinking"])
		assert.Equal(t, map[string]interface{}{"effort": "high"}, request["output_config"])
		assert.Equal(t, float64(1000), request["max_tokens"])
	})

	t.Run("effort without thinking passes as sent", func(t *testing.T) {
		body := `{"model":"MiniMax-M3.1-Flash","max_tokens":1000,"output_config":{"effort":"low"},"messages":[]}`
		normalized, err := NormalizeMessagesForPassthrough([]byte(body), nativeAdaptiveModel, true)
		require.NoError(t, err)
		assert.JSONEq(t, body, string(normalized))
	})

	t.Run("no controls pass as sent", func(t *testing.T) {
		body := `{"model":"MiniMax-M3.1-Flash","max_tokens":1000,"messages":[]}`
		normalized, err := NormalizeMessagesForPassthrough([]byte(body), nativeAdaptiveModel, false)
		require.NoError(t, err)
		assert.JSONEq(t, body, string(normalized))
	})
}

// A Messages usage with output_tokens_details.thinking_tokens, as the provider sends it.
const nativeAdaptiveMessagesResponse = `{
	"id": "msg_1", "type": "message", "role": "assistant", "model": "MiniMax-M3.1-Flash",
	"content": [
		{"type": "thinking", "thinking": "Compare 9.11 and 9.9.", "signature": "sig"},
		{"type": "text", "text": "9.9 is bigger."}
	],
	"stop_reason": "end_turn",
	"usage": {"input_tokens": 100, "output_tokens": 100, "cache_read_input_tokens": 900,
		"output_tokens_details": {"thinking_tokens": 60}}
}`

func TestAnthropicToOpenAI_ThinkingTokensBecomeReasoningTokens(t *testing.T) {
	converted, err := AnthropicToOpenAI([]byte(nativeAdaptiveMessagesResponse), "minimax-m3.1-flash")
	require.NoError(t, err)
	var response openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &response))
	require.NotNil(t, response.Usage)
	assert.Equal(t, 1000, response.Usage.PromptTokens)
	assert.Equal(t, 100, response.Usage.CompletionTokens, "thinking is inside output_tokens, not added to it")
	require.NotNil(t, response.Usage.CompletionTokensDetails)
	assert.Equal(t, 60, response.Usage.CompletionTokensDetails.ReasoningTokens)
	require.NotNil(t, response.Usage.PromptTokensDetails)
	assert.Equal(t, 900, response.Usage.PromptTokensDetails.CachedTokens)
	assert.Equal(t, "Compare 9.11 and 9.9.", response.Choices[0].Message.ReasoningContent)
	assert.Equal(t, "9.9 is bigger.", response.Choices[0].Message.Content)

	without := strings.Replace(nativeAdaptiveMessagesResponse, `,
		"output_tokens_details": {"thinking_tokens": 60}`, "", 1)
	require.NotEqual(t, nativeAdaptiveMessagesResponse, without)
	converted, err = AnthropicToOpenAI([]byte(without), "minimax-m3.1-flash")
	require.NoError(t, err)
	var plain openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &plain))
	assert.Nil(t, plain.Usage.CompletionTokensDetails)
}

func TestTransformAnthropicStreamToOpenAI_ThinkingTokens(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":900}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Compare."}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"9.9"}}`,
		`data: {"type":"content_block_stop","index":1}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":100,"output_tokens_details":{"thinking_tokens":60}}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n")
	var output strings.Builder
	require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(stream), "minimax-m3.1-flash", &output))

	var usage *openai.OpenAIUsage
	var reasoning, text string
	for _, line := range strings.Split(output.String(), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			reasoning += choice.Delta.ReasoningContent
			text += choice.Delta.Content
		}
	}
	assert.Equal(t, "Compare.", reasoning)
	assert.Equal(t, "9.9", text)
	require.NotNil(t, usage)
	assert.Equal(t, 1000, usage.PromptTokens)
	assert.Equal(t, 100, usage.CompletionTokens)
	require.NotNil(t, usage.CompletionTokensDetails)
	assert.Equal(t, 60, usage.CompletionTokensDetails.ReasoningTokens)
}

func TestChatUsageToMessages_ReasoningTokensBecomeThinkingTokens(t *testing.T) {
	usage := chatUsageToMessages(&openai.OpenAIUsage{
		PromptTokens:            1000,
		CompletionTokens:        100,
		PromptTokensDetails:     &openai.TokenDetails{CachedTokens: 900},
		CompletionTokensDetails: &openai.CompletionTokenDetails{ReasoningTokens: 60},
	}, 0, 0)
	assert.Equal(t, 100, usage.InputTokens)
	assert.Equal(t, 900, usage.CacheReadInputTokens)
	assert.Equal(t, 100, usage.OutputTokens)
	assert.Equal(t, 60, usage.ReasoningTokens())

	encoded, err := json.Marshal(usage)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), `"output_tokens_details":{"thinking_tokens":60}`)

	assert.Nil(t, chatUsageToMessages(&openai.OpenAIUsage{PromptTokens: 1, CompletionTokens: 1}, 0, 0).OutputTokensDetails)
}

// Web search runs as the provider's server tool: the converter counts it once, from the
// provider's counter when there is one, from the server_tool_use blocks otherwise, and
// not at all when the model did not search.
func TestAnthropicToOpenAI_NativeAdaptiveWebSearchCount(t *testing.T) {
	response := func(blocks int, counter string) string {
		content := `{"type":"thinking","thinking":"t","signature":"s"}`
		for i := 0; i < blocks; i++ {
			content += `,{"type":"server_tool_use","id":"srvtoolu_` + string(rune('a'+i)) + `","name":"web_search","input":{"query":"q"}}` +
				`,{"type":"web_search_tool_result","tool_use_id":"srvtoolu_` + string(rune('a'+i)) + `","content":[]}`
		}
		content += `,{"type":"text","text":"answer","citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Example","cited_text":"x"}]}`
		return `{"id":"msg_1","type":"message","role":"assistant","model":"MiniMax-M3.1-Flash","content":[` + content +
			`],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5` + counter + `}}`
	}
	tests := []struct {
		name   string
		body   string
		wantWS int
	}{
		{"no search", response(0, ""), 0},
		{"one search, no counter", response(1, ""), 1},
		{"two searches, no counter", response(2, ""), 2},
		{"two searches with counter", response(2, `,"server_tool_use":{"web_search_requests":2}`), 2},
		{"counter wins over blocks", response(1, `,"server_tool_use":{"web_search_requests":3}`), 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converted, err := AnthropicToOpenAI([]byte(tt.body), "minimax-m3.1-flash")
			require.NoError(t, err)
			var resp openai.OpenAIResponse
			require.NoError(t, json.Unmarshal(converted, &resp))
			if tt.wantWS == 0 {
				assert.Nil(t, resp.Usage.ServerToolUse)
				return
			}
			require.NotNil(t, resp.Usage.ServerToolUse)
			assert.Equal(t, tt.wantWS, resp.Usage.ServerToolUse.WebSearchRequests)
		})
	}
}
