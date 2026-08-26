package anthropic

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTruncateToolNamePreservesUTF8(t *testing.T) {
	// A multi-byte-rune name whose byte length exceeds maxOpenAIToolNameLength must
	// not be sliced mid-rune, which would produce invalid UTF-8.
	name := strings.Repeat("Привет", 20) // Cyrillic "Привет" x20, 2 bytes/rune
	truncated := truncateToolName(name)

	require.True(t, utf8.ValidString(truncated), "truncated name must be valid UTF-8: %q", truncated)
	require.LessOrEqual(t, len(truncated), maxOpenAIToolNameLength)
}

func TestMessagesToChat(t *testing.T) {
	longName := strings.Repeat("tool", 20)
	body := []byte(`{
		"model":"claude-alias",
		"max_tokens":512,
		"system":[{"type":"text","text":"Be concise","cache_control":{"type":"ephemeral"}}],
		"messages":[
			{"role":"assistant","content":[
				{"type":"text","text":"Calling "},
				{"type":"tool_use","id":"tool.1","name":"` + longName + `","input":{"q":"weather"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"tool.1","content":[{"type":"text","text":"sunny"}]},
				{"type":"text","text":"Continue"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}
			]}
		],
		"tools":[{"name":"` + longName + `","description":"Search","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"` + longName + `"},
		"stop_sequences":["STOP"],
		"metadata":{"user_id":"user-1"},
		"stream":true
	}`)

	converted, metadata, err := MessagesToChat(body)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))
	assert.Equal(t, "claude-alias", got["model"])
	assert.Equal(t, []interface{}{"STOP"}, got["stop"])
	assert.Equal(t, "user-1", got["user"])
	assert.Equal(t, map[string]interface{}{"include_usage": true}, got["stream_options"])
	require.Len(t, metadata.ToolNames, 1)

	messages := got["messages"].([]interface{})
	require.Len(t, messages, 4)
	assert.Equal(t, "system", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[1].(map[string]interface{})["role"])
	assert.Equal(t, "tool", messages[2].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[3].(map[string]interface{})["role"])

	tool := got["tools"].([]interface{})[0].(map[string]interface{})
	truncated := tool["function"].(map[string]interface{})["name"].(string)
	assert.Len(t, truncated, maxOpenAIToolNameLength)
	assert.Equal(t, longName, metadata.ToolNames[truncated])
	assert.Equal(t, truncated, got["tool_choice"].(map[string]interface{})["function"].(map[string]interface{})["name"])
}

// TestMessagesToChat_SystemRoleInMessagesArray covers clients that send the system
// prompt as a role:"system"/"developer" message inside "messages" instead of using the
// top-level "system" field the Messages API spec requires — accepted for compatibility
// rather than rejected with "unsupported message role".
func TestMessagesToChat_SystemRoleInMessagesArray(t *testing.T) {
	body := []byte(`{
		"model":"claude-opus-4-8",
		"max_tokens":512,
		"messages":[
			{"role":"system","content":"Be concise"},
			{"role":"user","content":"Hi"}
		]
	}`)

	converted, _, err := MessagesToChat(body)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))
	messages := got["messages"].([]interface{})
	require.Len(t, messages, 2)
	first := messages[0].(map[string]interface{})
	assert.Equal(t, "system", first["role"])
	assert.Equal(t, "Be concise", first["content"])
	assert.Equal(t, "user", messages[1].(map[string]interface{})["role"])
}

func TestMessagesToChat_DocumentProviderFileIDRejected(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{
			"role":"user",
			"content":[{
				"type":"document",
				"source":{"type":"file","file_id":"file_abc"}
			}]
		}]
	}`)

	_, _, err := MessagesToChat(body)

	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "messages.content.document.source.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
	assert.NotContains(t, err.Error(), "Anthropic")
}

func TestMessagesToChat_MalformedDocumentSourceRejected(t *testing.T) {
	tests := []struct {
		name      string
		source    string
		wantParam string
	}{
		{
			name:      "base64 missing data",
			source:    `{"type":"base64","media_type":"application/pdf"}`,
			wantParam: "messages.content.document.source.data",
		},
		{
			name:      "base64 missing media type",
			source:    `{"type":"base64","data":"JVBERi0="}`,
			wantParam: "messages.content.document.source.media_type",
		},
		{
			name:      "url missing url",
			source:    `{"type":"url"}`,
			wantParam: "messages.content.document.source.url",
		},
		{
			name:      "unsupported source type",
			source:    `{"type":"unknown"}`,
			wantParam: "messages.content.document.source.type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{
				"model":"claude-sonnet-4-5",
				"max_tokens":128,
				"messages":[{
					"role":"user",
					"content":[{
						"type":"document",
						"source":` + tt.source + `
					}]
				}]
			}`)

			_, _, err := MessagesToChat(body)

			require.Error(t, err)
			var validationErr *converterutil.RequestValidationError
			require.True(t, errors.As(err, &validationErr))
			assert.Equal(t, tt.wantParam, validationErr.Param)
		})
	}
}

func TestChatToMessages(t *testing.T) {
	longName := strings.Repeat("tool", 20)
	truncated := truncateToolName(longName)
	body := []byte(`{
		"id":"chatcmpl-1",
		"model":"model-alias",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":null,
				"reasoning_content":"thinking",
				"tool_calls":[{"id":"functions.weather:0","type":"function","function":{"name":"` + truncated + `","arguments":"{\"city\":\"Moscow\"}"}}]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{
			"prompt_tokens":120,
			"completion_tokens":8,
			"total_tokens":128,
			"prompt_tokens_details":{"cached_tokens":20,"cache_creation_tokens":10}
		}
	}`)

	converted, err := ChatToMessages(body, MessagesAdapterMetadata{ToolNames: map[string]string{truncated: longName}})
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))
	assert.Equal(t, "message", got["type"])
	assert.Equal(t, "assistant", got["role"])
	assert.Equal(t, "tool_use", got["stop_reason"])
	assert.Nil(t, got["stop_sequence"])
	content := got["content"].([]interface{})
	require.Len(t, content, 2)
	assert.Equal(t, "thinking", content[0].(map[string]interface{})["type"])
	tool := content[1].(map[string]interface{})
	assert.Equal(t, "tool_use", tool["type"])
	assert.Equal(t, "functions_weather_0", tool["id"])
	assert.Equal(t, longName, tool["name"])
	assert.Equal(t, "Moscow", tool["input"].(map[string]interface{})["city"])
	usage := got["usage"].(map[string]interface{})
	assert.Equal(t, float64(90), usage["input_tokens"])
	assert.Equal(t, float64(8), usage["output_tokens"])
	assert.Equal(t, float64(20), usage["cache_read_input_tokens"])
	assert.Equal(t, float64(10), usage["cache_creation_input_tokens"])
}

func TestMessagesToChatPreservesThinkingForAnthropicProvider(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet",
		"max_tokens":4096,
		"thinking":{"type":"enabled","budget_tokens":1024},
		"messages":[
			{"role":"assistant","content":[
				{"type":"thinking","thinking":"reasoning","signature":"signature"},
				{"type":"tool_use","id":"functions.weather:0","name":"weather","input":{"city":"Moscow"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"functions.weather:0","content":"sunny"}
			]}
		]
	}`)

	chat, _, err := MessagesToChat(body)
	require.NoError(t, err)
	roundTrip, err := OpenAIToAnthropic(chat, "claude-sonnet", true)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(roundTrip, &got))
	messages := got["messages"].([]interface{})
	assistant := messages[0].(map[string]interface{})["content"].([]interface{})
	assert.Equal(t, "thinking", assistant[0].(map[string]interface{})["type"])
	assert.Equal(t, "signature", assistant[0].(map[string]interface{})["signature"])
	assert.Equal(t, "functions_weather_0", assistant[1].(map[string]interface{})["id"])
	result := messages[1].(map[string]interface{})["content"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "functions_weather_0", result["tool_use_id"])
}

func TestMessagesDocumentBase64RoundTrip(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{
			"role":"user",
			"content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]
		}]
	}`)

	block := roundTripFirstUserBlock(t, body)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "application/pdf", source["media_type"])
	assert.Equal(t, "JVBERi0=", source["data"])
}

func TestMessagesDocumentURLRoundTrip(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{
			"role":"user",
			"content":[{"type":"document","source":{"type":"url","url":"https://example.com/document.pdf"}}]
		}]
	}`)

	block := roundTripFirstUserBlock(t, body)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "url", source["type"])
	assert.Equal(t, "https://example.com/document.pdf", source["url"])
}

func TestMessagesImageBase64RoundTrip(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{
			"role":"user",
			"content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]
		}]
	}`)

	block := roundTripFirstUserBlock(t, body)
	assert.Equal(t, "image", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "image/png", source["media_type"])
	assert.Equal(t, "iVBORw0KGgo=", source["data"])
}

func TestTransformChatStreamToMessages(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call.1","type":"function","function":{"name":"weather","arguments":""}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Moscow\"}"}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"model-alias","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToMessages(strings.NewReader(stream), &output, "fallback-model", MessagesAdapterMetadata{}))

	got := output.String()
	assert.Contains(t, got, "event: message_start")
	assert.Contains(t, got, `"id":"chatcmpl-1"`)
	assert.Contains(t, got, `"thinking":"think","type":"thinking_delta"`)
	assert.Contains(t, got, `"text":"hello","type":"text_delta"`)
	assert.Contains(t, got, `"id":"call_1","input":{},"name":"weather","type":"tool_use"`)
	assert.Contains(t, got, `"partial_json":"{\"city\":\"Moscow\"}"`)
	assert.Contains(t, got, `"stop_reason":"tool_use"`)
	assert.Contains(t, got, `"input_tokens":10`)
	assert.Contains(t, got, `"output_tokens":4`)
	assert.True(t, strings.HasSuffix(got, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
}

// TestMessagesToChatPreservesVideoBlock covers the round trip for Anthropic-compatible
// providers that accept video content. Anthropic itself has no video block, so a client
// sending one is deliberately targeting a provider that reads it; the intermediate chat
// body has to carry the block through unchanged, or the request reaches the provider with
// the attachment missing and the model answers about media it never saw.
func TestMessagesToChatPreservesVideoBlock(t *testing.T) {
	body := []byte(`{
		"model":"video-capable-model",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":[
				{"type":"text","text":"What happens in this video?"},
				{"type":"video","source":{"type":"url","url":"https://example.com/clip.mp4"}}
			]}
		]
	}`)

	converted, _, err := MessagesToChat(body)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))

	messages := got["messages"].([]interface{})
	require.Len(t, messages, 1)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, content, 2)

	video := content[1].(map[string]interface{})
	assert.Equal(t, "video", video["type"])
	source := video["source"].(map[string]interface{})
	assert.Equal(t, "url", source["type"])
	assert.Equal(t, "https://example.com/clip.mp4", source["url"])
}

// TestMessagesToChatDropsVideoBlockWithoutSource guards the conversion against a malformed
// block: without a source there is nothing to forward, and emitting an empty attachment
// would be worse than leaving the text alone.
func TestMessagesToChatDropsVideoBlockWithoutSource(t *testing.T) {
	body := []byte(`{
		"model":"video-capable-model",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":[
				{"type":"text","text":"Describe it"},
				{"type":"video"}
			]}
		]
	}`)

	converted, _, err := MessagesToChat(body)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))

	messages := got["messages"].([]interface{})
	require.Len(t, messages, 1)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, content, 1)
	assert.Equal(t, "text", content[0].(map[string]interface{})["type"])
}

// TestMessagesToChatDropsVideoBlockWithUnrecognizedSource guards against forwarding a video
// block that the later Anthropic-shape conversion (mediaSourceFromMap) can't reconstruct: a
// source shape that isn't url/base64 would otherwise round-trip through chat only to be
// silently dropped downstream instead of never being kept in the first place.
func TestMessagesToChatDropsVideoBlockWithUnrecognizedSource(t *testing.T) {
	body := []byte(`{
		"model":"video-capable-model",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":[
				{"type":"text","text":"Describe it"},
				{"type":"video","source":{"type":"file_id","file_id":"abc"}}
			]}
		]
	}`)

	converted, _, err := MessagesToChat(body)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(converted, &got))

	messages := got["messages"].([]interface{})
	require.Len(t, messages, 1)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, content, 1)
	assert.Equal(t, "text", content[0].(map[string]interface{})["type"])
}

func roundTripFirstUserBlock(t *testing.T, messagesBody []byte) map[string]interface{} {
	t.Helper()

	chat, _, err := MessagesToChat(messagesBody)
	require.NoError(t, err)
	roundTrip, err := OpenAIToAnthropic(chat, "claude-sonnet-4-5", true)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(roundTrip, &got))
	messages := got["messages"].([]interface{})
	require.NotEmpty(t, messages)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.NotEmpty(t, content)
	return content[0].(map[string]interface{})
}

// TestNormalizeMessagesForPassthrough_AdaptiveThinkingBeta covers the /v1/messages
// native-passthrough path: a client sending native Anthropic "thinking":{"type":"adaptive"}
// straight through still needs the effort-2025-11-24 beta and temperature=1.0 that
// OpenAIToAnthropic would otherwise have added during the Messages->Chat->Messages round trip.
func TestNormalizeMessagesForPassthrough_AdaptiveThinkingBeta(t *testing.T) {
	body := []byte(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"temperature":0.5,
		"messages":[{"role":"user","content":"hi"}],
		"thinking":{"type":"adaptive","effort":"high"},
		"anthropic_beta":["prompt-caching-2024-07-31"]
	}`)

	out, err := NormalizeMessagesForPassthrough(body, "claude-opus-4.7", true)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &got))
	assert.Equal(t, "adaptive", got["thinking"].(map[string]interface{})["type"])
	assert.Equal(t, 1.0, got["temperature"])
	assert.Equal(t,
		[]interface{}{"prompt-caching-2024-07-31", "effort-2025-11-24"},
		got["anthropic_beta"])
}

// TestNormalizeMessagesForPassthrough_LegacyBudgetUpgradedOnAdaptiveModel covers a client
// still sending the legacy Claude 3.x "enabled"+budget_tokens shape to a model that only
// accepts the adaptive format — must be upgraded the same way OpenAIToAnthropic does, or the
// real Anthropic API would reject it.
func TestNormalizeMessagesForPassthrough_LegacyBudgetUpgradedOnAdaptiveModel(t *testing.T) {
	body := []byte(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"thinking":{"type":"enabled","budget_tokens":20000}
	}`)

	out, err := NormalizeMessagesForPassthrough(body, "claude-opus-4.7", true)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &got))
	thinking := got["thinking"].(map[string]interface{})
	assert.Equal(t, "adaptive", thinking["type"])
	assert.Equal(t, "high", got["output_config"].(map[string]interface{})["effort"])
	assert.Equal(t, []interface{}{"effort-2025-11-24"}, got["anthropic_beta"])
}

// TestNormalizeMessagesForPassthrough_DisablesThinkingForNonAnthropicBackend guards the
// CometAPI/ProMan passthrough case: omitting "thinking" means off for real Claude models, but
// other vendors behind these multi-vendor gateways default to autonomous thinking, silently
// burning tokens and truncating the visible answer unless explicitly disabled.
func TestNormalizeMessagesForPassthrough_DisablesThinkingForNonAnthropicBackend(t *testing.T) {
	body := []byte(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hi"}]
	}`)

	out, err := NormalizeMessagesForPassthrough(body, "claude-opus-4.7", false)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &got))
	assert.Equal(t, "disabled", got["thinking"].(map[string]interface{})["type"])
}

// TestNormalizeMessagesForPassthrough_NoThinkingLeavesRealAnthropicBodyUntouched covers the
// common case: a plain request with no thinking config going to real Anthropic needs no
// normalization at all — omitting "thinking" already means off for Claude.
func TestNormalizeMessagesForPassthrough_NoThinkingLeavesRealAnthropicBodyUntouched(t *testing.T) {
	body := []byte(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hi"}]
	}`)

	out, err := NormalizeMessagesForPassthrough(body, "claude-opus-4.7", true)
	require.NoError(t, err)

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &got))
	assert.NotContains(t, got, "thinking")
}
