package anthropic

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIToAnthropic_AdaptiveThinkingDisplay(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "reasoning effort requests summary",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"reasoning_effort":"high"}`,
		},
		{
			name: "reasoning object requests summary",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"reasoning":{"effort":"high"}}`,
		},
		{
			name: "extra body reasoning requests summary",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"extra_body":{"reasoning":{"effort":"high"}}}`,
		},
		{
			name: "native display is preserved",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"thinking":{"type":"adaptive","effort":"high","display":"summarized"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := OpenAIToAnthropic([]byte(tt.body), "claude-opus-4-8", true)
			require.NoError(t, err)

			var request map[string]interface{}
			require.NoError(t, json.Unmarshal(result, &request))
			thinking := request["thinking"].(map[string]interface{})
			assert.Equal(t, "adaptive", thinking["type"])
			assert.Equal(t, "summarized", thinking["display"])
			assert.Equal(t, "high", request["output_config"].(map[string]interface{})["effort"])
		})
	}
}

func TestOpenAIToAnthropic_SamplingParamsDroppedForNewModels(t *testing.T) {
	// temperature / top_p / top_k must be stripped for models that reject them
	// (Claude Opus 4.7+), so an OpenAI-style temperature=0 does not 400 on an
	// Anthropic-wire route (CometAPI / Bedrock). Older models keep them.
	body := `{"model":"M","messages":[{"role":"user","content":"hi"}],"temperature":0,"top_p":0.5,"extra_body":{"top_k":10}}`

	dropped := []string{
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-sonnet-5", "claude-fable-5", "claude-fable-5-1",
		"claude-mythos-5",
		// Future minors/families must be covered automatically by the version parser,
		// without editing the classifier (this is the whole point of not hardcoding a list).
		"claude-opus-4-9", "claude-opus-4-10", "claude-haiku-5",
	}
	for _, m := range dropped {
		t.Run("dropped/"+m, func(t *testing.T) {
			result, err := OpenAIToAnthropic([]byte(body), m, false)
			require.NoError(t, err)
			var req map[string]interface{}
			require.NoError(t, json.Unmarshal(result, &req))
			_, hasTemp := req["temperature"]
			_, hasTopP := req["top_p"]
			_, hasTopK := req["top_k"]
			assert.False(t, hasTemp, "temperature must be dropped for %s", m)
			assert.False(t, hasTopP, "top_p must be dropped for %s", m)
			assert.False(t, hasTopK, "top_k must be dropped for %s", m)
		})
	}

	kept := []string{"claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"}
	for _, m := range kept {
		t.Run("kept/"+m, func(t *testing.T) {
			result, err := OpenAIToAnthropic([]byte(body), m, false)
			require.NoError(t, err)
			var req map[string]interface{}
			require.NoError(t, json.Unmarshal(result, &req))
			assert.Equal(t, float64(0), req["temperature"], "temperature must be preserved for %s", m)
			assert.Equal(t, 0.5, req["top_p"], "top_p must be preserved for %s", m)
			assert.Equal(t, float64(10), req["top_k"], "top_k (from extra_body) must be preserved for %s", m)
		})
	}
}

// TestSamplingRemoved locks the model-classification boundary directly, so the version
// parser cannot silently drift from the "Opus 4.7+" spec or from isAdaptiveThinkingModel.
func TestSamplingRemoved(t *testing.T) {
	drop := []string{
		// current shipping models
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-sonnet-5", "claude-fable-5", "claude-fable-5-1", "claude-mythos-5",
		// dotted (CometAPI) and platform/date-suffixed forms
		"claude-opus-4.7", "us.anthropic.claude-opus-4-7-20250101-v1:0",
		"global.anthropic.claude-opus-4-8",
		// future minors/families the parser must catch without a code edit
		"claude-opus-4-9", "claude-opus-4-10", "claude-haiku-5", "claude-sonnet-6",
		// bare / preview Mythos (no claude-...-5 form) — still new-gen
		"mythos-5", "mythos-5-preview", "claude-mythos-preview",
	}
	for _, m := range drop {
		assert.True(t, SamplingRemoved(m), "SamplingRemoved(%q) should be true", m)
	}

	keep := []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-sonnet-4-5",
		"claude-haiku-4-5", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
		"claude-opus-4-20250514", // bare 4.x base with a date suffix, not a real 4.6+ minor
		"gpt-5", "deepseek-v4-flash", "",
	}
	for _, m := range keep {
		assert.False(t, SamplingRemoved(m), "SamplingRemoved(%q) should be false", m)
	}
}

func TestOpenAIToAnthropic_ResponseFormatJSONSchemaIncludesSchema(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Where do you live?"}],
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"location_info",
				"schema":{
					"type":"object",
					"properties":{"city":{"type":"string"}},
					"required":["city"]
				},
				"strict":true
			}
		}
	}`), "claude-sonnet-4-6", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))

	// json_schema must go through native Structured Outputs (output_config.format),
	// enforced by the API, not a system-prompt suggestion the model can ignore.
	_, hasSystem := request["system"]
	assert.False(t, hasSystem, "json_schema must not fall back to a system-prompt instruction")

	format := request["output_config"].(map[string]interface{})["format"].(map[string]interface{})
	assert.Equal(t, "json_schema", format["type"])

	// The model must see the actual field name from the schema, not just a
	// generic "respond with JSON" instruction - otherwise it invents its own
	// field names (observed: "location" instead of the schema's "city").
	schema := format["schema"].(map[string]interface{})
	assert.Equal(t, []interface{}{"city"}, schema["required"])
	assert.Contains(t, schema["properties"], "city")
}

func TestOpenAIToAnthropic_ResponseFormatJSONObjectForbidsMarkdownFences(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"test"}],
		"response_format":{"type":"json_object"}
	}`), "claude-sonnet-4-6", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	system, ok := request["system"].(string)
	require.True(t, ok)
	assert.Contains(t, system, "no markdown code fences")
}

func TestOpenAIToAnthropic_ResponseFormatJSONObjectAppendsToExistingSystem(t *testing.T) {
	// Regression test: the JSON instruction used to be injected before the real
	// system message was assigned, so the unconditional `anthropicReq.System =
	// systemContent` a few lines later silently discarded it whenever the
	// request carried its own system message.
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"user","content":"test"}
		],
		"response_format":{"type":"json_object"}
	}`), "claude-sonnet-4-6", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	system, ok := request["system"].(string)
	require.True(t, ok)
	assert.Contains(t, system, "You are a helpful assistant.")
	assert.Contains(t, system, "no markdown code fences")
}

func TestOpenAIToAnthropic_ResponseFormatJSONSchemaPreservesExistingSystem(t *testing.T) {
	// json_schema now goes through output_config.format, not the system prompt,
	// so an existing system message must survive untouched.
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"user","content":"Where do you live?"}
		],
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"location_info",
				"schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
				"strict":true
			}
		}
	}`), "claude-sonnet-4-6", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	system, ok := request["system"].(string)
	require.True(t, ok)
	assert.Equal(t, "You are a helpful assistant.", system)
	assert.Equal(t, "json_schema", request["output_config"].(map[string]interface{})["format"].(map[string]interface{})["type"])
}

func TestOpenAIToAnthropic_ResponseFormatJSONSchemaMergesWithThinkingEffort(t *testing.T) {
	// Both live under output_config, so setting the schema format must not
	// clobber an effort value already set by adaptive thinking, or vice versa.
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"test"}],
		"reasoning_effort":"high",
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"location_info",
				"schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
				"strict":true
			}
		}
	}`), "claude-opus-5", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	outputConfig := request["output_config"].(map[string]interface{})
	assert.Equal(t, "high", outputConfig["effort"])
	assert.Equal(t, "json_schema", outputConfig["format"].(map[string]interface{})["type"])
}

func TestOpenAIToAnthropic_ReasoningEffortPrecedence(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"test"}],
		"reasoning_effort":"low",
		"extra_body":{"reasoning":{"effort":"medium"}},
		"reasoning":{"effort":"high"}
	}`), "claude-opus-5", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	assert.Equal(t, "high", request["output_config"].(map[string]interface{})["effort"])
}

func TestOpenAIToAnthropic_ReasoningObjectCanDisableTopLevelEffort(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"test"}],
		"reasoning_effort":"high",
		"reasoning":{"effort":"none"}
	}`), "claude-opus-5", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	assert.NotContains(t, request, "thinking")
	assert.NotContains(t, request, "output_config")
}

func TestOpenAIToAnthropic_ChatFileFileData(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0="}}]
		}]
	}`)

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.NoError(t, err)

	block := firstAnthropicUserBlock(t, result)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "application/pdf", source["media_type"])
	assert.Equal(t, "JVBERi0=", source["data"])
}

func TestOpenAIToAnthropic_ChatFileFileIDUnsupported(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"file","file":{"file_id":"file-abc"}}]
		}]
	}`)

	_, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "messages.content.file.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
	assert.NotContains(t, err.Error(), "Anthropic")
}

func TestOpenAIToAnthropic_ChatPDFAsImageURLRejected(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"image_url","image_url":{"url":"data:application/pdf;base64,JVBERi0="}}]
		}]
	}`)

	_, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "messages.content.image_url.url", validationErr.Param)
}

func TestOpenAIToAnthropic_ChatImageURLJPEGStillWorks(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQ"}}]
		}]
	}`)

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.NoError(t, err)

	block := firstAnthropicUserBlock(t, result)
	assert.Equal(t, "image", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "image/jpeg", source["media_type"])
	assert.Equal(t, "/9j/4AAQ", source["data"])
}

func TestOpenAIToAnthropic_ChatImageURLPNGStillWorks(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]
		}]
	}`)

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.NoError(t, err)

	block := firstAnthropicUserBlock(t, result)
	assert.Equal(t, "image", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "image/png", source["media_type"])
	assert.Equal(t, "iVBORw0KGgo=", source["data"])
}

func TestOpenAIToAnthropic_ChatNativeDocument(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]
		}]
	}`)

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5", true)
	require.NoError(t, err)

	block := firstAnthropicUserBlock(t, result)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "application/pdf", source["media_type"])
	assert.Equal(t, "JVBERi0=", source["data"])
}

// TestOpenAIToAnthropic_NonClaudeModelDefaultsThinkingDisabled verifies that
// requests for a non-Claude model (reached through this same Anthropic-shaped
// conversion path via a multi-vendor gateway like CometAPI/ProMan) get an
// explicit "thinking":{"type":"disabled"} when the caller didn't ask for
// reasoning. Unlike Claude, some backends (e.g. Gemini) default to
// autonomous thinking when the field is simply absent, which silently burns
// the max_tokens budget on invisible reasoning and truncates the visible
// answer.
func TestOpenAIToAnthropic_NonClaudeModelDefaultsThinkingDisabled(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"gemini-3.5-flash",
		"messages":[{"role":"user","content":"test"}],
		"max_tokens":100
	}`), "gemini-3.5-flash", false)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	thinking, ok := request["thinking"].(map[string]interface{})
	require.True(t, ok, "expected an explicit thinking config, got %v", request["thinking"])
	assert.Equal(t, "disabled", thinking["type"])
}

// TestOpenAIToAnthropic_ClaudeModelOmitsThinkingByDefault verifies the
// default-disable safeguard is scoped to non-Claude models only: real Claude
// requests keep omitting "thinking" entirely when not requested, matching
// existing Anthropic API behavior (see also
// TestOpenAIToAnthropic_ReasoningObjectCanDisableTopLevelEffort).
func TestOpenAIToAnthropic_ClaudeModelOmitsThinkingByDefault(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"test"}]
	}`), "claude-opus-5", true)
	require.NoError(t, err)

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &request))
	assert.NotContains(t, request, "thinking")
}

func TestExtractSystemBlocks(t *testing.T) {
	ephemeral := map[string]interface{}{"type": "ephemeral"}

	tests := []struct {
		name    string
		content interface{}
		want    []ContentBlock
	}{
		{
			name:    "string system prompt",
			content: "You are a helpful assistant.",
			want:    []ContentBlock{{Type: "text", Text: "You are a helpful assistant."}},
		},
		{
			name: "array with text blocks",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "First instruction."},
				map[string]interface{}{"type": "text", "text": "Second instruction."},
			},
			want: []ContentBlock{
				{Type: "text", Text: "First instruction."},
				{Type: "text", Text: "Second instruction."},
			},
		},
		{
			name: "cache_control preserved",
			content: []interface{}{
				map[string]interface{}{
					"type":          "text",
					"text":          "Cached instruction.",
					"cache_control": ephemeral,
				},
			},
			want: []ContentBlock{
				{Type: "text", Text: "Cached instruction.", CacheControl: ephemeral},
			},
		},
		{
			name: "mixed: one cached one plain",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "Plain."},
				map[string]interface{}{"type": "text", "text": "Cached.", "cache_control": ephemeral},
			},
			want: []ContentBlock{
				{Type: "text", Text: "Plain."},
				{Type: "text", Text: "Cached.", CacheControl: ephemeral},
			},
		},
		{
			name: "array with non-text blocks ignored",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "Keep this."},
				map[string]interface{}{"type": "image_url", "url": "https://example.com/img.png"},
			},
			want: []ContentBlock{{Type: "text", Text: "Keep this."}},
		},
		{
			name: "array with empty text included as empty block",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": ""},
			},
			want: []ContentBlock{{Type: "text", Text: ""}},
		},
		{
			name:    "nil returns nil",
			content: nil,
			want:    nil,
		},
		{
			name:    "empty string returns nil",
			content: "",
			want:    nil,
		},
		{
			name:    "non-string non-slice returns nil",
			content: 12345,
			want:    nil,
		},
		{
			name: "array with non-map elements ignored",
			content: []interface{}{
				"not a map",
				map[string]interface{}{"type": "text", "text": "Valid block."},
			},
			want: []ContentBlock{{Type: "text", Text: "Valid block."}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractSystemBlocks(tt.content)
			assert.Equal(t, tt.want, got)
		})
	}
}

func firstAnthropicUserBlock(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &request))
	messages := request["messages"].([]interface{})
	require.NotEmpty(t, messages)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.NotEmpty(t, content)
	return content[0].(map[string]interface{})
}
