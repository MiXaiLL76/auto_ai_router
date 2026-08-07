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
			result, err := OpenAIToAnthropic([]byte(tt.body), "claude-opus-4-8")
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

func TestOpenAIToAnthropic_ReasoningEffortPrecedence(t *testing.T) {
	result, err := OpenAIToAnthropic([]byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"test"}],
		"reasoning_effort":"low",
		"extra_body":{"reasoning":{"effort":"medium"}},
		"reasoning":{"effort":"high"}
	}`), "claude-opus-5")
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
	}`), "claude-opus-5")
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

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
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

	_, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "messages.content.file.file_id", validationErr.Param)
}

func TestOpenAIToAnthropic_ChatPDFAsImageURLRejected(t *testing.T) {
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[{"type":"image_url","image_url":{"url":"data:application/pdf;base64,JVBERi0="}}]
		}]
	}`)

	_, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
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

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
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

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
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

	result, err := OpenAIToAnthropic(body, "claude-sonnet-4-5")
	require.NoError(t, err)

	block := firstAnthropicUserBlock(t, result)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "application/pdf", source["media_type"])
	assert.Equal(t, "JVBERi0=", source["data"])
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
