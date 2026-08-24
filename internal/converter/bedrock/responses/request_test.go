package bedrockresponses

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponsesRequestToBedrock_AddsAnthropicVersion(t *testing.T) {
	body := `{
		"model": "anthropic.claude-opus-4-7",
		"input": [{"role": "user", "content": "Hello"}],
		"max_output_tokens": 100
	}`

	result, err := ResponsesRequestToBedrock([]byte(body), "anthropic.claude-opus-4-7")
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &req))

	assert.Equal(t, "bedrock-2023-05-31", req["anthropic_version"])
	assert.Nil(t, req["model"], "model field must be absent (goes in URL path)")
	assert.Nil(t, req["stream"], "stream field must be absent")
	assert.NotNil(t, req["messages"])
	assert.EqualValues(t, 100, req["max_tokens"])
}

func TestResponsesRequestToBedrock_StripStreamField(t *testing.T) {
	body := `{
		"model": "anthropic.claude-opus-4-7",
		"stream": true,
		"input": [{"role": "user", "content": "Hi"}]
	}`

	result, err := ResponsesRequestToBedrock([]byte(body), "anthropic.claude-opus-4-7")
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &req))

	assert.Nil(t, req["stream"], "stream field must be stripped for Bedrock")
	assert.Equal(t, "bedrock-2023-05-31", req["anthropic_version"])
}

func TestResponsesRequestToBedrock_InputFileFileIDUnsupported(t *testing.T) {
	body := `{
		"model": "anthropic.claude-opus-4-7",
		"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file-abc"}]}]
	}`

	_, err := ResponsesRequestToBedrock([]byte(body), "anthropic.claude-opus-4-7")

	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_file.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
}

func TestIsAnthropicBedrockModel(t *testing.T) {
	cases := []struct {
		modelID  string
		expected bool
	}{
		{"anthropic.claude-opus-4-7", true},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", true},
		{"us.anthropic.claude-3-sonnet-20240229-v1:0", true},
		{"eu.anthropic.claude-3-haiku-20240307-v1:0", true},
		{"meta.llama3-8b-instruct-v1:0", false},
		{"amazon.titan-text-express-v1", false},
		{"mistral.mistral-7b-instruct-v0:2", false},
	}

	for _, tc := range cases {
		t.Run(tc.modelID, func(t *testing.T) {
			assert.Equal(t, tc.expected, isAnthropicBedrockModel(tc.modelID))
		})
	}
}
