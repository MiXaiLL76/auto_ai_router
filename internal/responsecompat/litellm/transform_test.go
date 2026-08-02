package litellm

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTransformChatCompletion(t *testing.T) {
	transformer := New()
	result := transformer.Transform(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
		RequestID:      "request-1",
	}, Response{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":                   {"application/json"},
			"X-Ratelimit-Remaining-Requests": {"7"},
			"X-Provider-Request-Id":          {"provider-1"},
		},
		Body: []byte(`{
			"id": "",
			"created": null,
			"model": "provider-model",
			"choices": [{
				"index": 9,
				"message": {
					"role": null,
					"content": null,
					"refusal": null,
					"tool_calls": [{"id":"call-1","type":"function","function":{"name":"f","arguments":"{}"}}],
					"unused": null
				},
				"finish_reason": "stop",
				"content_filter_results": {"hate":{"filtered":false,"severity":"safe"}},
				"logprobs": null
			}],
			"usage": {"prompt_tokens": 2, "completion_tokens": null, "total_tokens": null}
		}`),
	})

	require.Equal(t, http.StatusOK, result.StatusCode)
	assert.Equal(t, "request-1", result.Headers.Get("x-litellm-call-id"))
	assert.Equal(t, "public-model", result.Headers.Get("x-litellm-model-name"))
	assert.Equal(t, "provider-1", result.Headers.Get("llm_provider-x-provider-request-id"))
	assert.Equal(t, "7", result.Headers.Get("x-ratelimit-remaining-requests"))
	assert.Empty(t, result.Headers.Get("X-Provider-Request-Id"))

	var body map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &body))
	assert.NotEmpty(t, body["id"])
	assert.NotNil(t, body["created"])
	assert.Equal(t, "chat.completion", body["object"])
	assert.Equal(t, "public-model", body["model"])

	choice := body["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, float64(0), choice["index"])
	assert.Equal(t, "tool_calls", choice["finish_reason"])
	assert.NotContains(t, choice, "logprobs")
	assert.NotContains(t, choice, "content_filter_results")
	choiceProviderFields := choice["provider_specific_fields"].(map[string]any)
	assert.Contains(t, choiceProviderFields, "content_filter_results")

	message := choice["message"].(map[string]any)
	assert.Equal(t, "assistant", message["role"])
	assert.Contains(t, message, "content")
	assert.Nil(t, message["content"])
	assert.NotContains(t, message, "unused")
	assert.NotContains(t, message, "refusal")
	messageProviderFields := message["provider_specific_fields"].(map[string]any)
	assert.Contains(t, messageProviderFields, "refusal")
	assert.Nil(t, messageProviderFields["refusal"])

	usage := body["usage"].(map[string]any)
	assert.Equal(t, float64(0), usage["completion_tokens"])
	assert.Equal(t, float64(0), usage["total_tokens"])
}

func TestTransformModelsUsesLiteLLMModelOwner(t *testing.T) {
	result := New().Transform(Context{Endpoint: "/v1/models"}, Response{
		StatusCode: http.StatusOK,
		Headers:    make(http.Header),
		Body:       []byte(`{"object":"list","data":[{"id":"openai/gpt-4.1-mini","object":"model","owned_by":"system"}]}`),
	})

	require.Equal(t, http.StatusOK, result.StatusCode)
	var body map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &body))
	model := body["data"].([]any)[0].(map[string]any)
	assert.Equal(t, "openai", model["owned_by"])
}

func TestTransformMissingChoices(t *testing.T) {
	result := New().Transform(Context{
		Endpoint:       "/v1/chat/completions",
		RequestedModel: "public-model",
	}, Response{
		StatusCode: http.StatusOK,
		Headers:    make(http.Header),
		Body:       []byte(`{"model":"provider-model","choices":[]}`),
	})

	assert.Equal(t, http.StatusInternalServerError, result.StatusCode)
	var body map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &body))
	errorBody := body["error"].(map[string]any)
	assert.Contains(t, errorBody["message"], "no 'choices'")
	assert.Equal(t, "500", errorBody["code"])
}

func TestTransformError(t *testing.T) {
	result := New().Transform(Context{}, Response{
		StatusCode: http.StatusTooManyRequests,
		Headers:    http.Header{"Content-Type": {"application/json"}},
		Body:       []byte(`{"error":{"message":"provider throttled","type":"provider_error","code":"quota"}}`),
	})

	assert.Equal(t, http.StatusTooManyRequests, result.StatusCode)
	var body map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &body))
	errorBody := body["error"].(map[string]any)
	assert.Equal(t, "provider throttled", errorBody["message"])
	assert.Equal(t, "provider_error", errorBody["type"])
	assert.Equal(t, "429", errorBody["code"])
	assert.Contains(t, errorBody, "param")
}

func TestTransformTextCompletion(t *testing.T) {
	result := New().Transform(Context{
		Endpoint:       "/v1/completions",
		RequestedModel: "public-model",
	}, Response{
		StatusCode: http.StatusOK,
		Headers:    make(http.Header),
		Body:       []byte(`{"model":"provider-model","choices":[{"index":4,"text":"hello","finish_reason":null}]}`),
	})

	require.Equal(t, http.StatusOK, result.StatusCode)
	var body map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &body))
	assert.Equal(t, "text_completion", body["object"])
	choice := body["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, "hello", choice["text"])
	assert.Equal(t, "stop", choice["finish_reason"])
	assert.NotContains(t, choice, "message")
}
