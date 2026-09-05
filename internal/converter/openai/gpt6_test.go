package openai

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestReplaceBodyParamGPT6(t *testing.T) {
	for _, model := range []string{"gpt-6", "gpt-6-astra", "openai/gpt-6-astra", "OpenAI:GPT-6-ASTRA"} {
		t.Run(model, func(t *testing.T) {
			body := []byte(`{"max_tokens":100,"max_completion_tokens":200,"max_output_tokens":300,"temperature":0.5,"top_p":0.9,"logprobs":true,"top_logprobs":3,"reasoning_effort":"max","frequency_penalty":0.1}`)
			got := bodyToMap(t, ReplaceBodyParam(model, body))
			for _, key := range []string{"max_tokens", "temperature", "top_p", "logprobs", "top_logprobs"} {
				assert.NotContains(t, got, key)
			}
			assert.Equal(t, float64(200), got["max_completion_tokens"])
			assert.Equal(t, float64(300), got["max_output_tokens"])
			assert.Equal(t, "max", got["reasoning_effort"])
			assert.Equal(t, 0.1, got["frequency_penalty"])
			assert.Equal(t, float64(100), bodyToMap(t, ReplaceBodyParam(model, []byte(`{"max_tokens":100}`)))["max_completion_tokens"])
		})
	}
}

func TestReplaceResponsesBodyParamGPT6(t *testing.T) {
	for _, model := range []string{"gpt-6-astra", "openai/gpt-6-astra"} {
		t.Run(model, func(t *testing.T) {
			body := []byte(`{"temperature":0.5,"top_p":0.9,"top_logprobs":3,"max_output_tokens":300,"reasoning":{"effort":"max","mode":"pro"},"include":["message.output_text.logprobs","reasoning.encrypted_content"],"prompt_cache_options":{"ttl":"30m","mode":"explicit"},"seed":9223372036854775807}`)
			result := ReplaceResponsesBodyParam(model, body)
			got := bodyToMap(t, result)
			for _, key := range []string{"temperature", "top_p", "top_logprobs"} {
				assert.NotContains(t, got, key)
			}
			assert.Equal(t, float64(300), got["max_output_tokens"])
			assert.Equal(t, []any{"reasoning.encrypted_content"}, got["include"])
			assert.Equal(t, map[string]any{"effort": "max", "mode": "pro"}, got["reasoning"])
			assert.Equal(t, map[string]any{"ttl": "30m", "mode": "explicit"}, got["prompt_cache_options"])
			assert.Contains(t, string(result), `9223372036854775807`)
			assert.NotContains(t, bodyToMap(t, ReplaceResponsesBodyParam(model, []byte(`{"include":["message.output_text.logprobs"]}`))), "include")
			assert.Equal(t, []byte(`invalid`), ReplaceResponsesBodyParam(model, []byte(`invalid`)))
		})
	}
}

func TestGPT6RulesLeaveOtherModelsUnchanged(t *testing.T) {
	body := []byte(`{"temperature":0.5,"include":["message.output_text.logprobs"]}`)
	for _, model := range []string{"gpt-5.6-sol", "gpt-4o", "gpt-60", "my-gpt-6-astra"} {
		assert.Equal(t, body, ReplaceResponsesBodyParam(model, body), model)
	}
	for _, model := range []string{"gpt-60", "my-gpt-6-astra"} {
		assert.Equal(t, body, ReplaceBodyParam(model, body), model)
	}
}
